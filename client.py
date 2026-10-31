# -*- coding: utf-8 -*-
import os
import math
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import SeqDataloader
from utils.io_utils import ensure_dir
from utils.train_utils import preserve_random_state, stream_seed
from utils.io_utils import atomic_write


class Client:
    def __init__(self, model_fn, c_id, args, adj, adjs_tracin, train_dataset, valid_dataset, test_dataset,tracin_dataset):
        self.num_items = train_dataset.num_items
        self.domain = train_dataset.domain
        self.max_seq_len = args.max_seq_len
        self.trainer = model_fn(args, self.num_items, self.max_seq_len)
        self.model = self.trainer.model
        self.method = args.method
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = args.id if len(args.id) > 1 else "0" + args.id
        self._method_ckpt_path = os.path.join(
            self.checkpoint_dir,
            "domain_" + "".join([domain[0] for domain in args.domains]),
            self.method + "_" + args.aggregation + "_" + self.model_id)
        ensure_dir(self._method_ckpt_path, verbose=True)
        if args.method == "VeriFRL_Fed":
            self.z_s = self.trainer.z_s
            self.z_g = self.trainer.z_g
        self.c_id = c_id
        self.args = args
        self.adj = adj
        self.adjs_tracin = adjs_tracin

        self.train_dataloader = SeqDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = SeqDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = SeqDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)
        self.tracin_dataloader = SeqDataloader(
            tracin_dataset, batch_size=args.batch_size, shuffle=False
        )

        self.n_samples_train = train_dataset.raw_sample_count
        self.online_audit_enabled = self.method == "VeriFRL_Fed" and not args.disable_online_scoring
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.init_global_params = copy.deepcopy(self.get_params())
        self.latest_eval_score = None
        self._pre_train_state = None
        self._cached_eval_batch = None

    def train_epoch(self, round, args, global_params=None):
        self.audit_round = round
        self.trainer.model.train()
        audit_enabled = getattr(self, 'online_audit_enabled', True)
        self._pre_train_state = copy.deepcopy(self.trainer.model.state_dict()) if audit_enabled else None
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            for _, sessions in self.train_dataloader:
                if ("Fed" in args.method) and args.mu:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args,
                        global_params=global_params)
                else:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args)
                loss += batch_loss
                step += 1

            gc.collect()
        self.latest_eval_score = None
        if audit_enabled:
            delta_vector = self._gather_shared_param_deltas_vector()
            post_train_state = copy.deepcopy(self.trainer.model.state_dict())
            self.latest_eval_score = self._compute_eval_tracin_score(
                delta_vector, post_train_state)
        # The score has restored the trained parameters; this GPU snapshot is
        # no longer needed between clients/rounds.
        self._pre_train_state = None
        logging.info("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.epochs, self.c_id, loss / step))
        self.latest_train_loss = loss / step
        return self.n_samples_train

    def _get_shared_modules(self):
        modules = []
        if self.method == "VeriFRL_Fed":
            modules.append(self.model.encoder_s)
        elif "VGSAN" in self.method:
            modules.append(self.model.encoder)
        elif "SASRec" in self.method:
            modules.append(self.model.encoder)
        elif "VSAN" in self.method:
            modules.extend([self.model.encoder, self.model.decoder])
        elif "ContrastVAE" in self.method:
            modules.extend([self.model.encoder, self.model.decoder])
        elif "CL4SRec" in self.method:
            modules.append(self.model.encoder)
        elif "DuoRec" in self.method:
            modules.append(self.model.encoder)
        return modules

    def _gather_shared_param_deltas_vector(self):
        modules = self._get_shared_modules()
        if not modules or self.init_global_params is None:
            return None
        deltas = []
        for branch_idx, module in enumerate(modules):
            if branch_idx >= len(self.init_global_params):
                return None
            init_state = self.init_global_params[branch_idx]
            for name, param in module.named_parameters():
                init_tensor = init_state[name].to(param.device)
                deltas.append((param.detach() - init_tensor).reshape(-1))
        if not deltas:
            return None
        return torch.cat(deltas)

    def _gather_shared_grad_vector(self):
        modules = self._get_shared_modules()
        if not modules:
            return None
        grads = []
        for module in modules:
            for _, param in module.named_parameters():
                if param.grad is None:
                    grads.append(torch.zeros_like(param, device=param.device).reshape(-1))
                else:
                    grads.append(param.grad.detach().reshape(-1))
        if not grads:
            return None
        return torch.cat(grads)

    def _get_eval_batch(self):
        if self._cached_eval_batch is not None:
            return self._cached_eval_batch
        dataset = self.tracin_dataloader.dataset
        if len(dataset) == 0:
            return None
        eval_batch_size = min(self.args.batch_size, len(dataset))
        batch_sessions = []
        for idx in range(eval_batch_size):
            _, session = dataset[idx]
            batch_sessions.append(copy.deepcopy(session))
        if not batch_sessions:
            return None
        batch_sessions = list(zip(*batch_sessions))
        self._cached_eval_batch = tuple(np.array(x, dtype=np.int64)
                                        for x in batch_sessions)
        return self._cached_eval_batch

    def _compute_eval_tracin_score(self, delta_vector, post_train_state):
        # Scoring is observational: preserve training streams and transient state
        # even when the score callback raises. External paired-audit wrappers may
        # additionally seed their own reproducible analysis streams.
        model, trainer = self.trainer.model, self.trainer
        modes = [(module, module.training) for module in model.modules()]
        parameters = list(trainer.params)
        gradients = [p.grad for p in parameters]
        caches = {name: (hasattr(model, name), getattr(model, name, None)) for name in
                  ('item_index_s', 'item_index_e', 'item_graph_embs_s', 'item_graph_embs_e')}
        representations = [(value, value[0]) for name in ('z_s', 'z_g')
                           if (value := getattr(self, name, None)) is not None]
        step = trainer.step
        for p in parameters:
            p.grad = None
        try:
            with preserve_random_state(stream_seed(self.args.eval_seed, self.domain,
                    'audit_score', self.audit_round)):
                return self._compute_eval_tracin_score_impl(delta_vector, post_train_state)
        finally:
            model.load_state_dict(post_train_state)
            trainer.step = step
            for module, mode in modes:
                module.training = mode
            for container, value in representations:
                container[0] = value
            for name, (present, value) in caches.items():
                if present:
                    setattr(model, name, value)
                elif hasattr(model, name):
                    delattr(model, name)
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient

    def _compute_eval_tracin_score_impl(self, delta_vector, post_train_state):
        if delta_vector is None or self._pre_train_state is None:
            return None
        if delta_vector.numel() == 0:
            return None
        eval_batch = self._get_eval_batch()
        if eval_batch is None:
            return None
        if not self.trainer.optimizer.param_groups:
            return None
        lr = self.trainer.optimizer.param_groups[0].get("lr", None)
        if lr is None or lr == 0:
            return None
        saved_z_s = None
        if self.method == "VeriFRL_Fed" and self.z_s[0] is not None:
            saved_z_s = copy.deepcopy(self.z_s[0].detach().clone())
        prev_training_mode = self.trainer.model.training
        self.trainer.model.load_state_dict(self._pre_train_state)
        self.trainer.model.train()
        self.trainer.optimizer.zero_grad()
        loss = self.trainer.compute_loss(
            eval_batch, self.adjs_tracin, self.num_items, self.args,
            global_params=self.init_global_params,
            include_prox=False, update_state=False)
        loss.backward()
        grad_vector = self._gather_shared_grad_vector()
        self.trainer.optimizer.zero_grad()
        self.trainer.model.load_state_dict(post_train_state)
        if saved_z_s is not None:
            self.z_s[0] = saved_z_s
        if prev_training_mode:
            self.trainer.model.train()
        else:
            self.trainer.model.eval()
        if grad_vector is None or grad_vector.numel() == 0:
            return None
        g_c_vector = -delta_vector / lr

        return torch.dot(g_c_vector, grad_vector).item()

    @torch.no_grad()
    def evaluation(self, mode="valid", global_params=None, round_index=0):
        if mode not in ('valid', 'test'):
            raise ValueError('Evaluation mode must be valid or test')
        model = self.model
        modules = self._get_shared_modules()
        saved_shared = [{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for module in modules]
        modes = [(module, module.training) for module in model.modules()]
        cache_names = ('item_index_s','item_index_e','item_graph_embs_s','item_graph_embs_e')
        caches = {name:(hasattr(model,name),getattr(model,name,None)) for name in cache_names}
        ranks, losses = [], []
        try:
            with preserve_random_state(stream_seed(self.args.eval_seed, self.domain,
                    'utility_evaluation', round_index)):
                if global_params is not None:
                    for module,state in zip(modules,global_params):
                        module.load_state_dict(state)
                model.eval()
                if hasattr(model,'graph_convolution'):
                    model.graph_convolution(self.adj)
                loader = self.valid_dataloader if mode == 'valid' else self.test_dataloader
                for _,sessions in loader:
                    batch_ranks,batch_losses = self.trainer.test_batch(sessions, return_losses=True)
                    ranks.extend(batch_ranks)
                    losses.extend(batch_losses)
            if not ranks or any(not math.isfinite(value) for value in losses):
                raise RuntimeError('Empty or nonfinite evaluation')
            result = {'MRR':sum(1/r for r in ranks)/len(ranks),
                'HR@10':sum(r<=10 for r in ranks)/len(ranks),
                'NDCG@10':sum(1/math.log2(r+1) if r<=10 else 0. for r in ranks)/len(ranks),
                'NEG_CE':-sum(losses)/len(losses)}
            self._latest_eval_log = result
            return result
        finally:
            for module,state in zip(modules,saved_shared):
                module.load_state_dict(state)
            for module,mode_value in modes:
                module.training = mode_value
            for name,(present,value) in caches.items():
                if present:
                    setattr(model,name,value)
                elif hasattr(model,name):
                    delattr(model,name)

    def get_old_eval_log(self):
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}

    @ staticmethod
    def cal_test_score(predictions):
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
        return MRR/valid_entity, NDCG_5 / valid_entity, \
            NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / \
            valid_entity, HR_10 / valid_entity


    def get_grads(self):
        assert hasattr(self, "init_global_params"), \
            "`init_global_params` missing. Did you forget to call `set_global_params`?"
        current_params = self.get_params()
        grads = []
        for branch_idx in range(len(current_params)):
            branch_grads = {}
            for key in current_params[branch_idx]:
                branch_grads[key] = (current_params[branch_idx][key]
                                     - self.init_global_params[branch_idx][key])
            grads.append(branch_grads)
        return grads

    def get_client_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone())
            else:
                grads.append(torch.zeros_like(param))
        return grads

    def get_params(self):
        if self.method == "VeriFRL_Fed":
            return copy.deepcopy([self.model.encoder_s.state_dict()])
        elif "VGSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "SASRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "VSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "ContrastVAE" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "CL4SRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "DuoRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])

    def get_reps_shared(self):
        assert (self.method == "VeriFRL_Fed")
        return copy.deepcopy(self.z_s[0].detach())

    def set_global_params(self, global_params):
        assert (self.method in ["VeriFRL_Fed", "FedVGSAN", "FedSASRec", "FedVSAN",
                                "FedContrastVAE", "FedCL4SRec", "FedDuoRec"])
        self.init_global_params = copy.deepcopy(global_params)
        if self.method == "VeriFRL_Fed":
            self.model.encoder_s.load_state_dict(global_params[0])
        elif self.method == "FedVGSAN":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedSASRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedVSAN":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedContrastVAE":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedCL4SRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedDuoRec":
            self.model.encoder.load_state_dict(global_params[0])

    def set_global_reps(self, global_rep):
        assert (self.method == "VeriFRL_Fed")
        self.z_g[0] = copy.deepcopy(global_rep)

    def save_params(self):
        ckpt_filename = os.path.join(
            self._method_ckpt_path, "client%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        try:
            atomic_write(ckpt_filename, params, tensor=True)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(
            self._method_ckpt_path, "client%d.pt" % self.c_id)
        try:
            checkpoint = torch.load(ckpt_filename, map_location=self.trainer.device, weights_only=True)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)

    def get_round_checkpoint_dir(self, round_idx):
        return os.path.join(self._method_ckpt_path,
                            "round_{:03d}".format(round_idx))

    def save_round_checkpoint(self, round_idx):
        round_dir = self.get_round_checkpoint_dir(round_idx)
        ensure_dir(round_dir, verbose=False)
        ckpt_filename = os.path.join(round_dir, "client%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        try:
            atomic_write(ckpt_filename, params, tensor=True)
            logging.info("Saved round %d checkpoint for client %d to %s",
                         round_idx, self.c_id, ckpt_filename)
        except IOError:
            logging.warning("[ Warning: Saving checkpoint for client %d in round %d failed. Continuing anyway. ]",
                            self.c_id, round_idx)

    def load_params_from_path(self, ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path,
                                     map_location=self.trainer.device, weights_only=True)
        except IOError:
            logging.warning("[ Warning: Cannot load model from %s. Skipping. ]",
                            ckpt_path)
            return False
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)
        return True

    def get_method_checkpoint_path(self):
        return self._method_ckpt_path
