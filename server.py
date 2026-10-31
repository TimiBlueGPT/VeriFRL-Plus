# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import torch


class ClientAttributor:
    def __init__(self):
        self.scores = {}

    def add_score(self, cid, score):
        self.scores[cid] = self.scores.get(cid, 0.) + float(score)

    def dump_topk(self, k=20, T=1500.):
        if not self.scores:
            return []
        temperature = T * math.log1p(sum(self.scores.values())) if sum(self.scores.values()) > 1e6 else T
        maximum = max(self.scores.values())
        values = {cid:math.exp((v-maximum)/temperature) for cid,v in self.scores.items()}
        total = math.fsum(values.values())
        return sorted(((cid,v/total) for cid,v in values.items()), key=lambda row:row[1], reverse=True)[:k]


class Server:
    def __init__(self, args, init_global_params):
        self.args = args
        self.global_params = [{k:v.detach().cpu().clone() for k,v in branch.items()} for branch in init_global_params]
        self.global_reps = None
        self.attributor = ClientAttributor()
        self.current_scores = {}
        self.ema_scores = {}
        self.weight_record = None

    def begin_round(self):
        self.current_scores = {}

    def add_eval_score(self, cid, score):
        if score is not None:
            if not math.isfinite(float(score)):
                raise ValueError('Nonfinite contribution score')
            self.current_scores[cid] = float(score)
            self.attributor.add_score(cid,score)

    def _ema_weights(self, members):
        """Historical EMA -> population z-score -> stable softmax, without a prior/floor."""
        beta = self.args.ema_beta
        temperature = self.args.aggregation_temperature
        epsilon = self.args.std_epsilon
        if (not math.isfinite(beta) or not 0 <= beta < 1
                or not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(epsilon) or epsilon < 0):
            raise ValueError('Invalid EMA beta, aggregation temperature or std epsilon')
        smooth = {}
        for cid in members:
            value = self.current_scores[cid]
            if not math.isfinite(value):
                raise ValueError('Nonfinite contribution score')
            prior = self.ema_scores.get(cid)
            # An unseen client starts from its first score, with no zero bias.
            # Absent clients retain their prior EMA until they participate again.
            if prior is None or beta == 0:
                smooth[cid] = value
            else:
                scale = max(abs(prior), abs(value))
                smooth[cid] = scale*(beta*(prior/scale)+(1-beta)*(value/scale)) if scale else 0.
            if not math.isfinite(smooth[cid]):
                raise ValueError('Nonfinite smoothed contribution score')
        scale = max(abs(v) for v in smooth.values())
        scaled = {cid: smooth[cid]/scale if scale else 0. for cid in members}
        mean = math.fsum(scaled.values())/len(members)
        std = math.sqrt(math.fsum((v-mean)**2 for v in scaled.values())/len(members))
        uniform = scale == 0 or std == 0 or std <= epsilon/scale
        z = {cid: 0. if uniform else (v-mean)/std for cid, v in scaled.items()}
        maximum = max(z.values())
        exponentials = {cid: math.exp((v-maximum)/temperature) for cid, v in z.items()}
        total = math.fsum(exponentials.values())
        q = {cid: value/total for cid, value in exponentials.items()}
        diagnostics = {'scale':scale, 'mean_scaled':mean, 'std_scaled':std, 'uniform':uniform}
        if (any(not math.isfinite(v) or v < 0 for v in q.values())
                or not math.isclose(math.fsum(q.values()), 1., rel_tol=0, abs_tol=1e-12)):
            raise ValueError('EMA aggregation weights are not a probability distribution')
        return smooth, z, q, diagnostics

    def update_weights(self, clients, members):
        members = sorted(members)
        policy = self.args.aggregation if self.args.method == 'VeriFRL_Fed' else 'data_size'
        if policy not in ('sqrt_floor','data_size','ema_softmax'):
            raise ValueError('Unknown aggregation policy')
        if policy in ('sqrt_floor','ema_softmax') and set(self.current_scores) != set(members):
            raise ValueError('Every participating client needs a current contribution score')
        # Preserve the source data-size policy's count after training-batch padding.
        counts = {cid:(len(clients[cid].train_dataloader.dataset) if policy == 'data_size'
                       else clients[cid].n_samples_train) for cid in members}
        total = sum(counts.values())
        data = {cid:counts[cid]/total for cid in members}
        q = None
        if policy == 'sqrt_floor':
            positive = {cid:max(self.current_scores[cid],0.) for cid in members}
            scale = max(positive.values())
            roots = {cid:math.sqrt(v/scale) if scale else 0. for cid,v in positive.items()}
            total_roots = math.fsum(roots.values())
            q = {cid:v/total_roots if total_roots else 1./len(members) for cid,v in roots.items()}
            actual = {cid:.7*v+.3/len(members) for cid,v in q.items()}
        elif policy == 'ema_softmax':
            smooth, z, q, diagnostics = self._ema_weights(members)
            actual = dict(q)
            self.ema_scores.update(smooth)
        else:
            actual = data
        self.weight_record = {'policy':policy,
            'raw_scores':{clients[cid].domain:self.current_scores.get(cid) for cid in members},
            'data_weights':{clients[cid].domain:v for cid,v in data.items()},
            'contribution_weights':None if q is None else {clients[cid].domain:v for cid,v in q.items()},
            'weights':{clients[cid].domain:v for cid,v in actual.items()}}
        if policy == 'ema_softmax':
            self.weight_record.update(
                ema_beta=self.args.ema_beta,
                aggregation_temperature=self.args.aggregation_temperature,
                std_epsilon=self.args.std_epsilon,
                smoothed_scores={clients[cid].domain:v for cid,v in smooth.items()},
                zscores={clients[cid].domain:v for cid,v in z.items()},
                normalizer_diagnostics=diagnostics)
        self.weights = actual
        return self.weight_record

    @staticmethod
    def weighted_state(states, weights, cpu=False):
        result = None
        with torch.no_grad():
            for cid,state in states.items():
                state = {k:v.detach().cpu() if cpu else v.detach() for k,v in state.items()}
                if result is None:
                    result = {k:v*weights[cid] for k,v in state.items()}
                else:
                    for key,value in state.items():
                        result[key] += weights[cid]*value
        return result

    def aggregate_params(self, clients, random_cids):
        members = sorted(random_cids)
        uploads = {cid:clients[cid].get_params() for cid in members}
        self.global_params = [self.weighted_state({cid:uploads[cid][i] for cid in members},
            self.weights,cpu=True) for i in range(len(self.global_params))]

    def aggregate_reps(self, clients, random_cids):
        # Preserve the historical multiply/add order and last-training-batch upload.
        states = {cid:{'representation':clients[cid].get_reps_shared()} for cid in sorted(random_cids)}
        self.global_reps = self.weighted_state(states,self.weights)['representation']

    def choose_clients(self, n_clients, ratio=1.):
        count = math.ceil(n_clients*ratio)
        canonical = [self.args.domains.index(domain) for domain in self.args.training_order]
        if count == n_clients:
            return canonical
        selected = set(np.random.permutation(n_clients)[:count].tolist())
        return [cid for cid in canonical if cid in selected]

    def get_global_params(self):
        return self.global_params

    def get_global_reps(self):
        return self.global_reps
