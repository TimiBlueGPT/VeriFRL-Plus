# -*- coding: utf-8 -*-
import copy
import logging
import math
from pathlib import Path
import time
import torch
from utils.io_utils import atomic_write, data_fingerprints
from utils.train_utils import preserve_random_state, stream_seed, LRDecay
from utils.influence_utils import compute_influence_for_clients


def evaluation_logging(eval_logs, round, weights=None, mode='valid'):
    macro = {key:sum(row[key] for row in eval_logs.values())/len(eval_logs)
             for key in next(iter(eval_logs.values()))}
    logging.info('Round %s %s macro: %s',round,mode,macro)
    return macro


def evaluate(clients, shared, round_index):
    result = {}
    for split in ('valid','test'):
        domains = {client.domain:client.evaluation(split,shared,round_index) for client in clients}
        result[split] = {'domains':domains,'macro':evaluation_logging(domains,round_index,mode=split)}
    return result


def load_and_eval_model(n_clients, clients, args):
    import json
    folder = Path(clients[0].get_method_checkpoint_path())
    saved_result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    checkpoint = (folder/saved_result['best_checkpoint']['path']).resolve()
    if checkpoint.parent != folder.resolve():
        raise ValueError('Best-model path is outside this run')
    selected = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if selected['data_sha256'] != data_fingerprints(args):
        raise ValueError('Saved model and current dataset differ')
    if selected['domains'] != args.domains:
        raise ValueError('Saved model domain order differs from requested data')
    for client in clients:
        client.model.load_state_dict(selected['models'][client.domain])
    return evaluate(clients,selected['shared'] if 'Fed' in args.method else None,selected['round'])


def run_fl(clients, server, args):
    if args.do_eval:
        return load_and_eval_model(len(clients),clients,args)
    folder = Path(clients[0].get_method_checkpoint_path())
    path = folder/'result.json'
    if path.exists():
        raise FileExistsError(f'Run already exists: {path}; select another --id')
    record = {'status':'running','config':vars(args).copy(),'domains':args.domains,
        'aggregation':args.aggregation,'data_sha256':data_fingerprints(args),
        'rounds_completed':0,'evaluations':{},
        'score_trace':[],'weight_trace':[],'training_losses':{},
        'candidate_sha256':{f'{c.domain}/{split}':getattr(getattr(c,split+'_dataloader').dataset,'candidate_sha256',None)
                            for c in clients for split in ('valid','test')},
        'sample_counts':{c.domain:{'raw':c.n_samples_train,'padded':len(c.train_dataloader.dataset)} for c in clients},
        'model_role':'validation-selected best model; no optimizer/RNG resume state'}
    atomic_write(path,record)
    start = time.perf_counter()
    best_score,best_round,bad_rounds = -math.inf,None,0
    lr_decay = LRDecay(args.lr,args.decay_epoch,args.optimizer,args.lr_decay,patience=args.ld_patience)
    try:
        for round_index in range(1,args.epochs+1):
            members = server.choose_clients(len(clients),args.frac)
            server.begin_round()
            losses = {}
            for cid in members:
                client = clients[cid]
                if 'Fed' in args.method:
                    client.set_global_params(server.get_global_params())
                    if args.method=='VeriFRL_Fed':
                        client.set_global_reps(server.get_global_reps())
                with preserve_random_state(stream_seed(args.seed,client.domain,'training_round',[args.method,round_index])):
                    client.train_epoch(round_index,args,global_params=server.global_params)
                losses[client.domain] = client.latest_train_loss
                if not math.isfinite(client.latest_train_loss):
                    raise RuntimeError(f'Nonfinite training loss in {client.domain}')
                server.add_eval_score(cid,client.latest_eval_score)
                record['score_trace'].append({'round':round_index,'domain':client.domain,
                    'raw_score':client.latest_eval_score,'cumulative_score':server.attributor.scores.get(cid)})
            weights = server.update_weights(clients,members)
            record['weight_trace'].append({'round':round_index,**weights})
            record['training_losses'][str(round_index)] = losses
            # Commit completed scoring before evaluation or model saving can fail.
            atomic_write(path,record)
            if 'Fed' in args.method:
                server.aggregate_params(clients,members)
                if args.method=='VeriFRL_Fed':
                    server.aggregate_reps(clients,members)
            if args.ckpt_interval and round_index%args.ckpt_interval==0:
                for client in clients:
                    client.save_round_checkpoint(round_index)
            if round_index%args.eval_interval!=0 and round_index!=args.epochs:
                record['rounds_completed'] = round_index
                atomic_write(path,record)
                continue
            metrics = evaluate(clients,server.global_params if 'Fed' in args.method else None,round_index)
            record['evaluations'][str(round_index)] = {'metrics':metrics}
            score = metrics['valid']['macro']['MRR']
            old_best = record.get('best_checkpoint',{}).get('path')
            if score>best_score:
                best_score,best_round,bad_rounds = score,round_index,0
                selected = {'round':round_index,'domains':args.domains,'aggregation':args.aggregation,
                    'config':vars(args).copy(),'data_sha256':record['data_sha256'],
                    'shared':[{k:v.detach().cpu().clone() for k,v in branch.items()} for branch in server.global_params],
                    'models':{c.domain:{k:v.detach().cpu().clone() for k,v in c.model.state_dict().items()} for c in clients}}
                best_name = f'best_round{round_index}.pt'
                atomic_write(folder/best_name,selected,tensor=True)
                record['best_checkpoint'] = {'path':best_name,'round':round_index}
                del selected
            else:
                bad_rounds += 1
            record.update(rounds_completed=round_index,best_validation_round=best_round,
                best_validation_mrr=best_score,elapsed_seconds=time.perf_counter()-start)
            atomic_write(path,record)
            if old_best and old_best != record['best_checkpoint']['path']:
                (folder/old_best).unlink(missing_ok=True)
            logging.info('ROUND %d/%d valid_MRR=%.9f best=%d patience=%d/%d',
                round_index,args.epochs,score,best_round,bad_rounds,args.es_patience)
            if bad_rounds>=args.es_patience:
                break
            lr_decay(round_index,{'MRR':score},clients)
        record['metrics'] = load_and_eval_model(len(clients),clients,args)
        original = record['evaluations'][str(best_round)]['metrics']
        differences = []
        for split in ('valid','test'):
            for scope in ('macro','domains'):
                domains = [None] if scope=='macro' else args.domains
                for domain in domains:
                    left = original[split][scope] if domain is None else original[split][scope][domain]
                    right = record['metrics'][split][scope] if domain is None else record['metrics'][split][scope][domain]
                    for key in left:
                        if left[key]!=right[key]:
                            differences.append({'split':split,'domain':domain or 'macro','metric':key,
                                'original':left[key],'restored':right[key],'delta':right[key]-left[key]})
        record.update(status='completed',elapsed_seconds=time.perf_counter()-start,
            restoration_comparison={'mode':'record_only','differences':differences},
            stop_reason='early_stopping' if bad_rounds>=args.es_patience else 'max_epochs')
        atomic_write(path,record)
        if args.method=='VeriFRL_Fed' and args.compute_influence and args.ckpt_interval:
            compute_influence_for_clients(clients,args)
        return record
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-start)
        atomic_write(path,record)
        raise
