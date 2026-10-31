# -*- coding: utf-8 -*-
import torch
from dataset import SeqDataset
from local_graph import LocalGraph


def load_dataset(args):
    datasets = {mode: [] for mode in ('train', 'valid', 'test', 'tracin')}
    graphs = []
    for domain in args.domains:
        model = 'DisenVGSAN' if args.method == 'VeriFRL_Fed' else args.method.replace('Local','').replace('Fed','')
        current = {mode: SeqDataset(domain, model, mode, args.max_seq_len,
                    load_prep=False, args=args) for mode in ('train','valid','test')}
        current['tracin'] = (SeqDataset(domain, model, 'tracin', args.max_seq_len, args=args)
            if args.method == 'VeriFRL_Fed' and not args.disable_online_scoring else current['valid'])
        for mode in datasets:
            datasets[mode].append(current[mode])
        graph = LocalGraph(args, domain, current['train'].num_items)
        graphs.append(graph.adj.to('cuda:' + args.gpu if args.cuda else 'cpu'))
    # Scoring uses the actual training graph, including the fixed TRAIN reference rows.
    return *(datasets[mode] for mode in ('train','valid','test','tracin')), graphs, list(graphs)


def init_clients_weight(clients):
    total = sum(client.n_samples_train for client in clients)
    for client in clients:
        client.train_weight = client.n_samples_train / total
        client.valid_weight = client.test_weight = 1 / len(clients)
