# -*- coding: utf-8 -*-
import os
import math
from pathlib import Path
import numpy as np
import random
import argparse
import torch
from trainer import ModelTrainer

import logging
from client import Client
from server import Server
from utils.data_utils import load_dataset, init_clients_weight
from utils.io_utils import save_config, ensure_dir
from fl import run_fl
from utils.train_utils import preserve_random_state, stream_seed


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(dest="domains", metavar="domains", nargs="*",
                        help="`Food Kitchen Clothing Beauty` or "
                        "`Movies Books Games` or `Sports Garden Home`")
    parser.add_argument("--load_prep", dest="load_prep", action="store_true",
                        default=False,
                        help="Compatibility flag; preprocessing is deterministic and in memory.")
    parser.add_argument("--max_seq_len", type=int,
                        default=16, help="maxisum sequence length")
    parser.add_argument("--method", type=str, default="VeriFRL_Fed",
                        help="method, possible are `VeriFRL_Fed`(ours), "
                        "`FedVGSAN`, `LocalVGSAN`, `FedSASRec`, "
                        "`LocalSASRec`, `FedVSAN`, `LocalVSAN`, "
                        "`FedContrastVAE`, `LocalContrastVAE`, `FedCL4SRec`, "
                        "`LocalCL4SRec`, `FedDuoRec`, `LocalDuoRec`")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of total training iterations.")
    parser.add_argument("--local_epoch", type=int, default=3,
                        help="Number of local training epochs.")
    parser.add_argument("--optimizer", choices=["sgd", "adagrad", "adam",
                                                "adamax"], default="adam",
                        help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Applies to sgd and adagrad.")
    parser.add_argument("--lr_decay", type=float, default=1,
                        help="Learning rate decay rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--decay_epoch", type=int, default=10,
                        help="Decay learning rate after this epoch.")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="Interval of evalution")
    parser.add_argument("--frac", type=float, default=1,
                        help="Fraction of participating clients")
    parser.add_argument("--mu", type=float, default=0,
                        help="hyper parameter for FedProx")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="Checkpoint Dir")
    parser.add_argument("--id", type=str, default="00",
                        help="Model ID under which to save models.")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--es_patience", type=int,
                        default=5, help="Early stop patience.")
    parser.add_argument("--ld_patience", type=int, default=1,
                        help="Learning rate decay patience.")
    parser.add_argument("--ckpt_interval", type=int, default=0,
                        help="Interval (in rounds) to snapshot client checkpoints.")
    parser.add_argument("--disable_influence", action="store_true",
                        help="Disable influence score computation after training.")
    parser.add_argument("--influence_train_ratio", type=float, default=0.05,
                        help="Ratio of training samples used to approximate the Hessian inverse.")
    parser.add_argument("--hvp_iterations", type=int, default=10,
                        help="Number of LiSSA iterations when approximating the inverse Hessian-vector product.")
    parser.add_argument("--hvp_damping", type=float, default=0.01,
                        help="Damping factor used in LiSSA iterations.")
    parser.add_argument("--hvp_scale", type=float, default=10.0,
                        help="Scaling factor used in LiSSA iterations.")
    parser.add_argument("--influence_branch_tail_ratio", type=float, default=1.0,
                        help="Portion of each branch's parameters (from the tail) "
                             "used when computing influence scores.")

    parser.add_argument("--anneal_cap", type=float, default=1.0, help="KL "
                        "annealing arguments for variantional method "
                        "(including ours). 1.0 for FKCB is the best, 0.01 for "
                        "MBG and SGH is the best")
    parser.add_argument("--total_annealing_step", type=int, default=10000)

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Contrastive arguments for contrastive method "
                        "(including ours)")
    parser.add_argument("--sim", type=str, default="dot")

    parser.add_argument('--aggregation', choices=('sqrt_floor','data_size','ema_softmax'),
                        default='sqrt_floor')
    parser.add_argument('--ema_beta', type=float, default=0.9,
                        help='EMA smoothing for ema_softmax only; 0 <= beta < 1.')
    parser.add_argument('--aggregation_temperature', type=float, default=1.0,
                        help='Positive softmax temperature for ema_softmax; distinct from model --temperature.')
    parser.add_argument('--std_epsilon', type=float, default=1e-12,
                        help='Nonnegative absolute EMA-score standard deviation threshold for uniform weights.')
    parser.add_argument('--disable_online_scoring', action='store_true')
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--training_order', nargs='+')
    parser.add_argument('--eval_seed', type=int, default=20261021)
    parser.add_argument('--probe_seed', type=int, default=42)
    parser.add_argument('--num_test_neg', type=int, default=999)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    # The source scoring objective uses one reference batch, matching the global representation.
    args.probe_size = args.batch_size
    if not args.domains:
        args.domains = ['Food','Kitchen','Clothing','Beauty']
    if len(args.domains) != len(set(args.domains)):
        parser.error('Domains must be distinct')
    orders = [(['Food','Kitchen','Clothing','Beauty'],['Food','Clothing','Beauty','Kitchen']),
              (['Movies','Books','Games'],['Games','Books','Movies']),
              (['Sports','Garden','Home'],['Garden','Sports','Home'])]
    if args.training_order is None:
        args.training_order = next((order for domains,order in orders if set(domains)==set(args.domains)),list(args.domains))
    if len(args.training_order)!=len(args.domains) or set(args.training_order)!=set(args.domains):
        parser.error('training_order must contain each domain exactly once')
    if args.method != 'VeriFRL_Fed':
        args.aggregation = 'data_size'
        args.disable_online_scoring = True
    if args.aggregation in ('sqrt_floor','ema_softmax') and args.disable_online_scoring:
        parser.error(f'{args.aggregation} requires online scoring; data_size supports disabling it')
    if not math.isfinite(args.ema_beta) or not 0 <= args.ema_beta < 1:
        parser.error('ema_beta must be finite and in [0,1)')
    if not math.isfinite(args.aggregation_temperature) or args.aggregation_temperature <= 0:
        parser.error('aggregation_temperature must be finite and positive')
    if not math.isfinite(args.std_epsilon) or args.std_epsilon < 0:
        parser.error('std_epsilon must be finite and nonnegative')
    if min(args.epochs,args.local_epoch,args.batch_size,args.probe_size,args.threads,args.eval_interval)<=0:
        parser.error('Epochs, batch size, probe size, threads and evaluation interval must be positive')
    if not 0 < args.frac <= 1:
        parser.error('frac must be in (0,1]')
    root = Path(__file__).resolve().parent
    for name in ('data_dir','log_dir','checkpoint_dir'):
        path = Path(getattr(args,name)).expanduser()
        setattr(args,name,str((root/path).resolve() if not path.is_absolute() else path.resolve()))
    args.cuda = args.cuda and not args.cpu
    assert (args.method in ["VeriFRL_Fed", "FedVGSAN", "LocalVGSAN", "FedSASRec",
                            "LocalSASRec", "FedVSAN", "LocalVSAN",
                            "FedContrastVAE", "LocalContrastVAE", "FedCL4SRec",
                            "LocalCL4SRec", "FedDuoRec", "LocalDuoRec"])
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(args):
    log_path = os.path.join(args.log_dir,
                            "domain_" + "".join([domain[0] for domain
                                                 in args.domains]))
    ensure_dir(log_path, verbose=True)

    model_id = args.id if len(args.id) > 1 else "0" + args.id
    log_file = os.path.join(log_path, args.method + "_" + args.aggregation + "_" + model_id + ".log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="a" if args.do_eval else "w"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def main():
    args = arg_parse()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.set_num_threads(args.threads)
    args.compute_influence = not args.disable_influence
    seed_everything(args)

    model_id = args.id if len(args.id)>1 else '0'+args.id
    run_folder = Path(args.checkpoint_dir)/('domain_'+''.join(d[0] for d in args.domains))/(args.method+'_'+args.aggregation+'_'+model_id)
    if not args.do_eval and (run_folder/'result.json').exists():
        raise FileExistsError(f'Run already exists: {run_folder}; select another --id')
    init_logger(args)

    train_datasets, valid_datasets, test_datasets,tracin_dataset, adjs, adjs_tracin = load_dataset(args)

    n_clients = len(args.domains)
    clients = []
    for c_id,domain in enumerate(args.domains):
        with preserve_random_state(stream_seed(args.seed,domain,'initialization',args.method)):
            clients.append(Client(ModelTrainer,c_id,args,adjs[c_id],adjs_tracin[c_id],
                train_datasets[c_id],valid_datasets[c_id],test_datasets[c_id],tracin_dataset[c_id]))
    init_clients_weight(clients)

    if not args.do_eval:
        save_config(args)

    server = Server(args, clients[0].get_params())

    if 'Fed' in args.method:
        for client in clients:
            client.set_global_params(server.get_global_params())
    run_fl(clients, server, args)


if __name__ == "__main__":
    main()
