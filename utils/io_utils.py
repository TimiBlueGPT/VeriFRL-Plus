# -*- coding: utf-8 -*-
import os
import json
import logging



def check_dir(dir):
    if not os.path.exists(dir):
        print("Directory {} does not exist. Exit.".format(dir))
        exit(1)


def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)


def ensure_dir(dir, verbose=True):
    if not os.path.exists(dir):
        if verbose:
            print("Directory {} do not exist; creating...".format(dir))
        os.makedirs(dir)




def save_config(args, verbose=True):
    model_id = args.id if len(args.id) > 1 else "0" + args.id
    method_ckpt_path = os.path.join(args.checkpoint_dir,
                                    "domain_" +
                                    "".join([domain[0]
                                            for domain in args.domains]),
                                    args.method + "_" + args.aggregation + "_" + model_id)
    ensure_dir(method_ckpt_path, verbose=True)

    config_filename = os.path.join(method_ckpt_path, "config.json")
    args_dict = vars(args)
    atomic_write(config_filename, vars(args))
    if verbose:
        print("Config saved to file {}.".format(config_filename))

    print_config(args_dict)


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    logging.info(info)


def atomic_write(path, value, tensor=False):
    """Replace a complete file only after the new contents reach disk."""
    from pathlib import Path
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if tensor else "w", **({} if tensor else {"encoding": "utf-8"})) as stream:
            if tensor:
                import torch
                torch.save(value, stream)
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def data_fingerprints(args):
    import hashlib
    from pathlib import Path
    return {f'{domain}/{name}':hashlib.sha256((Path(args.data_dir)/domain/name).read_bytes()).hexdigest()
        for domain in args.domains for name in ('num_items.txt','train_data.txt','valid_data.txt','test_data.txt')}
