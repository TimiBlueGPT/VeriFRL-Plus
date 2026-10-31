# VeriFRL+: Contribution-Aware Aggregation for Federated Cross-Domain Sequential Recommendation

This repository provides the PyTorch implementation of **VeriFRL+: Contribution-Aware Aggregation for Federated Cross-Domain Sequential Recommendation**.

VeriFRL+ uses client contribution scores to guide the aggregation of shared representations and model parameters. It provides square-root contribution weighting with a uniform floor, alongside data-size weighting and EMA-based contribution weighting.

English | [中文](README_zh.md)

## Environment Setup

We recommend using Conda to create an isolated Python environment, then installing the dependencies below:

```bash
conda create -n verifrl python=3.12.6
conda activate verifrl
pip install -r requirements.txt
```

## Project Structure

```text
.
├── checkpoint/          # Saved models and results
├── data/                # Datasets
├── log/                 # Training logs
├── models/              # Model architectures
├── utils/               # Utilities
├── main.py              # Training and evaluation entry point
└── requirements.txt     # Dependencies
```

## Data Preparation

Data preparation follows [VeriFRL](https://github.com/TimiBlueGPT/VeriFRL). Datasets are not included in this repository. Organize each domain as follows:

```text
data/Food/
├── num_items.txt
├── train_data.txt
├── valid_data.txt
└── test_data.txt
```

FKCB consists of `Food`, `Kitchen`, `Clothing`, and `Beauty`. Online contribution scoring uses samples from the training set; no separate `tracin_data.txt` is needed.

## Getting Started

`main.py` handles federated training, validation, and testing. The following example runs the four-domain FKCB setting with the default contribution-aware aggregation:

```bash
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence
```

Use `--aggregation` to select a policy:

| Value | Policy |
| --- | --- |
| `sqrt_floor` (default) | Square-root weighting of positive contribution scores, mixed with a 30% uniform floor |
| `data_size` | Data-size weighting |
| `ema_softmax` | EMA-smoothed contribution scores, followed by z-score normalization and softmax |

Run the alternative policies:

```bash
python main.py Food Kitchen Clothing Beauty --aggregation data_size --disable_online_scoring --id data42 --disable_influence
python main.py Food Kitchen Clothing Beauty --aggregation ema_softmax --ema_beta 0.9 --aggregation_temperature 1.0 --id ema42 --disable_influence
```

These commands each run FKCB once with seed 42 and skip post-training influence analysis. Models and results are saved under `checkpoint/domain_FKCB/VeriFRL_Fed_<aggregation>_<id>/`, with logs under `log/domain_FKCB/`. Use a new `--id` for another run; existing results are not overwritten.

Evaluate the saved best model from the default run:

```bash
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence --do_eval
```

**Common arguments:**

- `--id`: Identifier used for the run's output directory and log.
- `--seed`: Random seed; defaults to `42`.
- `--aggregation`: Aggregation policy; defaults to `sqrt_floor`.
- `--disable_online_scoring`: Skip online contribution scoring when using `data_size`.
- `--disable_influence`: Skip post-training influence-function analysis.
- `--do_eval`: Evaluate a saved best model without training.

For the full parameter list and default values, run:

```bash
python main.py --help
```

## Original VeriFRL Reference

VeriFRL+ builds on our [VeriFRL](https://github.com/TimiBlueGPT/VeriFRL) implementation. The original work is cited below:

```bibtex
@inproceedings{tang2026verifiable,
  title={Verifiable Federated Representation Learning for Cross-domain Sequential Recommendation},
  author={Tang, Tao and Liu, Botao and Peng, Ciyuan and Lee, Ivan and Kong, Xiangjie},
  booktitle={Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year={2026},
  location={Dubai, United Arab Emirates},
  publisher={ACM}
}
```

## License

This project is licensed under the **Apache License 2.0**. 

## Contact

For any questions, please open an **Issue** or contact `tao.tang@ieee.org`.
