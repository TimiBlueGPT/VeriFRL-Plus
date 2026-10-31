# VeriFRL+: Contribution-Aware Aggregation for Federated Cross-Domain Sequential Recommendation

VeriFRL+ implements contribution-aware aggregation for federated cross-domain sequential recommendation. It supports data-size weighting, square-root contribution weighting with a uniform floor, and EMA contribution weighting.

English | [中文](README_zh.md)

## Environment

```bash
conda create -n verifrl python=3.12.6
conda activate verifrl
pip install -r requirements.txt
```

Versions are pinned in [requirements.txt](requirements.txt). The local FKCB diagnostic used Python 3.12.6, NumPy 1.26.4, PyTorch 2.8.0+cu129 and an RTX 5070 Ti. This records the tested environment rather than guaranteeing identical results on other hardware.

## Layout

```text
.
├── data/                # Four input files per domain
├── models/              # VeriFRL and the original baseline architectures
├── utils/               # Data, RNG, saving and influence utilities
├── main.py              # Unified training/evaluation entry point
├── client.py            # Local training, online contribution scoring, evaluation
├── server.py            # The three aggregation policies
├── fl.py                # Federated training, selection and incremental records
├── checkpoint/          # Per-run config, metrics, best model, optional snapshots
├── log/                 # Plain-text run logs
└── requirements.txt
```

Core changes use the existing source files. Training has no dependency on experiment wrappers, old results or machine-specific paths. Relative data/model/log paths resolve from the project directory. `.gitignore` excludes data and run artifacts from source commits.

## Data

Data preparation follows [VeriFRL](https://github.com/TimiBlueGPT/VeriFRL). Each domain needs only:

```text
data/Food/
├── num_items.txt
├── train_data.txt
├── valid_data.txt
└── test_data.txt
```

FKCB means `Food`, `Kitchen`, `Clothing`, `Beauty`. No `tracin_data.txt`, `item_users.txt` or preprocessing cache is required. Online scoring selects one fixed reference batch per domain from TRAIN (256 rows by default, or all rows if fewer), using `--probe_seed` (default42). Those rows remain in full training and in the training graph. The legacy internal name `tracin` refers to this in-memory reference set and reads `train_data.txt`.

## Training and evaluation

Each command below runs FKCB once with seed42, full TRAIN and evaluation every round. All three skip post-training influence analysis. Their distinct ids prevent cross-policy collisions; training refuses an existing `result.json` and does not automatically resume it.

```bash
# Default: square-root contributions + 30% uniform floor; scoring on; no EMA.
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence

# Data-size aggregation; skip online scoring as well.
python main.py Food Kitchen Clothing Beauty --aggregation data_size --disable_online_scoring --id data42 --disable_influence

# Historical EMA + z-score + softmax; online scoring stays enabled.
python main.py Food Kitchen Clothing Beauty --aggregation ema_softmax --ema_beta 0.9 --aggregation_temperature 1.0 --id ema42 --disable_influence
```

| Run id | Config, results and best-model directory | Log |
| --- | --- | --- |
| `sqrt42` | `checkpoint/domain_FKCB/VeriFRL_Fed_sqrt_floor_sqrt42/` | `log/domain_FKCB/VeriFRL_Fed_sqrt_floor_sqrt42.log` |
| `data42` | `checkpoint/domain_FKCB/VeriFRL_Fed_data_size_data42/` | `log/domain_FKCB/VeriFRL_Fed_data_size_data42.log` |
| `ema42` | `checkpoint/domain_FKCB/VeriFRL_Fed_ema_softmax_ema42/` | `log/domain_FKCB/VeriFRL_Fed_ema_softmax_ema42.log` |

Evaluate the first run's saved best model with the same data and evaluation settings:

```bash
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence --do_eval
```

`--do_eval` performs no training, preserves the original metrics file and appends to the corresponding log. It checks data fingerprints and domain order and requires the saved best model. Metrics-only diagnostic archives cannot be evaluated this way.

Paths are configurable through `--data_dir`, `--checkpoint_dir` and `--log_dir`. See `python main.py --help` and [main.py](main.py) for all arguments.

## Aggregation and scoring

`--aggregation` selects one of three policies:

- `sqrt_floor` (default): normalize the square roots of the positive parts of current-round scores, then combine 70% contribution weights with a 30% uniform floor. No EMA. All nonpositive scores produce uniform weights; nonfinite scores raise an error. Online scoring is required.
- `data_size`: preserve the original weights based on sample counts after training-batch padding. `--disable_online_scoring` is supported. Weights renormalize among participating clients when participation is partial.
- `ema_softmax`: restore historical EMA contribution weighting. Initialize each client from its first score, then update `EMA = beta * previous_EMA + (1-beta) * current_score`. Apply population z-score and softmax among participants and use those weights directly, without a data-size mixture or 30% floor. Signed scores are retained; EMA-score standard deviation at/below the threshold gives uniform weights. Online scoring is required.

The selected weights aggregate both shared-encoder parameters and the last training batch's uploaded shared representations. There is no averaging across all training batches or user alignment. Other original model methods automatically use `data_size` with online scoring disabled.

EMA options are `--ema_beta 0.9` (finite, 0≤beta<1; zero disables smoothing), `--aggregation_temperature 1.0` (finite and positive) and `--std_epsilon 1e-12` (finite, nonnegative absolute standard-deviation threshold). Aggregation temperature is separate from the model's `--temperature`. With partial participation, only active clients update their EMA; absent clients retain it and normalization covers active clients only. The EMA consumes current scores and prior per-client EMA, never cumulative attribution. The default square-root policy remains unsmoothed.

Online contribution scoring uses TRAIN references. Post-training influence analysis is separate: it uses validation gradients and a training subset for the Hessian approximation. `--influence_train_ratio` controls that subset, not the online reference size.

## Defaults and reproducibility

| Setting | Current default or rule |
| --- | --- |
| Training | seed42, at most30 rounds, three local epochs, batch256 |
| Optimizer | Adam, learning rate0.001, betas(0.9,0.98), default LR multiplier1 |
| Optimizer compatibility | The retained `--weight_decay` value is not passed to the current constructor; effective weight decay is0 |
| Selection/stopping | Strictly higher validation macro MRR selects the best model; five consecutive non-improving evaluations stop training. Reaching the round limit is not early stopping |
| FKCB client training order | Food → Clothing → Beauty → Kitchen; `--training_order` accepts a fixed permutation |
| Loading, initialization, aggregation order | Command-line domain order; shared parameters aggregate on CPU, representations on their original device |
| Online references | One fixed TRAIN batch per domain, size follows `--batch_size`, probe_seed42 |
| Validation/test | Fixed candidates every round, eval_seed20261021, 999 negatives sampled with replacement excluding only the target; strict-greater ranking |
| Other | Four CPU threads; periodic snapshots and anomaly detection off by default |

Keep the domain argument order consistent. Use `--training_order` to change training order alone. Specify `--anneal_cap 0.01` when following the original MBG/SGH setting; FKCB defaults to1.0.

Candidates, initialization, preprocessing, training and scoring use separate random streams. Scoring/evaluation preserve training state and RNG state. Evaluation uses no-grad and batched ranking; attention masks are created on the input device, and evaluation skips the training-only Exclusive prediction. `--load_prep` is a compatibility flag and reads/writes no preprocessing cache. `--detect_anomaly` re-enables the original repository's gradient anomaly detection.

A fixed seed does not guarantee bitwise-identical training. Global PyTorch deterministic algorithms are not enabled, and CUDA `index_select` backward nondeterminism was reproduced locally. Initialization/RNG streams, scoring and aggregation order also differ from legacy experiment1, so exact historical metrics or stopping rounds are not claimed. A single data-size run scoring higher does not establish a stable advantage over `sqrt_floor`.

## Saved artifacts

Each run saves `config.json`, atomically updated per-round `result.json` and one `best_roundN.pt`. Records include validation/test metrics, losses, contribution scores and applied weights; scores are `null` when disabled. For EMA, each `weight_trace` entry also contains `smoothed_scores`, `zscores`, beta, aggregation temperature, the standard-deviation threshold and normalization diagnostics. These settings are included in the configuration and best-model metadata. The previous best model is removed only after the new model and results are saved.

Best models support evaluation but contain no optimizer/RNG resume state. Best-model restoration differences remain recorded rather than interrupting work solely over small numeric differences.

`--ckpt_interval N` explicitly enables `round_NNN/client*.pt` periodic model snapshots, which also are not complete resumable training states. Post-training influence analysis requires those snapshots and omission of `--disable_influence`; it writes `influence_scores.json` and does not control online aggregation.

## Local diagnostic records

The following archive predates the EMA option and retains its original source fingerprints/results. The newly added EMA option has not been validated by training.

The 2026-10-31 FKCB data-size diagnostic completed three seed42 repeats with fixed FCBK training order and three with random client order each round, all with online scoring off. Mean test MRR ± sample SD was `0.148829 ± 0.002289` and `0.149362 ± 0.001333`, respectively. These are same-seed repeats, not cross-seed robustness estimates. The random-order diagnostic used temporary code that has been removed; the production CLI does not expose that experiment mode.

Local lightweight records remain in `checkpoint/seed42_order_diagnosis/`: `findings.txt` indexes the archive, `summary.csv` contains results, `comparison.png` plots curves, and `runs/` retains original logs and per-round records. One-time test weights and temporary code were removed; see `cleanup.json`. Original model paths remain as historical provenance and no longer identify loadable models. This Git-ignored archive is optional, is not a runtime dependency and is absent from a fresh clone.

## Code provenance

This project builds on the original VeriFRL implementation. The citation below is retained from the original repository:

```
@inproceedings{tang2026verifiable,
  title={Verifiable Federated Representation Learning for Cross-domain Sequential Recommendation},
  author={Tang, Tao and Liu, Botao and Peng, Ciyuan and Lee, Ivan and Kong, Xiangjie},
  booktitle={Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year={2026},
  location={Dubai, United Arab Emirates},
  publisher={ACM}
}
```

## License and contact

[Apache License 2.0](LICENSE). For VeriFRL-Plus questions, open an Issue in this repository. The original code lists `tao.tang@ieee.org` as its author contact.
