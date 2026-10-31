# VeriFRL+: Contribution-Aware Aggregation for Federated Cross-Domain Sequential Recommendation

VeriFRL+ 面向联邦跨域序列推荐，支持数据量加权、开方贡献加均匀保底和 EMA 贡献加权三种聚合方式。

[English](README.md) | 中文

## 环境配置

```bash
conda create -n verifrl python=3.12.6
conda activate verifrl
pip install -r requirements.txt
```

依赖版本见 [requirements.txt](requirements.txt)。本地 FKCB 诊断使用 Python 3.12.6、NumPy 1.26.4、PyTorch 2.8.0+cu129 和 RTX 5070 Ti；这是已运行环境的记录，不代表所有硬件上的结果都完全一致。

## 项目结构

```text
.
├── data/                # 每个域的四个输入文件
├── models/              # VeriFRL 与原有基线模型
├── utils/               # 数据、随机状态、保存及影响函数工具
├── main.py              # 统一训练与评测入口
├── client.py            # 本地训练、在线贡献评分和评测
├── server.py            # 三种聚合策略
├── fl.py                # 联邦训练、选模和逐轮记录
├── checkpoint/          # 每次运行的配置、指标、最佳模型及可选周期快照
├── log/                 # 文本运行日志
└── requirements.txt
```

训练和聚合更新位于原有源码文件中，不依赖外接实验入口、旧结果或本机历史路径。相对数据、模型及日志路径均以项目目录为基准。数据和运行产物默认被 `.gitignore` 排除，不会随源码提交。

## 数据准备

沿用 [FedDCSR](https://github.com/orion-orion/FedDCSR) 的数据准备来源。每个域只需：

```text
data/Food/
├── num_items.txt
├── train_data.txt
├── valid_data.txt
└── test_data.txt
```

FKCB 对应 `Food`、`Kitchen`、`Clothing`、`Beauty`。不需要 `tracin_data.txt`、`item_users.txt` 或预处理缓存。在线评分直接从每个域的 TRAIN 中固定抽取一个 batch（默认256条，不足时取全部）作为参考样本，`--probe_seed` 默认42；样本仍参与完整训练和训练图，不另存文件。代码内部保留的 `tracin` 名称表示这个内存参考集，实际读取 `train_data.txt`。

## 训练与评测

以下三条训练命令各运行一次 FKCB、seed42，使用完整 TRAIN，每轮评测，并跳过训练后影响函数分析。它们使用不同 id，不会相互覆盖；同一目录已有 `result.json` 时，训练入口拒绝覆盖，也不会自动续训。

```bash
# 默认：开方贡献 + 30%均匀保底，在线评分开启，无 EMA。
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence

# 数据量加权，并跳过在线评分。
python main.py Food Kitchen Clothing Beauty --aggregation data_size --disable_online_scoring --id data42 --disable_influence

# 历史 EMA + z-score + softmax，在线评分开启。
python main.py Food Kitchen Clothing Beauty --aggregation ema_softmax --ema_beta 0.9 --aggregation_temperature 1.0 --id ema42 --disable_influence
```

| 命令 id | 配置、结果与最佳模型目录 | 日志 |
| --- | --- | --- |
| `sqrt42` | `checkpoint/domain_FKCB/VeriFRL_Fed_sqrt_floor_sqrt42/` | `log/domain_FKCB/VeriFRL_Fed_sqrt_floor_sqrt42.log` |
| `data42` | `checkpoint/domain_FKCB/VeriFRL_Fed_data_size_data42/` | `log/domain_FKCB/VeriFRL_Fed_data_size_data42.log` |
| `ema42` | `checkpoint/domain_FKCB/VeriFRL_Fed_ema_softmax_ema42/` | `log/domain_FKCB/VeriFRL_Fed_ema_softmax_ema42.log` |

只评测第一条命令已保存的最佳模型，使用相同数据和评测设置：

```bash
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence --do_eval
```

`--do_eval` 不训练、不覆盖原指标，日志追加到对应文件；会检查模型对应的数据文件和域顺序。它要求该次最佳模型文件仍存在，不适用于只保留指标的诊断归档。

可用 `--data_dir`、`--checkpoint_dir`、`--log_dir` 配置位置；完整参数见 `python main.py --help` 和 [main.py](main.py)。

## 聚合与评分

通过 `--aggregation` 选择三种策略：

- `sqrt_floor`（默认）：对当轮贡献分数取正部、开方并归一化，再混合70%贡献权重与30%均匀权重。无 EMA；所有分数非正时退回均匀权重，非有限分数报错。该策略必须开启在线评分。
- `data_size`：保留原仓库按训练 batch 补齐后样本数加权的口径，支持 `--disable_online_scoring`。部分客户端参与时，在参与者中重新归一化。
- `ema_softmax`：恢复历史 EMA 贡献聚合。首次参与时 EMA 直接取当轮分数，后续按 `EMA = beta × 上次EMA + (1-beta) × 当轮分数` 更新，再对参与客户端做总体 z-score 和 softmax，直接使用所得权重，不混合数据量权重或30%保底。保留负分；EMA 分数标准差不大于阈值时使用均匀权重。必须开启在线评分。

三种策略均作用于共享编码器参数和最后训练 batch 上传的共享表示，不包含跨 batch 累计平均或用户对齐。其他原有模型方法自动使用 `data_size` 并关闭在线贡献评分。

EMA 选项的参数为 `--ema_beta 0.9`（有限且0≤beta<1，0表示不平滑）、`--aggregation_temperature 1.0`（有限且大于0）和 `--std_epsilon 1e-12`（有限且非负的绝对标准差阈值）。聚合温度与模型训练用的 `--temperature` 是独立参数。部分参与时，只更新本轮参与客户端的 EMA，未参与者保留原值；归一化仅在参与者之间进行。EMA 使用当轮分数和该客户端的历史 EMA，不使用累计归因分数。默认开方策略不做 EMA。

在线贡献评分使用 TRAIN 参考集；训练后的影响函数分析是独立功能，使用验证集梯度及训练子集近似 Hessian。`--influence_train_ratio` 只控制后者，不控制在线参考集大小。

## 默认设置与复现范围

| 设置 | 当前默认值或规则 |
| --- | --- |
| 训练 | `seed=42`，最多30轮，每轮本地3轮，batch256 |
| 优化器 | Adam，学习率0.001，`betas=(0.9, 0.98)`，默认学习率倍率1 |
| 优化器兼容行为 | `--weight_decay` 虽保留，但当前构造未传入该值，有效 weight decay 为0 |
| 选模与早停 | 验证集宏平均 MRR 严格提高才更新最佳；连续5次评测未提高则早停；达到30轮上限不等于早停 |
| FKCB 训练顺序 | Food → Clothing → Beauty → Kitchen；`--training_order` 可指定固定排列 |
| 加载、初始化及聚合顺序 | 按命令中的域列表顺序；共享参数在 CPU 聚合，共享表示在原设备聚合 |
| 在线参考集 | 每域一个固定 TRAIN batch，大小跟随 `--batch_size`，`probe_seed=42` |
| 验证及测试 | 每轮使用固定候选，`eval_seed=20261021`，999个有放回负候选，仅排除目标正例，排名按严格大于比较 |
| 其他 | `threads=4`，周期快照默认关闭，梯度异常检测默认关闭 |

保持命令中的域顺序一致；单独调整训练顺序请使用 `--training_order`。MBG、SGH 沿用原设置时需显式传 `--anneal_cap 0.01`，FKCB 默认1.0。

候选、初始化、预处理、训练与评分使用独立随机流；评分及评测保护训练状态和随机状态。验证及测试关闭梯度记录，使用批量排名；注意力掩码直接在输入设备创建，评测跳过仅供训练使用的 Exclusive 预测。`--load_prep` 仅用于旧命令兼容，不读取或写入预处理缓存；`--detect_anomaly` 可重新开启原仓库已有的梯度异常检测。

固定 seed 不保证逐位复现。当前未开启 PyTorch 全局确定性算法，本地已复现 CUDA `index_select` 反传的非确定性。独立随机流、评分开关和聚合顺序也与旧实验1不同，不能声称精确复现其历史指标或停止轮次。一次运行中数据量加权更高，也不足以判断其稳定优于开方保底。

## 结果与模型保存

每次运行保存 `config.json`、逐轮原子更新的 `result.json` 和一份 `best_roundN.pt`。记录包含验证/测试指标、训练损失、贡献分数与实际权重；关闭评分时分数为 `null`。 EMA 策略的 `weight_trace` 还逐轮保存 `smoothed_scores`、`zscores`、EMA 系数、聚合温度、标准差阈值和归一化诊断；配置与最佳模型中同步保存这些参数。新最佳模型及结果保存成功后，才删除上一份最佳模型。

最佳模型只用于评测，不含优化器和 RNG 状态，不能用于中断轮无损续训。恢复最佳模型前后的指标差异原样保存在结果中；不会仅因微小差异中断后续处理。

只有显式设置 `--ckpt_interval N` 才保存 `round_NNN/client*.pt` 周期模型快照；它们也不包含完整可恢复训练状态。训练后影响函数分析要求保存周期快照，且不传 `--disable_influence`，输出 `influence_scores.json`，不会作为在线聚合依据。

## 本地诊断记录

以下历史记录对应加入 EMA 选项前的源码版本，保留原有源码哈希及结果；此次新增 EMA 选项尚未训练验证。

2026-10-31 的 FKCB 数据量加权诊断已完成：固定 FCBK 顺序与每轮随机客户端顺序各重复 seed42 三次，在线评分均关闭。测试 MRR 均值±样本标准差分别为 `0.148829 ± 0.002289` 和 `0.149362 ± 0.001333`；这些是同 seed 重复，不能当作跨种子稳健性结论。随机顺序诊断由已移除的临时代码完成，正式入口不提供每轮随机顺序实验模式。

本地轻量记录位于 `checkpoint/seed42_order_diagnosis/`：以 `findings.txt` 为索引，`summary.csv` 为结果表，`comparison.png` 为曲线，原始日志与逐轮记录保留在 `runs/`。一次性测试权重与临时代码已清理，删除清单见 `cleanup.json`；归档中的原始模型路径保留为历史来源，不能再直接加载。该目录被 Git 忽略，不是运行依赖，也不随新克隆分发。

## 代码来源

本项目基于原始 VeriFRL 实现。以下引用沿用原始仓库的记录：

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

## 协议与联系

采用 [Apache License 2.0](LICENSE)。VeriFRL-Plus 的问题请在本仓库提交 Issue；原始代码列出的作者联系邮箱为 `tao.tang@ieee.org`。
