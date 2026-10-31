# VeriFRL+: Contribution-Aware Aggregation for Federated Cross-Domain Sequential Recommendation

本仓库提供 **VeriFRL+: Contribution-Aware Aggregation for Federated Cross-Domain Sequential Recommendation** 的 PyTorch 实现。

VeriFRL+ 使用客户端贡献分数指导共享表示与模型参数的聚合，提供开方贡献加均匀保底策略，并支持数据量加权和基于 EMA 的贡献加权。

[English](README.md) | 中文

## 环境配置

建议使用 Conda 创建独立的 Python 环境，然后安装项目依赖：

```bash
conda create -n verifrl python=3.12.6
conda activate verifrl
pip install -r requirements.txt
```

## 项目结构

```text
.
├── checkpoint/          # 模型与结果
├── data/                # 数据集
├── log/                 # 训练日志
├── models/              # 模型定义
├── utils/               # 工具函数
├── main.py              # 训练与评测入口
└── requirements.txt     # 依赖列表
```

## 数据准备

数据准备沿用 [VeriFRL](https://github.com/TimiBlueGPT/VeriFRL) 的设置，数据集不随本仓库上传。每个域按以下结构组织：

```text
data/Food/
├── num_items.txt
├── train_data.txt
├── valid_data.txt
└── test_data.txt
```

FKCB 包含 `Food`、`Kitchen`、`Clothing` 和 `Beauty`。在线贡献评分使用训练集样本，无需单独准备 `tracin_data.txt`。

## 快速上手

`main.py` 集成了联邦训练、验证与测试流程。以下示例使用默认的贡献感知聚合方式，在 FKCB 四个域上运行：

```bash
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence
```

通过 `--aggregation` 选择聚合方式：

| 参数值 | 聚合方式 |
| --- | --- |
| `sqrt_floor`（默认） | 对正贡献分数开方并归一化，混合 30% 均匀保底权重 |
| `data_size` | 按数据量加权 |
| `ema_softmax` | 贡献分数经 EMA 平滑后，进行 z-score 标准化与 softmax 加权 |

其他两种聚合方式：

```bash
python main.py Food Kitchen Clothing Beauty --aggregation data_size --disable_online_scoring --id data42 --disable_influence
python main.py Food Kitchen Clothing Beauty --aggregation ema_softmax --ema_beta 0.9 --aggregation_temperature 1.0 --id ema42 --disable_influence
```

以上命令各运行一次 FKCB，使用 seed 42，并跳过训练后的影响函数分析。模型与结果保存在 `checkpoint/domain_FKCB/VeriFRL_Fed_<aggregation>_<id>/`，日志保存在 `log/domain_FKCB/`。再次运行请更换 `--id`，已有结果不会被覆盖。

评测默认运行保存的最佳模型：

```bash
python main.py Food Kitchen Clothing Beauty --id sqrt42 --disable_influence --do_eval
```

**常用参数：**

- `--id`：运行标识，用于区分输出目录和日志。
- `--seed`：随机种子，默认为 `42`。
- `--aggregation`：聚合方式，默认为 `sqrt_floor`。
- `--disable_online_scoring`：使用 `data_size` 时可关闭在线贡献评分。
- `--disable_influence`：跳过训练后的影响函数分析。
- `--do_eval`：加载已保存的最佳模型进行评测，不执行训练。

查看完整参数及默认值：

```bash
python main.py --help
```

## 原始 VeriFRL 引用

VeriFRL+ 基于我们的 [VeriFRL](https://github.com/TimiBlueGPT/VeriFRL) 实现，原始工作的引用如下：

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

## 开源协议

本项目采用 **Apache License 2.0** 协议开源。

## 联系方式

如有任何疑问，请提交 **Issue** 或联系邮箱：`tao.tang@ieee.org`。
