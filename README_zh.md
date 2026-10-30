# 可验证的联邦表示学习用于跨域序列推荐

这是论文 **"Verifiable Federated Representation Learning for Cross-domain Sequential Recommendation"** 的官方 PyTorch 实现，该论文已被 **ACM Web Conference 2026 (WWW '26)** 接收。

<p align="center"> <a href="README.md">English</a> | <b>中文</b> </p>

------

## 🛠️ 环境配置

建议使用 `Conda` 来管理您的环境。

```
# 创建并激活环境
conda create -n your_project_name python=3.12.6
conda activate your_project_name

# 安装依赖
pip install -r requirements.txt
```

------

## 📂 项目结构

```
.
├── checkpoint/          # 存放模型权重和训练检查点 (.pt)
├── data/                # 原始数据集和处理后数据的目录
├── log/                 # 训练日志、TensorBoard 事件和实验结果
├── models/              # 神经网络架构定义
├── utils/               # 工具函数和辅助脚本
├── main.py              # 启动训练过程的主程序入口
└── requirements.txt     # Python 依赖列表
```

------

## 💾 数据准备

1. 获取数据集： 本项目遵循 [FedDCSR](https://github.com/orion-orion/FedDCSR) 的数据预处理流程。您可以从其 [官方仓库 ](https://github.com/orion-orion/FedDCSR)中获取预处理好的数据集。

2. 请按以下结构组织数据：

   ```
   data/data_domain/
   ├── item_users.txt
   ├── num_items.txt
   ├── test_data.txt
   ├── tracin_data.txt
   ├── train_data.txt
   └── valid_data.txt
   ```

3. **关于 `tracin` 数据的重要说明**： `tracin` 目录下的数据是 `valid` (验证集) 数据的拷贝。它专门用于训练/评估过程中，计算训练集与评估集之间的梯度相似度。

------

## 🚀 快速上手

### 5.1 训练与验证

我们的 `main.py` 脚本集成了训练和定期验证逻辑。运行以下命令启动流程：

```
python main.py Food Kitchen Clothing Beauty [--id 01] [--load_prep] [--method VeriFRL_Fed]
```

**参数说明：**

- `Datasets`: 位置参数，用于指定目标领域（例如：`Food`, `Kitchen`, `Clothing`, `Beauty`）。
- `--id`: (可选) 实验运行的唯一标识符。
- `--load_prep`: (可选) 使用此 Flag 标记以加载预处理后的数据。
- `--method`: (可选) 指定训练方法（默认值为 `VeriFRL_Fed`）。
- **`--influence_train_ratio`**: 一个关键参数，用于调整计算 **影响函数 (Influence Function)** 时所使用的数据量。例如，`0.1` 表示使用 10% 的训练数据进行影响评估。

### 5.2 完整参数列表与说明

本项目提供了丰富的可配置参数（如超参数、数据路径等）。您可以通过以下方式查看：

#### A. 命令行帮助

运行以下命令查看所有可用参数及其默认值：

```
python main.py --help
```

#### B. 代码参考

若需了解每个参数背后的详细逻辑，请参考 `main.py` 中的参数解析部分。

------

## 📝 引用信息

如果您发现本工作对您的研究有所帮助，请考虑引用我们的论文：

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

------

## 📄 开源协议

本项目采用 **Apache License 2.0** 协议开源。

------

## ✉️ 联系方式

如有任何疑问，请提交 **Issue** 或联系邮箱：`tao.tang@ieee.org`。