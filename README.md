# Verifiable Federated Representation Learning for Cross-domain Sequential Recommendation

This is the official PyTorch implementation of the paper **"Verifiable Federated Representation Learning for Cross-domain Sequential Recommendation"**, accepted by **The ACM Web Conference 2026 (WWW '26)**.

<p align="center">
<b>English</b> | <a href="README_zh.md">中文</a>



------

## 🛠️ Environment Setup

We recommend using `Conda` to manage your environment.

```
# Create and activate the environment
conda create -n your_project_name python=3.12.6
conda activate your_project_name

# Install dependencies
pip install -r requirements.txt
```

------

## 📂 Project Structure

```
.
├── checkpoint/          # Saved model weights and training checkpoints (.pt)
├── data/                # Directory for raw and processed datasets
├── log/                 # Training logs, TensorBoard events, and experimental results
├── models/              # Neural network architecture definitions
├── utils/               # Utility functions and helper scripts
├── main.py              # Main entry point for starting the training process
└── requirements.txt     # List of Python dependencies
```

------

## 💾 Data Preparation

1. Get the Dataset: We follow the data preprocessing protocols of [FedDCSR](https://github.com/orion-orion/FedDCSR). The preprocessed datasets can be obtained from their [official repository](https://github.com/orion-orion/FedDCSR).

2. Organize the data as follows:

   ```
   data/data_domain/
   ├── item_users.txt
   ├── num_items.txt
   ├── test_data.txt
   ├── tracin_data.txt
   ├── train_data.txt
   └── valid_data.txt
   ```

   3.Important Note on `tracin` Data: The `tracin` directory contains a copy of the `valid` dataset. It is specifically used during the training/evaluation process to compute the gradient similarity between the training set and the evaluation set.

------

## 🚀 Getting Started

###  Training & Validation

Our `train.py` script handles both training and periodic validation. To start the process, run:

```
python main.py Food Kitchen Clothing Beauty [--id 01] [--load_prep] [--method VeriFRL_Fed]
```

**Argument Descriptions:**

- `Datasets`: Positional arguments to specify the target domains (e.g., `Food`, `Kitchen`, `Clothing`, `Beauty`).
- `--id`: (Optional) A unique identifier for the experiment run.
- `--load_prep`: (Optional) Use this flag to load pre-processed data.
- `--method`: (Optional) Specify the training method (default `VeriFRL_Fed`).
- `--influence_train_ratio`: A critical parameter to adjust the amount of data used when calculating the **Influence Function**. For example, `0.1` means using 10% of the training data for influence estimation.

**Full Parameter List & Descriptions**

This project provides a wide range of configurable parameters (e.g., hyper-parameters, data paths). You can explore them by this :

**Command Line Help**

To see the full list of available arguments and their default values, run:

```
python main.py --help
```

------

## 📝 Citation

If you find our work useful in your research, please consider citing:

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

## 📄 License

This project is licensed under the **Apache License 2.0**. 

------

## ✉️ Contact

For any questions, please open an **Issue** or contact `tao.tang@ieee.org`.