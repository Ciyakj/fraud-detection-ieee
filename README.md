# 🕵️ Fraud Detection on IEEE-CIS Dataset

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Model](https://img.shields.io/badge/Model-LightGBM-orange)

---

## 📌 Overview
This project builds a fraud detection pipeline on the IEEE-CIS Fraud Detection dataset.  
It handles preprocessing, exploratory analysis, model training (LightGBM), and submission file generation.

**Current result:** LightGBM validation **ROC-AUC ≈ 0.968** on our local split.

---

## ⚙️ Project Structure
fraud-detection-ieee/
│
├── src/
│ ├── make_dataset.py # load raw CSVs, merge, clean, encode -> train_processed.csv / test_processed.csv
│ ├── eda.py # quick plots & data checks -> saved in reports/
│ ├── model_training.py # LightGBM training + validation AUC + final model + submission CSV
│ ├── predict.py # (optional) predict using a saved model
│ ├── feature_engineering.py # (optional) extra features (template)
│ ├── train.py # (optional) alternative training entrypoint (template)
│ └── utils.py # helper utilities (paths, timers, etc.)
│
├── reports/ # generated plots (PNG)
├── submissions/ # generated submissions (CSV)
├── data/
│ ├── raw/ # place Kaggle CSVs here (NOT tracked by git)
│ └── processed/ # train_processed.csv / test_processed.csv (NOT tracked by git)
│
├── requirements.txt
├── .gitignore
└── README.md



> 📝 Note: Large data files are **ignored** from Git. Download the dataset from Kaggle and place the CSVs in `data/raw/`.

---

## 📊 Dataset
- **Source:** Kaggle – IEEE-CIS Fraud Detection  
- **Files expected in `data/raw/`:**  
  - `train_transaction.csv`, `train_identity.csv`  
  - `test_transaction.csv`, `test_identity.csv`  
  - `sample_submission.csv`  

---

## 🚀 Setup
```bash
# clone
git clone https://github.com/Ciyakj/fraud-detection-ieee.git
cd fraud-detection-ieee

# create venv (Windows)
python -m venv venv
venv\Scripts\activate

# install deps
pip install -r requirements.txt







