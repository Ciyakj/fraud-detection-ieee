# 🕵️ Fraud Detection on IEEE-CIS Dataset

## 📌 Overview
This project implements a **Fraud Detection System** using the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).  
The dataset is highly imbalanced and complex, simulating real-world online transactions.  
The goal is to build **machine learning pipelines** that can detect fraudulent transactions effectively while minimizing false positives.  

---

## ⚙️ Project Structure
fraud-detection-ieee/
│
├── src/ # Source code
│ ├── eda.py # Exploratory Data Analysis
│ ├── preprocess.py # Data preprocessing & feature engineering
│ ├── train.py # Model training script
│ ├── evaluate.py # Evaluation metrics & results
│ └── utils.py # Helper functions
│
├── reports/ # Visualizations & generated plots
│
├── requirements.txt # Dependencies
├── .gitignore # Ignored files (datasets, cache, etc.)
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Ciyakj/fraud-detection-ieee.git
cd fraud-detection-ieee
2️⃣ Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
📊 Usage
🧹 Run preprocessing
bash
Copy
Edit
python src/preprocess.py
🔍 Run Exploratory Data Analysis (EDA)
bash
Copy
Edit
python src/eda.py
🤖 Train the model
bash
Copy
Edit
python src/train.py
📈 Evaluate model performance
bash
Copy
Edit
python src/evaluate.py
📂 Data
The dataset comes from the IEEE-CIS Fraud Detection Kaggle competition.

Due to size restrictions, raw data is not included in this repo.

Download it from Kaggle: IEEE-CIS Dataset.

Place the extracted files inside a data/raw/ folder.

📊 Results (Sample)
Models tested: Logistic Regression, Random Forest, XGBoost, LightGBM

Metrics used: AUC, F1-Score, Precision, Recall

Example output (on test split):

AUC: 0.91

F1-Score: 0.84

Precision: 0.80

Recall: 0.88

🛠️ Tech Stack
Python 3.11

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

XGBoost, LightGBM

📌 Next Steps
Add deep learning models (LSTMs, Autoencoders)

Perform hyperparameter tuning

Implement model explainability (SHAP, LIME)

Deploy a Flask/Streamlit app for live fraud detection demo

👩‍💻 Author
Ciya K J
MSc Data Science | Aspiring Data Scientist | Passionate about ML & AI

🔗 LinkedIn | GitHub
