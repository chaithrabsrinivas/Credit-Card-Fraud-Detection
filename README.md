# Credit Card Fraud Detection

## Overview
This project aims to develop a machine learning model to detect fraudulent credit card transactions. The goal is to minimize false positives while maximizing fraud detection accuracy.

## Dataset
- The dataset consists of transaction details, including:
  - **Transaction Amount**
  - **User ID**
  - **Merchant Information**
  - **Timestamps**
  - **Transaction Type**
  - **Fraud Label (0 = Legitimate, 1 = Fraudulent)**
- Ensure the dataset (`transactions.csv`) is placed in the `dataset/` folder.

## Project Structure
```
credit-card-fraud-detection/
│── dataset/              # Dataset files
│── models/               # Trained models
│── notebooks/            # Jupyter notebooks (if any)
│── src/                  # Python scripts
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── .gitignore            # Ignore unnecessary files
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Preprocess and Train Models**
   Run the `fraud_detection.py` script to train models:
   ```bash
   python src/fraud_detection.py
   ```
   This will save trained models in the `models/` directory.

2. **Make Predictions**
   Load a trained model and predict whether a new transaction is fraudulent:
   ```python
   import joblib
   import pandas as pd
   
   model = joblib.load('models/fraud_model.pkl')
   new_transaction = pd.DataFrame([{ "amount": 200.0, "merchant": "XYZ", "user_id": 1234, "time": "12:30:00" }])
   prediction = model.predict(new_transaction)
   print("Fraudulent Transaction:" if prediction[0] == 1 else "Legitimate Transaction")
   ```

## Models Used
- **Random Forest Classifier**
- **Logistic Regression**
- **XGBoost Classifier**

## Evaluation Metrics
- **Accuracy**
- **Precision & Recall** (to handle class imbalance)
- **F1 Score**
- **ROC-AUC Score**

## Contributions
Feel free to contribute by submitting pull requests or opening issues.

## License
This project is licensed under the MIT License.

