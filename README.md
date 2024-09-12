Fraud Detection Using RandomForest Classifier
This repository contains code for training a Random Forest model to detect fraudulent transactions. The model is built using the dataset PS_20174392719_1491204439457_log.csv and aims to predict whether a transaction is fraudulent based on various features.

Project Overview
Fraudulent transactions are a significant problem in the financial industry, and machine learning models are often employed to detect anomalies in transaction patterns. This project demonstrates the use of a Random Forest classifier for detecting fraud using features like transaction type, amount, and others.

Dataset
The dataset used in this project is loaded from a CSV file PS_20174392719_1491204439457_log.csv. The features include:

type: Type of transaction (e.g., Cash Out, Transfer)
amount: The amount involved in the transaction
oldbalanceOrg: The account balance before the transaction
newbalanceOrig: The account balance after the transaction
oldbalanceDest: The recipient's balance before the transaction
newbalanceDest: The recipient's balance after the transaction
isFraud: The target variable (whether the transaction was fraudulent or not)
Steps:
Data Preprocessing:

One-hot encoding is applied to the type feature.
Non-numeric columns (nameOrig, nameDest) are dropped.
Missing values in the dataset are imputed using the median.
Train-Test Split:

The dataset is split into training, validation, and test sets. 70% of the data is used for training, and 30% is split equally between validation and testing.
Modeling:

A RandomForestClassifier is trained on the training data.
Model Evaluation:

The model is evaluated using various metrics such as:
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Visualization:

Confusion Matrix
ROC Curve
Precision-Recall Curve
Setup and Installation
Prerequisites
Python 3.x
Required Libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/fraud-detection-randomforest.git
cd fraud-detection-randomforest
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Code
Download the dataset and place it in the same directory as the code.
Run the Python script:
bash
Copy code
python fraud_detection.py
The script will preprocess the data, train the model, and display evaluation metrics and visualizations.

Model Evaluation
The model's performance is evaluated on the validation set using the following metrics:

Accuracy: Measures how often the classifier is correct.
Precision: Proportion of true positives among all predicted positives.
Recall: Proportion of actual positives that were correctly identified.
F1 Score: Harmonic mean of precision and recall.
ROC-AUC Score: Measures the area under the ROC curve, showing the trade-off between true positive and false positive rates.
Visualization
Confusion Matrix
A heatmap of the confusion matrix showing the actual vs predicted classes.

ROC Curve
Displays the trade-off between the true positive rate and false positive rate.

Precision-Recall Curve
Shows the balance between precision and recall for different thresholds.

Future Work
Experiment with different machine learning models such as XGBoost or LightGBM.
Hyperparameter tuning using Grid Search or Random Search.
Explore feature engineering techniques to improve model accuracy.
Handle class imbalance using oversampling, undersampling, or SMOTE.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

