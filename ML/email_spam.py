"""
QUESTION 2 (ML):
Classify the email using the binary classification method (Normal = not
spam / Abnormal = spam). Use K-Nearest Neighbors and Support Vector Machine
for classification and analyze their performance.

Dataset: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

HOW IT WORKS
    emails.csv is loaded from outside the script. At run time the program
    asks for the file path (press Enter to use emails.csv from the current
    folder). It is already a word-count matrix:
        Email No. : the email id
        3000 word columns : how many times that word occurs
        Prediction : 1 = spam, 0 = not spam
"""

# 1. Import Libraries
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# 2. Load and Inspect Dataset
user_path = input("Enter CSV path (press Enter for emails.csv): ").strip().strip('"')
DATASET_FILE = user_path if user_path else "emails.csv"
while not os.path.exists(DATASET_FILE):
    DATASET_FILE = input("File not found. Enter the path again: ").strip().strip('"')

data = pd.read_csv(DATASET_FILE)
print("Dataset Shape:", data.shape)
print(data.head())

print("\nColumns:")
print(data.columns)

# 3. Clean Data
data = data.drop(columns=["Email No."], errors="ignore")
data = data.drop_duplicates().dropna(subset=["Prediction"]).copy()
print("Cleaned Dataset Shape:", data.shape)
print("\nClass Distribution:")
print(data["Prediction"].value_counts())

# 4. Select Features and Split Dataset
X = data.drop(columns=["Prediction"])
y = data["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. TF-IDF Feature Extraction
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

# 6. Train and Evaluate KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)
y_pred_knn = knn.predict(X_test_tfidf)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, zero_division=0)
knn_recall = recall_score(y_test, y_pred_knn, zero_division=0)
knn_f1 = f1_score(y_test, y_pred_knn, zero_division=0)

print("----- KNN PERFORMANCE -----")
print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1 Score :", knn_f1)
print("\nClassification Report - KNN")
print(classification_report(y_test, y_pred_knn, zero_division=0))

# 7. Train and Evaluate SVM
svm = SVC(kernel="linear")
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, zero_division=0)
svm_recall = recall_score(y_test, y_pred_svm, zero_division=0)
svm_f1 = f1_score(y_test, y_pred_svm, zero_division=0)

print("----- SVM PERFORMANCE -----")
print("Accuracy :", svm_accuracy)
print("Precision:", svm_precision)
print("Recall   :", svm_recall)
print("F1 Score :", svm_f1)
print("\nClassification Report - SVM")
print(classification_report(y_test, y_pred_svm, zero_division=0))

# 8. Model Comparison
results = pd.DataFrame({
    "Model": ["KNN", "SVM"],
    "Accuracy": [knn_accuracy, svm_accuracy],
    "Precision": [knn_precision, svm_precision],
    "Recall": [knn_recall, svm_recall],
    "F1 Score": [knn_f1, svm_f1]
})
print("----- MODEL COMPARISON -----")
print(results.to_string(index=False))

if svm_accuracy > knn_accuracy:
    print("SVM performs better than KNN based on accuracy.")
elif knn_accuracy > svm_accuracy:
    print("KNN performs better than SVM based on accuracy.")
else:
    print("Both models have the same accuracy.")



# ============================================================
# ============================================================
Practical 2: Email Spam Classification using KNN and SVM
# ============================================================
# ============================================================
"My second practical is Email Spam Classification. The objective is to classify emails into spam and non-spam categories.

First, I loaded the emails dataset and performed preprocessing by removing duplicate and missing records. My dataset already contained numerical word-frequency features, so TF-IDF conversion was not required. I separated the input features and target column, where 0 represents non-spam and 1 represents spam.

Then I divided the dataset into 80% training and 20% testing data. I trained two algorithms: K-Nearest Neighbors and Support Vector Machine. Finally, I evaluated both models using accuracy, precision, recall and F1 score. In my results, SVM performed better than KNN, achieving approximately 96.72% accuracy."

Remember:
- Type: Supervised Learning
- Problem: Classification
- Algorithms: KNN, SVM
- Target: Prediction (0 or 1)
- Better model in your result: SVM













# ============================================================
# SECTION 3: DETAILED CODE EXPLANATION
# ============================================================

# 1. IMPORT LIBRARIES
#
# pandas:
# Used to load, process and manipulate datasets.
#
# train_test_split:
# Divides the dataset into training and testing data.
#
# KNeighborsClassifier:
# Implements the K-Nearest Neighbors algorithm.
#
# SVC:
# Implements Support Vector Machine classification.
#
# accuracy_score:
# Calculates the percentage of correct predictions.
#
# precision_score:
# Measures how many predicted spam emails are actually spam.
#
# recall_score:
# Measures how many actual spam emails are correctly detected.
#
# f1_score:
# Calculates the harmonic mean of precision and recall.
#
# classification_report:
# Displays precision, recall, F1 score and support for each class.


# 2. LOAD AND INSPECT DATASET
#
# data = pd.read_csv("emails.csv")
# Reads the CSV dataset and stores it in a DataFrame.
#
# data.shape:
# Returns the number of rows and columns.
# Our dataset contains 5172 rows and 3002 columns.
#
# data.head():
# Displays the first five rows.
#
# data.columns:
# Displays all column names.
#
# data.info():
# Displays column names, data types and non-null values.
#
# data.isnull().sum().sum():
# Calculates the total number of missing values.


# 3. CLEAN DATA
#
# data.drop_duplicates():
# Removes duplicate rows from the dataset.
#
# data.dropna():
# Removes rows containing missing values.
#
# data["Prediction"].value_counts():
# Displays the number of samples in each class.
#
# Prediction = 0 means Non-spam.
# Prediction = 1 means Spam.
#
# NOTE:
# dropna() removes a row if any column in that row has
# a missing value.


# 4. SELECT FEATURES AND SPLIT DATASET
#
# X = data.drop(["Email No.", "Prediction"], axis=1)
#
# Selects all input features except:
# Email No.  = Identifier, not a useful prediction feature.
# Prediction = Target/output column.
#
# y = data["Prediction"]
#
# Stores the target variable.
#
# train_test_split():
# Divides data into training and testing sets.
#
# test_size=0.2:
# Uses 20% data for testing and 80% for training.
#
# random_state=42:
# Ensures the same random split whenever code is executed.
#
# stratify=y:
# Maintains the class proportions in training and testing data.
#
# X_train and y_train:
# Used to train the models.
#
# X_test and y_test:
# Used to evaluate the models.
#
# IMPORTANT:
# This dataset already contains numerical word-frequency
# features, so TF-IDF conversion is not required.


# 5. TRAIN AND EVALUATE KNN
#
# KNN = K-Nearest Neighbors.
#
# knn = KNeighborsClassifier(n_neighbors=5)
#
# Creates a KNN classifier with K = 5.
#
# knn.fit(X_train, y_train):
# Trains the KNN model using training data.
#
# knn.predict(X_test):
# Predicts whether each test email is spam or non-spam.
#
# KNN WORKING:
# 1. Calculates distances between data points.
# 2. Finds the nearest K neighbors.
# 3. Checks the majority class among those neighbors.
# 4. Assigns that class to the new data point.
#
# n_neighbors=5 means the model considers five nearest
# neighbors while making a prediction.
#
# Evaluation metrics:
#
# Accuracy = Correct predictions / Total predictions
#
# Precision = TP / (TP + FP)
#
# Recall = TP / (TP + FN)
#
# F1 Score = 2 * (Precision * Recall) /
#            (Precision + Recall)
#
# zero_division=0:
# Returns zero instead of an undefined metric when
# its denominator is zero.


# 6. TRAIN AND EVALUATE SVM
#
# SVM = Support Vector Machine.
#
# svm = SVC(kernel="linear")
#
# Creates an SVM classifier using a linear kernel.
#
# svm.fit(X_train, y_train):
# Trains the SVM model.
#
# svm.predict(X_test):
# Predicts the classes of test emails.
#
# SVM WORKING:
# 1. Identifies different classes in the dataset.
# 2. Finds an optimal separating hyperplane.
# 3. Maximizes the margin between the classes.
# 4. Uses the hyperplane to classify new data.
#
# kernel="linear":
# Uses a linear decision boundary.
#
# SVM also uses the same evaluation metrics as KNN.


# 7. MODEL COMPARISON
#
# pd.DataFrame():
# Creates a comparison table for both algorithms.
#
# Model:
# Name of the machine learning algorithm.
#
# Accuracy:
# Overall correctness of predictions.
#
# Precision:
# Correctness of positive (spam) predictions.
#
# Recall:
# Ability to detect actual spam emails.
#
# F1 Score:
# Balance between precision and recall.
#
# The if-elif-else condition compares the F1 scores.
#
# Higher F1 Score generally indicates better balance
# between precision and recall.


# 8. FINAL OUTPUT
#
# Prints a completion message after all steps finish.


# ============================================================
# SECTION 4: TIME AND SPACE COMPLEXITY
# ============================================================

# Let:
# N = Number of training samples
# M = Number of testing samples
# D = Number of input features
# K = Number of nearest neighbors
#
# DATA PREPROCESSING:
# Time Complexity: O(N * D)
# Because dataset rows and feature values are processed.
#
# KNN:
# Training: Approximately O(1) for storing the training data.
# Prediction: O(M * N * D)
# Because distances are calculated between test and
# training samples across all features.
#
# SVM:
# Training complexity depends on the solver, kernel
# and dataset. It is generally more computationally
# expensive than simple linear models.
#
# Prediction: Depends on the number of support vectors
# and input features.
#
# SPACE COMPLEXITY:
# Dataset storage: O(N * D)
# KNN stores training samples, so prediction also requires
# access to the training data.
# SVM stores the learned model and support vectors.


# ============================================================
# SECTION 5: IMPORTANT VIVA QUESTIONS AND ANSWERS
# ============================================================

# Q1. What is Machine Learning?
# Ans: Machine Learning is a field of AI that enables
# computers to learn patterns from data and make predictions.
#
# Q2. What is classification?
# Ans: Classification is a supervised learning technique
# used to predict categories or class labels.
#
# Q3. What is spam email?
# Ans: An unwanted or unsolicited email, often containing
# advertisements, scams or suspicious links.
#
# Q4. What is KNN?
# Ans: KNN is a supervised learning algorithm that classifies
# data based on the majority class of its nearest K neighbors.
#
# Q5. What is SVM?
# Ans: SVM is a supervised learning algorithm that finds
# an optimal hyperplane to separate different classes.
#
# Q6. What is a hyperplane?
# Ans: A decision boundary that separates classes in
# a feature space.
#
# Q7. What is a kernel in SVM?
# Ans: A kernel helps SVM create decision boundaries.
# Examples: Linear, Polynomial and RBF.
#
# Q8. What is TF-IDF?
# Ans: TF-IDF stands for Term Frequency-Inverse Document
# Frequency. It represents the importance of words in text.
#
# Q9. Did we use TF-IDF in this practical?
# Ans: No. Our dataset already contains numerical
# word-frequency features.
#
# Q10. What is n_neighbors=5?
# Ans: KNN considers the five nearest data points
# to classify a new sample.
#
# Q11. What is train_test_split?
# Ans: It divides the dataset into training and testing data.
#
# Q12. Why do we use random_state=42?
# Ans: To obtain reproducible random splitting.
#
# Q13. Why do we use stratify=y?
# Ans: To maintain the class distribution in both sets.
#
# Q14. What is accuracy?
# Ans: The proportion of total predictions that are correct.
#
# Q15. What is precision?
# Ans: Out of all emails predicted as spam, how many
# were actually spam?
#
# Q16. What is recall?
# Ans: Out of all actual spam emails, how many were
# correctly identified?
#
# Q17. What is F1 Score?
# Ans: The harmonic mean of precision and recall.
#
# Q18. Which algorithm performed better in our practical?
# Ans: SVM performed better based on the F1 Score.
#
# Q19. What is overfitting?
# Ans: When a model learns training data too closely
# and performs poorly on unseen data.
#
# Q20. What is the difference between classification
# and regression?
# Ans: Classification predicts categories, whereas
# regression predicts continuous numerical values.
#
# Q21. What is supervised learning?
# Ans: Learning using labelled training data.
#
# Q22. What is a feature?
# Ans: An input variable used by a machine learning model.
#
# Q23. What is a target variable?
# Ans: The output variable that the model tries to predict.
#
# Q24. Why remove Email No.?
# Ans: It is only an identifier and does not represent
# meaningful information for spam classification.
#
# Q25. Why compare two algorithms?
# Ans: To identify which algorithm performs better
# on the given dataset and evaluation metrics.


# ============================================================
# SECTION 6: SAMPLE INPUT AND OUTPUT
# ============================================================

# SAMPLE DATASET:
#
# File name: emails.csv
#
# Total rows: 5172
# Total columns: 3002
#
# Columns:
# Email No. | Word Features (3000 columns) | Prediction
#
# Prediction:
# 0 = Non-spam
# 1 = Spam
#
#
# SAMPLE OUTPUT FROM YOUR PREVIOUS RUN:
#
# ----- MODEL COMPARISON -----
#
# Model  Accuracy  Precision  Recall  F1 Score
# KNN    0.871498  0.744868   0.846667  0.792512
# SVM    0.967150  0.937500   0.950000  0.943709
#
# SVM performs better than KNN based on F1 Score.
#
# Email Spam Classification Completed Successfully!
#
# NOTE:
# Exact results depend on the dataset and preprocessing.


# ============================================================
# SECTION 7: LINUX COMMANDS
# ============================================================

# 1. Check Python version:
# python3 --version
#
# 2. Check pip version:
# pip3 --version
#
# 3. Install required libraries:
# pip install pandas scikit-learn
#
# 4. Start Jupyter Notebook:
# jupyter notebook
#
# 5. List files in current directory:
# ls
#
# 6. Change directory:
# cd folder_name
#
# 7. Run Python file:
# python3 practical2.py
#
# 8. Display current directory:
# pwd
#
# ============================================================
# END OF ML PRACTICAL 2
# ============================================================
