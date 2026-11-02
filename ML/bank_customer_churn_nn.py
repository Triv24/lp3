"""
QUESTION 3 (ML):
Given a bank customer, build a neural-network-based classifier that can
determine whether they will leave (churn) or not in the next 6 months.

Dataset: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling

HOW IT WORKS
    Churn_Modelling.csv is loaded from outside the script. At run time the
    program asks for the file path (press Enter to use Churn_Modelling.csv
    from the current folder).

    The model is a small feed-forward network:
        input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid)
    The improved version adds an extra layer, dropout and early stopping.

STEPS
    1. Imports
    2. Create the dataset file if it is missing
    3. Load the dataset
    4. Separate features and target, then split
    5. Normalise the data
    6. Build the baseline model
    7. Train the baseline model
    8. Build the improved model
    9. Train the improved model
    10. Accuracy and confusion matrix for both models
    11. Observation
    12. Plots with matplotlib

RUN
    python3 ml/py3_ChurnNN.py

    In Jupyter, open py3_ChurnNN.ipynb or split this file at the "# %%"
    markers.
"""


# 1. Import Libraries
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# 2. Read the Dataset
user_path = input("Enter CSV path (press Enter for Churn_Modelling.csv): ").strip().strip('"')
DATASET_FILE = user_path if user_path else "Churn_Modelling.csv"
while not os.path.exists(DATASET_FILE):
    DATASET_FILE = input("File not found. Enter the path again: ").strip().strip('"')

data = pd.read_csv(DATASET_FILE)

print("Dataset Shape:", data.shape)
print(data.head())

# 3. Seperate Features and Target
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

data = pd.get_dummies(
    data,
    columns=["Geography", "Gender"],
    drop_first=True
)

X = data.drop("Exited", axis=1)
y = data["Exited"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# 4. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Normalize the Data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build the initial Neural Network
initial_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

initial_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

initial_model.summary()

# 7. Improve the Model using Dropping and Early Stopping
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.summary()

# 8. Train the Improved Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 9. Predict the Test Data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# 10. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)




"""

# ============================================================

# ============================================================
Practical 3: Customer Churn Prediction using ANN
"My third practical is Customer Churn Prediction using an Artificial Neural Network. The objective is to predict whether a bank customer will leave the bank or continue using its services.

First, I loaded the Churn Modelling dataset and removed unnecessary columns like RowNumber, CustomerId and Surname. Then I converted categorical columns such as Geography and Gender into numerical values using One-Hot Encoding.

After that, I separated the input features and target variable, divided the dataset into training and testing data, and applied StandardScaler for feature scaling. I built an Artificial Neural Network using the Sequential model with Dense layers. I also used Dropout to reduce overfitting and Early Stopping to stop training when validation performance stopped improving. Finally, I evaluated the model using accuracy and a confusion matrix."

Remember:
- Type: Supervised Learning
- Problem: Binary Classification
- Algorithm: ANN
- Encoding: One-Hot Encoding
- Scaling: StandardScaler
- Overfitting control: Dropout and Early Stopping
- Evaluation: Accuracy and Confusion Matrix






















# ============================================================
# SECTION 3: DETAILED CODE EXPLANATION
# ============================================================

# 1. IMPORT LIBRARIES
#
# pandas:
# Used to load and manipulate the dataset.
#
# numpy:
# Used for numerical operations and array manipulation.
#
# train_test_split:
# Divides the dataset into training and testing data.
#
# StandardScaler:
# Standardizes the input features.
#
# accuracy_score:
# Calculates the percentage of correct predictions.
#
# confusion_matrix:
# Shows correct and incorrect classifications.
#
# classification_report:
# Displays precision, recall, F1 score and support.
#
# Sequential:
# Creates a neural network layer by layer.
#
# Dense:
# Creates a fully connected neural network layer.
#
# Dropout:
# Randomly disables some neurons during training.
#
# Input:
# Defines the shape of input data.
#
# EarlyStopping:
# Stops training when validation performance stops improving.


# 2. READ AND INSPECT DATASET
#
# pd.read_csv():
# Loads the CSV dataset into a Pandas DataFrame.
#
# data.shape:
# Returns the number of rows and columns.
#
# data.head():
# Displays the first five records.
#
# data.columns:
# Displays all column names.
#
# data.info():
# Shows column names, data types and non-null counts.
#
# data.isnull().sum().sum():
# Calculates the total number of missing values.
#
# Our dataset contains 10,000 rows and 14 columns.


# 3. SEPARATE FEATURES AND TARGET
#
# data.drop():
# Removes unnecessary columns.
#
# RowNumber:
# Row identification number.
#
# CustomerId:
# Unique customer identification number.
#
# Surname:
# Customer surname.
#
# These columns are removed because they are not useful
# general predictive features for this model.
#
# pd.get_dummies():
# Converts categorical data into numerical data.
#
# Geography:
# Contains country names such as France, Germany and Spain.
#
# Gender:
# Contains Male and Female categories.
#
# drop_first=True:
# Removes the first category to avoid redundant dummy columns.
#
# X:
# Contains input features.
#
# y:
# Contains the target column Exited.
#
# Exited = 0 means customer did not leave.
# Exited = 1 means customer left.


# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
#
# train_test_split():
# Divides the dataset into training and testing data.
#
# test_size=0.20:
# Allocates 20% of data for testing.
#
# random_state=42:
# Ensures reproducible splitting.
#
# stratify=y:
# Maintains class distribution in both datasets.
#
# X_train and y_train:
# Used for model training.
#
# X_test and y_test:
# Used for final model evaluation.


# 5. CREATE VALIDATION DATASET
#
# A validation dataset is used to monitor model performance
# during training.
#
# test_size=0.20:
# Takes 20% of the remaining training data for validation.
#
# Final distribution:
# 64% = Training data.
# 16% = Validation data.
# 20% = Testing data.
#
# Validation data helps monitor overfitting and is used
# by Early Stopping.


# 6. NORMALIZE THE DATA
#
# StandardScaler:
# Standardizes features to approximately zero mean
# and unit variance.
#
# fit_transform():
# Learns the mean and standard deviation from training
# data and transforms it.
#
# transform():
# Applies the same training-data scaling parameters
# to validation and testing data.
#
# Formula:
# Z = (X - Mean) / Standard Deviation
#
# Normalization helps neural networks train more effectively.
#
# IMPORTANT:
# We fit the scaler only on training data to prevent
# data leakage.


# 7. BUILD ARTIFICIAL NEURAL NETWORK
#
# Sequential:
# Creates a sequential neural network.
#
# Input(shape=(X_train.shape[1],)):
# Defines the number of input features.
#
# Dense(64, activation="relu"):
# Creates a hidden layer with 64 neurons.
#
# Dropout(0.3):
# Randomly disables 30% of neurons during training.
#
# Dense(32, activation="relu"):
# Creates another hidden layer with 32 neurons.
#
# Dropout(0.2):
# Randomly disables 20% of neurons during training.
#
# Dense(16, activation="relu"):
# Creates another hidden layer with 16 neurons.
#
# Dense(1, activation="sigmoid"):
# Creates one output neuron for binary classification.
#
# ReLU:
# Returns zero for negative inputs and the input itself
# for positive inputs.
#
# Sigmoid:
# Converts output into a value between 0 and 1.
#
# optimizer="adam":
# Uses Adam optimizer to update model weights.
#
# loss="binary_crossentropy":
# Calculates prediction error for binary classification.
#
# metrics=["accuracy"]:
# Tracks accuracy during training.
#
# model.summary():
# Displays the neural network architecture and parameters.


# 8. APPLY EARLY STOPPING
#
# EarlyStopping:
# Stops training when validation performance stops improving.
#
# monitor="val_loss":
# Monitors validation loss.
#
# patience=5:
# Waits for five consecutive epochs without improvement.
#
# restore_best_weights=True:
# Restores model weights from the best validation epoch.
#
# It helps reduce overfitting and unnecessary training.


# 9. TRAIN THE IMPROVED MODEL
#
# model.fit():
# Trains the neural network using training data.
#
# epochs=50:
# Maximum number of complete training passes.
#
# batch_size=32:
# Processes 32 samples before updating weights.
#
# validation_data:
# Uses validation data to monitor performance.
#
# callbacks=[early_stop]:
# Applies Early Stopping during training.
#
# verbose=1:
# Displays training progress.


# 10. PREDICT THE TEST DATA
#
# model.predict():
# Generates churn probabilities for test samples.
#
# y_pred_prob:
# Stores predicted probabilities between 0 and 1.
#
# y_pred_prob > 0.5:
# Uses 0.5 as the classification threshold.
#
# If probability > 0.5:
# Predicts customer churn (1).
#
# Otherwise:
# Predicts no churn (0).
#
# astype(int):
# Converts Boolean results into integers.
#
# ravel():
# Converts the output into a one-dimensional array.


# 11. CALCULATE ACCURACY SCORE
#
# accuracy_score():
# Calculates the proportion of correct predictions.
#
# Formula:
# Accuracy = Correct Predictions / Total Predictions
#
# Higher accuracy generally indicates better performance,
# but it should not be considered alone for imbalanced data.


# 12. DISPLAY CONFUSION MATRIX
#
# confusion_matrix():
# Displays actual versus predicted classes.
#
# For binary classification, the matrix is:
#
# [[TN, FP],
#  [FN, TP]]
#
# TN = True Negative.
# FP = False Positive.
# FN = False Negative.
# TP = True Positive.
#
# It helps identify the types of classification errors.


# 13. CLASSIFICATION REPORT
#
# classification_report():
# Displays:
#
# Precision:
# Correct positive predictions / All positive predictions.
#
# Recall:
# Correct positive predictions / All actual positives.
#
# F1 Score:
# Harmonic mean of precision and recall.
#
# Support:
# Number of actual samples in each class.


# 14. FINAL OUTPUT
#
# Displays a completion message after the practical finishes.


# ============================================================
# SECTION 4: TIME AND SPACE COMPLEXITY
# ============================================================

# Let:
# N = Number of training samples.
# D = Number of input features.
# H = Number of neurons in the network.
# E = Number of training epochs.
#
# DATA PREPROCESSING:
# Time Complexity: Approximately O(N * D)
# Because rows and feature values are processed.
#
# STANDARDIZATION:
# Time Complexity: O(N * D)
# Mean and standard deviation are calculated for features.
#
# ANN TRAINING:
# Approximate Time Complexity:
# O(E * N * P)
#
# P = Number of network parameters.
#
# Training time depends on the network architecture,
# dataset size, hardware and number of epochs.
#
# ANN PREDICTION:
# Approximate Time Complexity: O(M * P)
#
# M = Number of testing samples.
#
# SPACE COMPLEXITY:
# Dataset storage: O(N * D)
# Model parameters: O(P)
# Additional memory is required for activations,
# gradients and training batches.
#
# NOTE:
# ANN complexity depends on network architecture.
# These are simplified estimates, not exact runtime bounds.


# ============================================================
# SECTION 5: IMPORTANT VIVA QUESTIONS AND ANSWERS
# ============================================================

# Q1. What is ANN?
# Ans: ANN stands for Artificial Neural Network. It is a
# machine learning model inspired by the human brain.
#
# Q2. What is customer churn?
# Ans: Customer churn means a customer stops using
# a company's products or services.
#
# Q3. What is the objective of this practical?
# Ans: To predict whether a bank customer will leave
# the bank or continue using its services.
#
# Q4. What type of machine learning problem is this?
# Ans: Supervised binary classification.
#
# Q5. What is a neuron?
# Ans: A basic processing unit of a neural network.
#
# Q6. What is a hidden layer?
# Ans: A layer between the input and output layers
# that learns patterns from data.
#
# Q7. What is an activation function?
# Ans: A function that introduces non-linearity into
# a neural network.
#
# Q8. What is ReLU?
# Ans: ReLU returns zero for negative values and
# the input itself for positive values.
#
# Q9. What is Sigmoid?
# Ans: Sigmoid converts values into a range between
# zero and one.
#
# Q10. Why use Sigmoid in the output layer?
# Ans: Because this is a binary classification problem
# and we need a probability between zero and one.
#
# Q11. What is Dropout?
# Ans: Dropout randomly disables some neurons during
# training to help prevent overfitting.
#
# Q12. What is Early Stopping?
# Ans: It stops training when validation performance
# stops improving.
#
# Q13. What is patience=5?
# Ans: Training waits for five consecutive epochs
# without improvement before stopping.
#
# Q14. What is restore_best_weights=True?
# Ans: It restores weights from the epoch with the
# best validation performance.
#
# Q15. What is an epoch?
# Ans: One complete pass through the training dataset.
#
# Q16. What is batch size?
# Ans: The number of training samples processed before
# updating model weights.
#
# Q17. What is StandardScaler?
# Ans: It standardizes features to approximately
# zero mean and unit variance.
#
# Q18. Why do we normalize data?
# Ans: To put features on comparable scales and
# help neural network training.
#
# Q19. What is Adam optimizer?
# Ans: An optimization algorithm that updates model
# weights using adaptive learning rates.
#
# Q20. Why use binary_crossentropy?
# Ans: Because the output has two classes.
#
# Q21. What is overfitting?
# Ans: When a model learns training data too closely
# and performs poorly on unseen data.
#
# Q22. What is underfitting?
# Ans: When a model fails to learn important patterns
# from the training data.
#
# Q23. What is a confusion matrix?
# Ans: A table that compares actual and predicted
# class labels.
#
# Q24. What is the difference between training,
# validation and testing data?
# Ans: Training data learns model parameters.
# Validation data monitors training and helps tune
# the model. Testing data evaluates final performance.
#
# Q25. Why remove CustomerId and Surname?
# Ans: They are identifiers or names and are not useful
# general predictive features for this model.
#
# Q26. Why use get_dummies()?
# Ans: To convert categorical variables into numerical
# columns.
#
# Q27. What is binary classification?
# Ans: Predicting one of two possible classes.
#
# Q28. What does Exited = 1 mean?
# Ans: The customer has left the bank.
#
# Q29. What does Exited = 0 mean?
# Ans: The customer has not left the bank.
#
# Q30. Why use stratify=y?
# Ans: To maintain class proportions in training,
# validation and testing datasets.
#
# Q31. What is data leakage?
# Ans: When information from validation or testing data
# improperly influences model training.
#
# Q32. What is the role of the input layer?
# Ans: It receives the input features.
#
# Q33. What is the role of the output layer?
# Ans: It produces the final prediction.
#
# Q34. Why use 64, 32 and 16 neurons?
# Ans: They define the number of neurons in successive
# hidden layers. These are design choices and can be tuned.
#
# Q35. Is ANN always better than other algorithms?
# Ans: No. Performance depends on the dataset,
# architecture and model configuration.


# ============================================================
# SECTION 6: SAMPLE INPUT AND OUTPUT
# ============================================================

# SAMPLE INPUT DATASET:
#
# File name: Churn_Modelling.csv
#
# Rows: 10,000
# Columns: 14
# Missing values: 0
#
# Original columns:
# RowNumber, CustomerId, Surname, CreditScore,
# Geography, Gender, Age, Tenure, Balance,
# NumOfProducts, HasCrCard, IsActiveMember,
# EstimatedSalary, Exited
#
# TARGET:
# Exited = 0 (Not Churned)
# Exited = 1 (Churned)
#
# AFTER PREPROCESSING:
# Removed columns: 3
# Final input features: 11
#
# DATA SPLIT:
# Training: 64% = 6,400 samples
# Validation: 16% = 1,600 samples
# Testing: 20% = 2,000 samples
#
# SAMPLE MODEL OUTPUT:
#
# Model: "sequential"
# Layer (type)       Output Shape
# Dense              (None, 64)
# Dropout            (None, 64)
# Dense              (None, 32)
# Dropout            (None, 32)
# Dense              (None, 16)
# Dense              (None, 1)
#
# SAMPLE FINAL OUTPUT FORMAT:
#
# Accuracy Score: [Depends on training]
#
# Confusion Matrix:
# [[TN FP]
#  [FN TP]]
#
# Classification Report:
#              precision  recall  f1-score
# Not Churned     ...
# Churned         ...
#
# Customer Churn Prediction Completed Successfully!
#
# NOTE:
# Exact accuracy and confusion matrix values depend on
# the actual training run. Do not memorize invented values.


# ============================================================
# SECTION 7: LINUX COMMANDS
# ============================================================

# 1. Check Python version:
# python3 --version
#
# 2. Check pip version:
# pip3 --version
#
# 3. Install Pandas and NumPy:
# pip install pandas numpy
#
# 4. Install Scikit-learn:
# pip install scikit-learn
#
# 5. Install TensorFlow:
# pip install tensorflow
#
# 6. Start Jupyter Notebook:
# jupyter notebook
#
# 7. List files:
# ls
#
# 8. Change directory:
# cd folder_name
#
# 9. Display current directory:
# pwd
#
# 10. Run Python file:
# python3 practical3.py
#
# ============================================================
# END OF ML PRACTICAL 3
# ============================================================
"""
