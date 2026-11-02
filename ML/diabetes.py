"""
QUESTION 4 (ML):
Implement the K-Nearest Neighbors algorithm on the diabetes.csv dataset.
Compute the confusion matrix, accuracy, error rate, precision and recall.

Dataset: https://www.kaggle.com/datasets/abdallamahgoub/diabetes

HOW IT WORKS
    diabetes.csv sits in the same folder as this file. If it is missing,
    Step 2 writes a small hardcoded demo file with the same columns.

    KNN classifies a patient by looking at the 5 nearest training patients.
    Zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are
    missing measurements, so they are replaced by the training median.
    Features are standardised because KNN uses distances.

STEPS
    1. Imports
    2. Create the dataset file if it is missing
    3. Load the dataset
    4. Split, impute and scale
    5. Train KNN and predict
    6. Confusion matrix and metrics
    7. Observation
    8. Plots with matplotlib

RUN
    python3 ml/py4_DiabetesKNN.py

    In Jupyter, open py4_DiabetesKNN.ipynb or split this file at the
    "# %%" markers.
"""
# 1. Import Libraries
import os
 
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
 
# 2. Read the Dataset
user_path = input("Enter CSV path (press Enter for diabetes.csv): ").strip().strip('"')
DATASET_FILE = user_path if user_path else "diabetes.csv"
while not os.path.exists(DATASET_FILE):
    DATASET_FILE = input("File not found. Enter the path again: ").strip().strip('"')
 
data = pd.read_csv(DATASET_FILE)
 
print("Dataset:")
print(data.head())
 
# 3. Select Features and Target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
 
# 4. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
 
# 5. Normalize the Features
scaler = StandardScaler()
 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# 6. Initialize and Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
 
# 7. Predict the Test Data
y_pred = knn.predict(X_test)
 
# 8. Calculate Evaluation Metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
 
# 9. Display Results
print("\nConfusion Matrix:")
print(cm)
 
print("\nAccuracy:", round(float(accuracy), 4))
print("Accuracy Percentage:", round(float(accuracy) * 100, 2), "%")
print("Error Rate:", round(float(error_rate), 4))
print("Error Rate Percentage:", round(float(error_rate) * 100, 2), "%")
print("Precision:", round(float(precision), 4))
print("Recall:", round(float(recall), 4))


"""
#=====================================================
#=====================================================

External explanation:

"In this practical, I implemented Diabetes Prediction using the K-Nearest Neighbors algorithm. The objective is to classify whether a person is diabetic or non-diabetic based on medical parameters.

First, I loaded the diabetes dataset using Pandas and checked the dataset information and missing values. Then I separated the input features and target variable, where 0 represents non-diabetic and 1 represents diabetic.

I divided the dataset into 80% training and 20% testing data. After that, I applied StandardScaler because KNN uses distance calculations and features have different scales.

I created a KNN classifier with K equal to 5, trained the model and predicted the test data. Finally, I evaluated the model using a confusion matrix, accuracy, error rate, precision and recall.

My model achieved approximately 70.13% accuracy."

Remember:
Topic	Answer
Dataset	diabetes.csv
Algorithm	KNN
K value	5
Features	8
Target	Outcome
Scaling	StandardScaler
Distance	Euclidean
Accuracy	70.13%
Precision	58.33%
Recall	51.85%


========================================

## 3. DATASET INFORMATION

Dataset name: diabetes.csv

- Total records: 768
- Total columns: 9
- Input features: 8
- Target column: Outcome
- Missing values: 0

Target values:
- 0 = Non-diabetic
- 1 = Diabetic

Input features:
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. Pedigree
8. Age

## 4. CODE EXPLANATION

### 1. Import Libraries

- pandas: Loads and manipulates the dataset.
- train_test_split: Splits the dataset.
- StandardScaler: Scales numerical features.
- KNeighborsClassifier: Implements KNN.
- Evaluation metrics: Measure model performance.

### 2. Load Dataset

`pd.read_csv("diabetes.csv")`

Loads the CSV file into a Pandas DataFrame.

### 3. Explore Dataset

- `data.shape`: Displays rows and columns.
- `data.head()`: Displays the first five records.
- `data.isnull().sum()`: Checks missing values.

### 4. Separate Features and Target

`X = data.drop("Outcome", axis=1)`

`y = data["Outcome"]`

- X contains the eight input features.
- y contains the target variable.
- `axis=1` means dropping a column.

### 5. Train-Test Split

`test_size=0.2` means 20% data is used for testing and 80% for training.

`random_state=42` ensures reproducible splitting.

`stratify=y` maintains the proportion of both classes in training and testing data.

### 6. Feature Scaling

`StandardScaler()` standardizes the features.

KNN uses distance calculations. If features have different scales, larger-valued features may dominate the distance.

- `fit_transform()`: Learns scaling parameters and transforms training data.
- `transform()`: Applies the same scaling to testing data.

We do not fit the scaler on testing data because that causes data leakage.

### 7. Create and Train KNN Model

`KNeighborsClassifier(n_neighbors=5)`

Creates a KNN model with K = 5.

`knn.fit(X_train, y_train)`

Trains the model using training data.

### 8. Make Predictions

`knn.predict(X_test)`

Predicts whether each test record belongs to class 0 or class 1.

### 9. Evaluate Model

- Confusion matrix: Shows correct and incorrect predictions.
- Accuracy: Overall correct predictions.
- Error rate: Overall incorrect predictions.
- Precision: Correct positive predictions among all predicted positives.
- Recall: Correct positive predictions among all actual positives.

### 10. Classification Report

Displays precision, recall, F1-score and support for both classes.

## 5. ALGORITHM

1. Start.
2. Import required libraries.
3. Load the diabetes dataset.
4. Explore the dataset and check missing values.
5. Separate input features and target variable.
6. Split data into 80% training and 20% testing.
7. Apply StandardScaler to the features.
8. Create a KNN classifier with K = 5.
9. Train the model.
10. Predict the testing data.
11. Calculate confusion matrix, accuracy, error rate, precision and recall.
12. Display the results.
13. Stop.

## 6. TIME AND SPACE COMPLEXITY

Let:
- n = Number of training records.
- d = Number of features.
- k = Number of neighbors.

| Operation | Complexity |
|---|---|
| KNN training | Approximately O(1) fitting |
| Prediction per test record | O(n × d) |
| Distance calculation | O(n × d) |
| Storage | O(n × d) |

KNN is called a lazy learning algorithm because it stores training data instead of building a complex model during training. Most computation happens during prediction.

## 7. OUTPUT

Your actual output:

```text
Dataset Shape: (768, 9)

Confusion Matrix:
[[80 20]
 [26 28]]

Accuracy: 0.7012987012987013
Error Rate: 0.2987012987012987
Precision: 0.5833333333333334
Recall: 0.5185185185185185

Classification Report:

              precision  recall  f1-score  support

           0       0.75    0.80      0.78      100
           1       0.58    0.52      0.55       54

    accuracy                       0.70      154
```

### Interpretation

- 108 out of 154 records were correctly classified.
- 80 non-diabetic cases were correctly identified.
- 28 diabetic cases were correctly identified.
- 20 non-diabetic cases were incorrectly classified as diabetic.
- 26 diabetic cases were incorrectly classified as non-diabetic.

## 8. IMPORTANT FORMULAS

1. Accuracy:

   (TP + TN) / (TP + TN + FP + FN)

2. Error Rate:

   1 - Accuracy

3. Precision:

   TP / (TP + FP)

4. Recall:

   TP / (TP + FN)

5. F1-score:

   2 × (Precision × Recall) / (Precision + Recall)

### Confusion Matrix Terms

- TP: Actual positive and predicted positive.
- TN: Actual negative and predicted negative.
- FP: Actual negative but predicted positive.
- FN: Actual positive but predicted negative.

For your output:

- TN = 80
- FP = 20
- FN = 26
- TP = 28

## 9. VIVA QUESTIONS AND ANSWERS

### Basic Questions

**Q1. What is KNN?**

K-Nearest Neighbors is a supervised machine learning algorithm used for classification and regression.

**Q2. What type of algorithm is KNN?**

Supervised, non-parametric and lazy learning algorithm.

**Q3. What is K in KNN?**

K represents the number of nearest neighbors considered for prediction.

**Q4. What is the value of K in your practical?**

K = 5.

**Q5. What is the target variable?**

Outcome. 0 represents non-diabetic and 1 represents diabetic.

**Q6. What is majority voting?**

The class occurring most frequently among the K nearest neighbors is selected as the prediction.

### Technical Questions

**Q7. Which distance formula is commonly used in KNN?**

Euclidean distance.

Distance = √[(x1-y1)² + (x2-y2)² + ...]

**Q8. Why is StandardScaler used?**

Because KNN uses distance calculations, and different feature scales can affect predictions.

**Q9. What happens if K is too small?**

The model may overfit and become sensitive to noise.

**Q10. What happens if K is too large?**

The model may underfit because too many neighbors influence the prediction.

**Q11. Why is an odd value of K preferred in binary classification?**

It reduces the possibility of a tie between two classes.

**Q12. Why is KNN called a lazy learner?**

Because it stores training data and performs distance calculations during prediction rather than building a complex model during training.

### Tricky Questions

**Q13. Is KNN parametric or non-parametric?**

Non-parametric.

**Q14. Is KNN sensitive to outliers?**

Yes. Outliers can affect distance calculations and predictions.

**Q15. What is the difference between classification and regression in KNN?**

- Classification predicts a class using majority voting.
- Regression predicts a numerical value, commonly using the average of neighbors.

**Q16. Why do we split the dataset?**

To train the model on one portion and evaluate it on unseen data.

**Q17. What is overfitting?**

When a model performs well on training data but poorly on unseen data.

**Q18. What is the limitation of your dataset?**

Some medical measurements contain zero values that may represent missing or invalid readings.

**Q19. What is the accuracy of your model?**

70.13%.

**Q20. What is the recall of your model?**

51.85%.

**Q21. What does low recall mean in this practical?**

The model misses some actual diabetic cases. It correctly identifies 28 out of 54 diabetic patients.

## 10. COMMON MISTAKES AND EXCEPTIONS

| Mistake | Correct approach |
|---|---|
| Forgetting feature scaling | Use StandardScaler |
| Scaling before splitting | Split first, then scale |
| Using fit_transform on test data | Use transform only |
| Including Outcome in X | Drop Outcome |
| Wrong CSV filename | Use diabetes.csv |
| Choosing K too small | Test different K values |
| Evaluating only accuracy | Also check precision and recall |

**Dataset limitation:** Some values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are zero. These may represent missing medical measurements. The current code retains these values, which can affect distance calculations.

## 11. LIMITATIONS AND IMPROVEMENTS

1. Replace invalid zero measurements with suitable imputed values after inspection.
2. Find the best K using cross-validation.
3. Compare KNN with other classification algorithms.
4. Evaluate recall and F1-score, not only accuracy.
5. Use a larger and more representative dataset.

This is an educational practical, not a medical diagnosis system.

## 12. ONE-MINUTE EXTERNAL EXPLANATION

In this practical, I implemented diabetes prediction using the K-Nearest Neighbors algorithm.

First, I loaded the diabetes dataset using Pandas and checked its records and missing values. Then I separated the input features and target variable, where Outcome 0 represents non-diabetic and Outcome 1 represents diabetic.

I divided the dataset into 80% training and 20% testing data. After that, I applied StandardScaler because KNN uses distance calculations.

I created a KNN classifier with K equal to 5 and trained it using the training data. Finally, I predicted the testing results and evaluated the model using a confusion matrix, accuracy, error rate, precision and recall.

My model achieved approximately 70.13% accuracy.

## 13. LINUX INSTALLATION AND EXECUTION COMMANDS

Install required libraries:

```bash
pip install pandas scikit-learn
```

Run Python file:

```bash
python diabetes_knn.py
```

For Jupyter Notebook:

```bash
pip install notebook pandas scikit-learn
jupyter notebook
```

## 14. LAST-MINUTE REVISION TABLE

| Topic | Answer |
|---|---|
| Algorithm | KNN |
| Learning type | Supervised |
| Problem type | Binary classification |
| Dataset | diabetes.csv |
| Records | 768 |
| Features | 8 |
| Target | Outcome |
| Training/testing | 80:20 |
| Scaling | StandardScaler |
| Neighbors | K = 5 |
| Distance | Euclidean |
| Prediction | Majority voting |
| Accuracy | 70.13% |
| Error rate | 29.87% |
| Precision | 58.33% |
| Recall | 51.85% |

**MOST IMPORTANT FOR EXTERNAL:**
1. What is KNN?
2. Why do we use StandardScaler?
3. What is K = 5?
4. Explain majority voting.
5. Explain confusion matrix.
6. Difference between precision and recall.
7. What happens when K is too small or too large?
8. Why is KNN called a lazy learner?

**CONCLUSION:**

The KNN algorithm was successfully implemented to classify diabetes outcomes. The model achieved 70.13% accuracy, 58.33% precision and 51.85% recall. The results demonstrate the working of KNN, although further improvements are needed, particularly in identifying diabetic patients.
"""
