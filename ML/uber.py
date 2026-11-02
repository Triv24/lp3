# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 2. Load Dataset
df = pd.read_csv("uber.csv")
# Normalize possible PDF-copy spelling errors in column names.
df.rename(columns={"dropoƯ_longitude": "dropoff_longitude", "dropoƯ_latitude": "dropoff_latitude"}, inplace=True)
print("First 5 rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Data Preprocessing
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

numeric_columns = [
    "fare_amount", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(inplace=True)

# 4. Remove Invalid Values
df = df[df["fare_amount"] > 0]
df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]
df = df[
    df["pickup_latitude"].between(-90, 90)
    & df["dropoff_latitude"].between(-90, 90)
    & df["pickup_longitude"].between(-180, 180)
    & df["dropoff_longitude"].between(-180, 180)
].copy()

# 5. Calculate Distance using Haversine Formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)
df = df[df["distance_km"] > 0].copy()

# 6. Identify and Remove Outliers using IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_limit) & (data[column] <= upper_limit)].copy()

print("Shape before removing outliers:", df.shape)
df = remove_outliers(df, "fare_amount")
df = remove_outliers(df, "distance_km")
print("Shape after removing outliers:", df.shape)

# 7. Correlational Analysis
correlation = df[[
    "fare_amount", "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "distance_km"
]].corr()
print("Correlation Matrix:")
print(correlation)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 8. Select Features and Split Dataset
X = df[[
    "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "distance_km"
]]
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 9. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_prediction = linear_model.predict(X_test)

# 10. Random Forest Regression Model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
random_forest.fit(X_train, y_train)
rf_prediction = random_forest.predict(X_test)

# 11. Evaluate Models
def evaluate_model(name, actual, predicted):
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print("\n" + name)
    print("---------------------------")
    print("R2 Score :", round(r2, 4))
    print("MAE      :", round(mae, 4))
    print("RMSE     :", round(rmse, 4))
    return r2, mae, rmse

linear_scores = evaluate_model("Linear Regression", y_test, linear_prediction)
rf_scores = evaluate_model("Random Forest Regression", y_test, rf_prediction)

# 12. Compare Models
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest Regression"],
    "R2 Score": [linear_scores[0], rf_scores[0]],
    "MAE": [linear_scores[1], rf_scores[1]],
    "RMSE": [linear_scores[2], rf_scores[2]]
})
print("Model Comparison:")
display(results)

# 13. Actual vs Predicted Fare
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_prediction, alpha=0.5)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare - Random Forest")
plt.show()





+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
====================================================================
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




Practical 1: Uber Fare Prediction using Machine Learning
"My first practical is Uber Fare Prediction using Machine Learning. The objective is to predict the fare of an Uber ride based on pickup location, drop-off location and passenger count.

First, I loaded the Uber dataset using Pandas. Then I performed data preprocessing by removing missing values, duplicate records and invalid data. After that, I calculated the distance between pickup and drop-off locations using the Haversine formula. I removed outliers using the IQR method and performed correlation analysis using a heatmap.

Then I divided the dataset into 80% training and 20% testing data. I trained two algorithms: Linear Regression and Random Forest Regression. Finally, I compared their performance using R² score, MAE and RMSE."

Remember:
- Type: Supervised Learning
- Problem: Regression
- Algorithms: Linear Regression, Random Forest
- Distance: Haversine Formula


====================================

"Linear Regression is a supervised machine learning algorithm that predicts continuous numerical values by finding the best-fit straight line between input and output variables."

2. What is Random Forest Regression?

"Random Forest Regression is a supervised machine learning algorithm that uses multiple decision trees and averages their predictions to predict a continuous numerical value."

Remember:
- Linear Regression → Best-fit straight line.
- Random Forest Regression → Multiple decision trees + average prediction.



# ML PRACTICAL 1 – UBER FARE PREDICTION

## 1. AIM OF THE PRACTICAL

To predict Uber ride fare amounts using Machine Learning by performing data preprocessing, outlier detection, correlation analysis, Linear Regression and Random Forest Regression.

**Dataset:** Uber Fares Dataset (Kaggle)

**Algorithms used:**
1. Linear Regression
2. Random Forest Regression

**Evaluation metrics:**
1. R² Score
2. Mean Absolute Error (MAE)
3. Root Mean Squared Error (RMSE)

## 2. OVERALL PROJECT FLOW

Dataset → Data Preprocessing → Remove Invalid Values → Calculate Distance → Remove Outliers → Correlation Analysis → Train-Test Split → Model Training → Prediction → Model Evaluation → Comparison.

---

# 3. BLOCK-WISE CODE EXPLANATION

## BLOCK 1 – IMPORT NECESSARY LIBRARIES

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
```

**Explanation:**

- `pandas`: Used for reading and manipulating datasets.
- `numpy`: Used for mathematical and numerical operations.
- `matplotlib.pyplot`: Used to create graphs.
- `seaborn`: Used for statistical visualizations and heatmaps.
- `train_test_split`: Divides data into training and testing sets.
- `LinearRegression`: Imports the Linear Regression algorithm.
- `RandomForestRegressor`: Imports the Random Forest Regression algorithm.
- `r2_score`: Calculates the R² score.
- `mean_absolute_error`: Calculates MAE.
- `mean_squared_error`: Calculates MSE, which is used to calculate RMSE.

**Base concepts:**

- Library: Collection of pre-written functions and classes.
- Pandas: Python library for data manipulation and analysis.
- NumPy: Python library for numerical calculations.
- Scikit-learn: Python library for machine-learning algorithms.
- Regression: Predicting a continuous numerical value.

**Viva Questions:**

Q1. Why do we use Pandas?  
Ans: To read, manipulate, clean and analyze datasets.

Q2. Difference between NumPy and Pandas?  
Ans: NumPy mainly works with numerical arrays, while Pandas works with labelled tabular data.

Q3. What is Scikit-learn?  
Ans: A Python library used for machine-learning algorithms and model evaluation.

Q4. Why do we use Seaborn?  
Ans: To create statistical graphs and visualizations.

---

## BLOCK 2 – LOAD THE DATASET

```python
df = pd.read_csv("uber.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())
```

**Explanation:**

- `pd.read_csv()` reads the CSV file.
- `df` stores the dataset as a DataFrame.
- `df.head()` displays the first five rows.
- `df.shape` returns the number of rows and columns.
- `df.info()` displays column names, data types and non-null counts.
- `df.isnull().sum()` counts missing values in every column.

**Base concepts:**

- DataFrame: A two-dimensional table containing rows and columns.
- Missing values: Data entries that are empty or unavailable.
- CSV: Comma-Separated Values, a file format used to store tabular data.

**Viva Questions:**

Q1. What is a DataFrame?  
Ans: A two-dimensional data structure containing rows and columns.

Q2. What does `head()` do?  
Ans: Displays the first five rows by default.

Q3. What does `shape` return?  
Ans: A tuple containing the number of rows and columns.

Q4. How do you display the last five rows?  
Ans: Using `df.tail()`.

Q5. How do you check data types?  
Ans: Using `df.dtypes` or `df.info()`.

---

## BLOCK 3 – DATA PREPROCESSING

```python
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

numeric_columns = [
    "fare_amount", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)

print("Shape after preprocessing:")
print(df.shape)
```

**Explanation:**

1. `if "Unnamed: 0" in df.columns:` checks whether an unnecessary column exists.
2. `df.drop()` removes that column.
3. `axis=1` means column-wise operation.
4. `inplace=True` modifies the existing DataFrame.
5. `drop_duplicates()` removes repeated records.
6. `dropna()` removes rows containing missing values.
7. `numeric_columns` stores the names of columns that should contain numbers.
8. The `for` loop visits each selected column.
9. `pd.to_numeric()` converts values into numerical format.
10. `errors="coerce"` converts invalid values into NaN.
11. The second `dropna()` removes rows where invalid text was converted into NaN.

**Base concepts:**

- Data cleaning: Removing or correcting incorrect, missing or duplicate data.
- Duplicate: A repeated record.
- NaN: Not a Number; commonly used to represent missing or invalid numerical data.
- Type conversion: Changing data from one type to another.
- `inplace=True`: Modifies the existing DataFrame.

**Viva Questions:**

Q1. What is preprocessing?  
Ans: The process of cleaning and preparing raw data before applying machine-learning algorithms.

Q2. Why remove duplicates?  
Ans: Repeated records can distort analysis and model training.

Q3. What does `dropna()` do?  
Ans: Removes rows containing missing values by default.

Q4. What is `errors="coerce"`?  
Ans: It converts values that cannot be parsed as numbers into NaN.

Q5. Difference between `dropna()` and `fillna()`?  
Ans: `dropna()` removes missing-value rows, whereas `fillna()` replaces missing values.

---

## BLOCK 4 – REMOVE INVALID VALUES

```python
df = df[df["fare_amount"] > 0]

df = df[
    (df["passenger_count"] >= 1) &
    (df["passenger_count"] <= 6)
]

df = df[
    df["pickup_latitude"].between(-90, 90)
    & df["dropoff_latitude"].between(-90, 90)
    & df["pickup_longitude"].between(-180, 180)
    & df["dropoff_longitude"].between(-180, 180)
].copy()

print("Shape after removing invalid values:")
print(df.shape)
```

**Explanation:**

1. `df["fare_amount"] > 0` keeps only records with positive fare.
2. Passenger count is restricted to values between 1 and 6.
3. `&` combines multiple conditions using AND.
4. `between(-90, 90)` checks whether latitude is within its valid global range.
5. `between(-180, 180)` checks whether longitude is within its valid global range.
6. `.copy()` creates a separate copy of the filtered DataFrame.
7. `df.shape` displays the updated dataset size.

**Base concepts:**

- Filtering: Selecting rows that satisfy a condition.
- Latitude: Measures north-south geographical position.
- Longitude: Measures east-west geographical position.
- Boolean condition: A condition that evaluates to True or False.

**Viva Questions:**

Q1. Why remove fare amounts less than or equal to zero?  
Ans: Because they are invalid for normal positive-fare prediction.

Q2. What is the valid latitude range?  
Ans: -90 to +90 degrees.

Q3. What is the valid longitude range?  
Ans: -180 to +180 degrees.

Q4. Difference between `&` and `and`?  
Ans: `&` performs element-wise AND for Pandas conditions. `and` is used for single Boolean expressions.

Q5. Why use `.copy()`?  
Ans: To create an independent copy and avoid unintended modifications.

---

## BLOCK 5 – CALCULATE DISTANCE USING HAVERSINE FORMULA

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return R * c


df["distance_km"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

df = df[df["distance_km"] > 0].copy()

print("Distance calculated successfully!")
```

**Explanation:**

1. `def haversine()` defines a function that accepts pickup and drop-off coordinates.
2. `R = 6371` stores Earth's approximate mean radius in kilometres.
3. `np.radians()` converts degrees into radians.
4. `dlat` calculates the difference between latitudes.
5. `dlon` calculates the difference between longitudes.
6. The variable `a` calculates the Haversine value using trigonometric operations.
7. `np.clip(a, 0, 1)` restricts the value between 0 and 1 to avoid floating-point errors.
8. `np.sqrt()` calculates the square root.
9. `np.arcsin()` calculates inverse sine.
10. `return R * c` calculates the geographical distance in kilometres.
11. `df["distance_km"]` creates a new column containing calculated distances.
12. The final filter removes zero-distance rides.

**Base concepts:**

Haversine formula: A mathematical formula used to calculate the great-circle distance between two geographical points using latitude and longitude.

- Earth's radius = approximately 6371 km.
- Radians: Angular measurement unit. 180 degrees = π radians.
- Great-circle distance: Shortest surface distance between two points on a sphere.

**Viva Questions:**

Q1. Why use the Haversine formula?  
Ans: To calculate geographical distance between two latitude-longitude coordinates.

Q2. Why is Earth's radius 6371?  
Ans: It is the approximate mean radius of Earth in kilometres.

Q3. Why convert degrees into radians?  
Ans: Because trigonometric functions expect angles in radians.

Q4. What is the output unit?  
Ans: Kilometres, because the radius is given in kilometres.

Q5. Is Haversine distance the actual road distance?  
Ans: No. It is straight-line distance over Earth's surface.

Q6. What does `np.clip()` do?  
Ans: Restricts values to a specified minimum and maximum.

---

## BLOCK 6 – IDENTIFY AND REMOVE OUTLIERS USING IQR

```python
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    return data[
        (data[column] >= lower_limit) &
        (data[column] <= upper_limit)
    ].copy()


print("Shape before removing outliers:")
print(df.shape)

df = remove_outliers(df, "fare_amount")
df = remove_outliers(df, "distance_km")

print("Shape after removing outliers:")
print(df.shape)
```

**Explanation:**

1. `remove_outliers()` defines a function to identify and remove potential outliers.
2. `Q1` calculates the 25th percentile.
3. `Q3` calculates the 75th percentile.
4. `IQR = Q3 - Q1` calculates the Interquartile Range.
5. `lower_limit` calculates the lower boundary.
6. `upper_limit` calculates the upper boundary.
7. The return statement keeps values within the calculated limits.
8. The function is applied first to `fare_amount`.
9. It is then applied to `distance_km`.
10. The shapes before and after filtering are displayed.

**Important formulas:**

IQR = Q3 - Q1

Lower Limit = Q1 - 1.5 × IQR

Upper Limit = Q3 + 1.5 × IQR

**Base concepts:**

- Outlier: A value unusually far from most other observations.
- Quartile: A value that divides sorted data into four parts.
- IQR: Difference between the third and first quartiles.
- Q1: 25th percentile.
- Q3: 75th percentile.

**Viva Questions:**

Q1. What is an outlier?  
Ans: A data value significantly different from most other values.

Q2. What is IQR?  
Ans: The difference between the 75th and 25th percentiles.

Q3. Why use 1.5?  
Ans: It is the conventional multiplier used by the IQR rule to flag potential outliers.

Q4. Can outliers always be removed?  
Ans: No. Some outliers may be genuine observations and should be investigated first.

Q5. Why remove outliers?  
Ans: To reduce the influence of extreme values that may distort analysis and model training.

---

## BLOCK 7 – CORRELATION ANALYSIS

```python
correlation = df[[
    "fare_amount", "passenger_count",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "distance_km"
]].corr()

print("Correlation Matrix:")
print(correlation)

plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()
```

**Explanation:**

1. Selects the required numerical columns.
2. `.corr()` calculates pairwise correlation between the columns.
3. `plt.figure(figsize=(10, 6))` creates a graph of the specified size.
4. `sns.heatmap()` creates a correlation heatmap.
5. `annot=True` displays numerical correlation values.
6. `cmap="coolwarm"` selects the colour gradient.
7. `fmt=".2f"` displays values up to two decimal places.
8. `plt.title()` adds the graph title.
9. `plt.show()` displays the graph.

**Base concepts:**

Correlation measures the strength and direction of the linear relationship between two variables.

Correlation values range from -1 to +1.

- +1: Perfect positive correlation.
- 0: No linear correlation.
- -1: Perfect negative correlation.

**Viva Questions:**

Q1. What is correlation?  
Ans: It measures the strength and direction of a linear relationship between two variables.

Q2. What does +1 mean?  
Ans: Perfect positive correlation.

Q3. What does -1 mean?  
Ans: Perfect negative correlation.

Q4. What does 0 mean?  
Ans: No linear correlation.

Q5. Does correlation mean causation?  
Ans: No. Correlation does not prove that one variable causes another.

Q6. Why use a heatmap?  
Ans: To visually understand relationships between multiple numerical variables.

Q7. Which feature would you expect to be related to fare?  
Ans: Distance is generally expected to have a positive relationship with fare.

---

## BLOCK 8 – SELECT FEATURES AND SPLIT DATASET

```python
X = df[[
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km"
]]

y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
```

**Explanation:**

1. `X` stores the independent input features.
2. `y` stores the dependent target variable.
3. `train_test_split()` divides the dataset into training and testing sets.
4. `X_train` contains training features.
5. `X_test` contains testing features.
6. `y_train` contains actual training target values.
7. `y_test` contains actual testing target values.
8. `test_size=0.20` reserves 20% for testing.
9. `random_state=42` makes the split reproducible.
10. The final print statements display the sizes of training and testing data.

**Base concepts:**

- Feature: Input variable used by a model.
- Target: Value the model needs to predict.
- Training data: Data used to teach the model.
- Testing data: Unseen data used to evaluate the model.

**Viva Questions:**

Q1. What is X?  
Ans: Independent input features.

Q2. What is y?  
Ans: Dependent target variable, here `fare_amount`.

Q3. Why split the dataset?  
Ans: To train the model on one part and evaluate it on unseen data.

Q4. What does 80:20 mean?  
Ans: 80% training data and 20% testing data.

Q5. Why use `random_state=42`?  
Ans: To get the same random split when the code is run again.

Q6. Can we train and test on the same data?  
Ans: We should not for normal evaluation because it can give overly optimistic results.

---

## BLOCK 9 – LINEAR REGRESSION MODEL

```python
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)
```

**Explanation:**

1. `LinearRegression()` creates a Linear Regression model.
2. `.fit(X_train, y_train)` trains the model using training data.
3. The model learns the relationship between input features and fare amount.
4. `.predict(X_test)` predicts fare amounts for testing data.
5. `linear_prediction` stores the predicted values.

**Base concepts:**

Linear Regression is a supervised machine-learning algorithm used to predict continuous numerical values by fitting a linear relationship between input and output variables.

Simple equation:

y = mx + c

Multiple Linear Regression:

y = b0 + b1x1 + b2x2 + ... + bnxn

- y: Predicted fare.
- x: Input feature.
- b: Coefficient.
- b0: Intercept.

**Viva Questions:**

Q1. What is Linear Regression?  
Ans: A supervised learning algorithm used to predict continuous numerical values.

Q2. Is Linear Regression supervised or unsupervised?  
Ans: Supervised learning.

Q3. Why is it called regression?  
Ans: Because it predicts continuous numerical values.

Q4. What does `fit()` do?  
Ans: Trains the model.

Q5. What does `predict()` do?  
Ans: Predicts output values for input data.

Q6. What is an intercept?  
Ans: Predicted output when all input features are zero.

Q7. What is a coefficient?  
Ans: It represents how the predicted output changes with a one-unit change in a feature, holding other features constant.

---

## BLOCK 10 – RANDOM FOREST REGRESSION MODEL

```python
random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_prediction = random_forest.predict(X_test)
```

**Explanation:**

1. `RandomForestRegressor()` creates a Random Forest Regression model.
2. `n_estimators=100` builds 100 decision trees.
3. `random_state=42` makes random operations reproducible.
4. `n_jobs=-1` uses all available CPU cores where supported.
5. `.fit()` trains the Random Forest model.
6. `.predict()` predicts fare amounts for testing data.
7. `rf_prediction` stores predicted values.

**Base concepts:**

- Decision Tree: A tree-like model that makes predictions by splitting data according to conditions.
- Random Forest: An ensemble algorithm combining multiple decision trees.
- Ensemble Learning: Combining multiple models to produce a prediction.

For regression, Random Forest generally combines the predictions of its trees by averaging them.

**Viva Questions:**

Q1. What is Random Forest?  
Ans: An ensemble machine-learning algorithm combining multiple decision trees.

Q2. Is Random Forest supervised or unsupervised?  
Ans: Supervised learning.

Q3. Why is it called Random Forest?  
Ans: Because it builds multiple decision trees using randomness in data sampling and feature selection.

Q4. What does `n_estimators=100` mean?  
Ans: The model builds 100 decision trees.

Q5. How does Random Forest predict numerical values?  
Ans: It generally averages the predictions from its trees.

Q6. Difference between Decision Tree and Random Forest?  
Ans: Decision Tree uses one tree, whereas Random Forest combines multiple trees.

Q7. Is Random Forest always better than Linear Regression?  
Ans: No. Performance depends on the dataset, features and model settings.

Q8. What does `n_jobs=-1` mean?  
Ans: Uses all available CPU cores for parallel processing where supported.

---

## BLOCK 11 – EVALUATE MODELS

```python
def evaluate_model(name, actual, predicted):
    r2 = r2_score(actual, predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print("\n" + name)
    print("---------------------------")

    print("R2 Score :", round(r2, 4))
    print("MAE      :", round(mae, 4))
    print("RMSE     :", round(rmse, 4))

    return r2, mae, rmse


linear_scores = evaluate_model(
    "Linear Regression", y_test, linear_prediction
)

rf_scores = evaluate_model(
    "Random Forest Regression", y_test, rf_prediction
)
```

**Explanation:**

1. `evaluate_model()` defines a reusable evaluation function.
2. `name` stores the model name.
3. `actual` stores the actual values.
4. `predicted` stores predicted values.
5. `r2_score()` calculates the R² score.
6. `mean_absolute_error()` calculates MAE.
7. `mean_squared_error()` calculates MSE.
8. `np.sqrt()` calculates the square root of MSE to obtain RMSE.
9. `round(value, 4)` rounds the result to four decimal places.
10. `return r2, mae, rmse` returns all three metrics.
11. The function is called separately for Linear Regression and Random Forest Regression.

**Important formulas:**

1. R² Score:

Measures how much of the variation in the target is explained by the model.

- Closer to 1: Better fit.
- Around 0: Similar to predicting the mean, in terms of R².
- Negative: Worse than predicting the mean on the test set.

2. Mean Absolute Error:

MAE = (Sum of absolute errors) / Number of observations.

It measures average absolute prediction error.

3. Root Mean Squared Error:

RMSE = Square Root of Mean Squared Error.

It penalizes large prediction errors more strongly than MAE.

**Viva Questions:**

Q1. Why evaluate a model?  
Ans: To measure how accurately it predicts unseen data.

Q2. What is R² score?  
Ans: A metric that measures how much target variation is explained by the model.

Q3. What is MAE?  
Ans: Average absolute difference between actual and predicted values.

Q4. What is RMSE?  
Ans: Square root of average squared prediction error.

Q5. Which is better: lower MAE or higher MAE?  
Ans: Lower MAE.

Q6. Which is better: lower RMSE or higher RMSE?  
Ans: Lower RMSE.

Q7. Can R² be negative?  
Ans: Yes. It can be negative when the model performs worse than the mean-prediction baseline.

Q8. Why use both MAE and RMSE?  
Ans: MAE measures average absolute error, while RMSE penalizes larger errors more strongly.

Q9. Difference between actual and predicted values?  
Ans: Actual values are true fare amounts; predicted values are the model's estimated fare amounts.

---

## BLOCK 12 – COMPARE THE MODELS

```python
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest Regression"],

    "R2 Score": [linear_scores[0], rf_scores[0]],

    "MAE": [linear_scores[1], rf_scores[1]],

    "RMSE": [linear_scores[2], rf_scores[2]]
})

print("Model Comparison:")
display(results)
```

**Explanation:**

1. `pd.DataFrame()` creates a comparison table.
2. `"Model"` stores the names of the two algorithms.
3. `"R2 Score"` stores the R² scores.
4. `"MAE"` stores the MAE values.
5. `"RMSE"` stores the RMSE values.
6. Index `[0]` represents R².
7. Index `[1]` represents MAE.
8. Index `[2]` represents RMSE.
9. `display(results)` displays the comparison table in Jupyter Notebook.

**Viva Questions:**

Q1. Why compare two models?  
Ans: To understand which model performs better on testing data.

Q2. Which model is better?  
Ans: Generally, the model with higher R² and lower MAE and RMSE is preferred.

Q3. Can one model have better R² but worse MAE?  
Ans: Yes. Different metrics measure different aspects of prediction performance.

Q4. Why use a DataFrame for comparison?  
Ans: It presents results in a structured table for easy comparison.

---

## BLOCK 13 – ACTUAL VS PREDICTED FARE GRAPH

```python
plt.figure(figsize=(8, 5))

plt.scatter(y_test, rf_prediction, alpha=0.5)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")

plt.title("Actual vs Predicted Fare - Random Forest")

plt.show()

print("Uber Fare Prediction Completed Successfully!")
```

**Explanation:**

1. `plt.figure(figsize=(8, 5))` sets graph size.
2. `plt.scatter()` creates a scatter plot.
3. `y_test` represents actual fare values on the X-axis.
4. `rf_prediction` represents predicted fare values on the Y-axis.
5. `alpha=0.5` controls transparency of points.
6. `plt.xlabel()` labels the X-axis.
7. `plt.ylabel()` labels the Y-axis.
8. `plt.title()` adds the graph title.
9. `plt.show()` displays the graph.
10. The final print statement displays the completion message.

**How to interpret the graph:**

- Points closer to the diagonal line represent more accurate predictions.
- Points far away from the line represent larger prediction errors.

**Viva Questions:**

Q1. What is a scatter plot?  
Ans: A graph displaying the relationship between two numerical variables using points.

Q2. What is the X-axis?  
Ans: Actual fare.

Q3. What is the Y-axis?  
Ans: Predicted fare.

Q4. What does `alpha=0.5` do?  
Ans: Makes plotted points partially transparent.

Q5. What does a point near the diagonal indicate?  
Ans: Predicted fare is close to actual fare.

---

# 4. IMPORTANT MACHINE-LEARNING BASE CONCEPTS

## 4.1 What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data and make predictions without explicitly programming every rule.

## 4.2 Types of Machine Learning

1. Supervised Learning: Learns from labelled data. Example: Uber fare prediction.
2. Unsupervised Learning: Finds patterns in unlabelled data. Example: Customer clustering.
3. Reinforcement Learning: Learns through rewards and penalties. Example: Game-playing AI.

Our project uses Supervised Learning because the dataset contains actual fare amounts.

## 4.3 Classification vs Regression

Classification:
- Predicts categories.
- Example: Spam or not spam.
- Output is a class.

Regression:
- Predicts numerical values.
- Example: Uber fare prediction.
- Output is a continuous numerical value.

Our project uses Regression because fare amount is numerical.

## 4.4 Important Terminology

- Dataset: Collection of data.
- Feature: Input variable.
- Target: Value to predict.
- Model: Algorithm trained on data.
- Training: Learning patterns from data.
- Testing: Evaluating using unseen data.
- Prediction: Model's estimated output.
- Overfitting: Model learns training data too closely and performs poorly on unseen data.
- Underfitting: Model fails to learn important patterns.
- Generalization: Ability to perform well on unseen data.
- Feature Selection: Selecting relevant input variables.
- Feature Engineering: Creating or transforming features to improve a model.
- Data Leakage: When information that should not be available during prediction enters the training process, resulting in overly optimistic evaluation.

## 4.5 Important Python Functions

- `pd.read_csv()` – Reads CSV file.
- `df.head()` – Displays first five rows.
- `df.tail()` – Displays last five rows.
- `df.shape` – Returns rows and columns.
- `df.info()` – Displays dataset information.
- `df.isnull().sum()` – Counts missing values.
- `df.dropna()` – Removes missing values.
- `df.drop_duplicates()` – Removes duplicate rows.
- `pd.to_numeric()` – Converts values to numeric.
- `np.radians()` – Converts degrees into radians.
- `np.sqrt()` – Calculates square root.
- `np.sin()` – Calculates sine.
- `np.cos()` – Calculates cosine.
- `np.arcsin()` – Calculates inverse sine.
- `train_test_split()` – Splits dataset.
- `.fit()` – Trains model.
- `.predict()` – Predicts output.
- `plt.show()` – Displays graph.

---

# 5. ADDITIONAL VIVA QUESTIONS

Q1. What is the aim of your project?  
Ans: To predict Uber ride fare amounts using pickup and drop-off coordinates, passenger count and calculated distance.

Q2. Which dataset did you use?  
Ans: Uber Fares Dataset from Kaggle.

Q3. What is the target variable?  
Ans: `fare_amount`.

Q4. What are the input features?  
Ans: Passenger count, pickup longitude, pickup latitude, drop-off longitude, drop-off latitude and calculated distance.

Q5. Why did you calculate distance?  
Ans: Because distance travelled is an important factor influencing fare.

Q6. Why did you use two algorithms?  
Ans: To compare Linear Regression and Random Forest Regression performance.

Q7. Which algorithm is simpler?  
Ans: Linear Regression.

Q8. Which algorithm can capture complex non-linear relationships?  
Ans: Random Forest Regression.

Q9. What is non-linear data?  
Ans: Data where the relationship between input and output cannot be adequately represented by a straight line.

Q10. What happens if a dataset contains too many missing values?  
Ans: It can reduce useful training data and negatively affect model performance.

Q11. What is feature selection?  
Ans: Selecting relevant input variables for training a model.

Q12. What is feature engineering?  
Ans: Creating or transforming features to improve a machine-learning model. Calculating `distance_km` is feature engineering in our project.

Q13. What is data leakage?  
Ans: When information that should not be available during prediction enters model training or evaluation.

Q14. Does this project predict actual road distance?  
Ans: No. It calculates straight-line geographical distance using the Haversine formula.

Q15. Can this model predict fare for a new ride?  
Ans: Yes, if we provide the required input features in the same format used during training. The current code demonstrates predictions on test data.

Q16. Why divide the dataset into training and testing data?  
Ans: To check whether the model can predict values it has not seen during training.

Q17. Difference between `fit()` and `predict()`?  
Ans: `fit()` trains the model; `predict()` generates predictions using the trained model.

Q18. Difference between MAE and MSE?  
Ans: MAE averages absolute errors; MSE averages squared errors.

Q19. Difference between MSE and RMSE?  
Ans: RMSE is the square root of MSE and is expressed in the same unit as the target.

Q20. What is an ensemble algorithm?  
Ans: An algorithm that combines multiple models to make a prediction. Random Forest is an ensemble algorithm.

Q21. Is Random Forest always better than Linear Regression?  
Ans: No. The result depends on the dataset and model settings.

Q22. What is the difference between actual and predicted fare?  
Ans: Actual fare is the real value from the dataset, while predicted fare is estimated by the model.

---

# 6. IMPORTANT LIMITATIONS AND IMPROVEMENTS

1. Small dataset: A 20-row demo dataset is insufficient for reliable model evaluation. Use the full Kaggle dataset for the actual practical.
2. Straight-line distance: Haversine distance is not road distance.
3. Outlier removal: IQR can remove genuine long-distance rides.
4. Missing time information: `pickup_datetime` is not included as a feature, even though time can influence fares.
5. Location validation: Global coordinate ranges do not confirm that a coordinate is actually within the service area.

**If the examiner asks how to improve the project:**

Ans: We can improve the model by using a larger dataset, including pickup time and location-based features, using actual road distance, and tuning model parameters.

---

# 7. EXPLAIN THE ENTIRE PROJECT IN 60 SECONDS

"My project is Uber Fare Prediction using Machine Learning.

The main objective is to predict the fare amount of an Uber ride based on its pickup and drop-off locations and passenger count.

First, I loaded the Kaggle dataset using Pandas. Then I performed preprocessing by removing missing values, duplicates and invalid records.

After that, I calculated the distance between pickup and drop-off locations using the Haversine formula.

I identified and removed potential outliers using the IQR method. Then I performed correlation analysis using a heatmap.

Next, I divided the dataset into training and testing data in an 80:20 ratio.

I trained two models: Linear Regression and Random Forest Regression.

Finally, I evaluated and compared both models using R² score, MAE and RMSE, and displayed an actual-versus-predicted graph."

---

# 8. LAST-MINUTE REVISION TABLE

| Topic | Remember |
|---|---|
| Project type | Supervised Learning |
| Problem type | Regression |
| Dataset | Uber Fares |
| Target | `fare_amount` |
| Distance formula | Haversine |
| Outlier method | IQR |
| Correlation range | -1 to +1 |
| Data split | 80% training, 20% testing |
| Models | Linear Regression, Random Forest |
| Number of trees | 100 |
| Evaluation | R², MAE, RMSE |
| Better R² | Higher |
| Better MAE | Lower |
| Better RMSE | Lower |

**Most important blocks to revise:** 5 (Haversine), 6 (IQR), 8 (Train-Test Split), 9 (Linear Regression), 10 (Random Forest), and 11 (Evaluation).

Remember: Be able to explain why each algorithm is used, not just what its Python function does.
