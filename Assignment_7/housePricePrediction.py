import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the data
train_data = pd.read_csv('F:\\Celebal\\Assignment_7\\train.csv')
test_data = pd.read_csv('F:\\Celebal\\Assignment_7\\test.csv')
submission_data = pd.read_csv('F:\\Celebal\\Assignment_7\\sample_submission.csv')

# Separate features and target variable from training data
X = train_data.drop(['Id', 'SalePrice'], axis=1)
y = train_data['SalePrice']
test_ids = test_data['Id']
X_test = test_data.drop(['Id'], axis=1)

# Identify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

# Split data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocess and fit the model
pipeline.fit(X_train, y_train)

# Get predictions
preds = pipeline.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, preds)
print(f'Mean Absolute Error: {mae}')

# Preprocess and predict on test data
test_preds = pipeline.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)

print("Submission file has been created successfully.")
