# predict_model.py

import joblib
import pandas as pd
import sqlite3
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Loading the pre-trained XGBoost model
xgb_model = joblib.load('xgb_model.pkl')

# Connecting to the SQLite database to load data
conn = sqlite3.connect('C:\\Users\\Sarabjeet Kour\\Database (1).db')
data = pd.read_sql_query('SELECT * FROM Heart_disease', conn)
conn.close()

# Data Pre-processing (same as during training)
categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory','Race', 
                        'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']

# Using One-hot encoding for the categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Converting boolean columns to integers (0 and 1)
bool_columns = data_encoded.select_dtypes(include='bool').columns
data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

# Converting 'HeartDisease' to 1/0
data_encoded['HeartDisease'] = data_encoded['HeartDisease'].map({'Yes': 1, 'No': 0})

# Converting other columns to numeric (for columns like BMI, PhysicalHealth, etc.)
data_encoded['BMI'] = pd.to_numeric(data_encoded['BMI'], errors='coerce')
data_encoded['PhysicalHealth'] = pd.to_numeric(data_encoded['PhysicalHealth'], errors='coerce')
data_encoded['MentalHealth'] = pd.to_numeric(data_encoded['MentalHealth'], errors='coerce')
data_encoded['SleepTime'] = pd.to_numeric(data_encoded['SleepTime'], errors='coerce')

# Handling missing values by imputing with median strategy
imputer = SimpleImputer(strategy='median')
data_encoded['BMI'] = imputer.fit_transform(data_encoded[['BMI']])
data_encoded['PhysicalHealth'] = imputer.fit_transform(data_encoded[['PhysicalHealth']])

# Checking if there are any missing values after imputation
missing_values = data_encoded.isnull().sum()
if missing_values[missing_values > 0].any():
    print("There are still missing values after imputation.")

# Ensuring the columns match the expected format
expected_columns = [
    'BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 
    'Smoking_No', 'Smoking_Yes', 'AlcoholDrinking_Yes', 'Stroke_Yes', 
    'DiffWalking_No', 'DiffWalking_Yes', 'Sex_Female', 'Sex_Male', 
    'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 
    'AgeCategory_40-44', 'AgeCategory_45-49', 'AgeCategory_50-54', 
    'AgeCategory_55-59', 'AgeCategory_60-64', 'AgeCategory_65-69', 
    'AgeCategory_70-74', 'AgeCategory_75-79', 'AgeCategory_80 or older', 
    'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White', 
    'Diabetic_No', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes', 
    'Diabetic_Yes (during pregnancy)', 'PhysicalActivity_Yes', 'GenHealth_Excellent', 
    'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good', 
    'Asthma_No', 'Asthma_Yes', 'KidneyDisease_Yes', 'SkinCancer_No', 'SkinCancer_Yes'
]

# Adding missing columns with default value of 0
for col in expected_columns:
    if col not in data_encoded.columns:
        data_encoded[col] = 0

# Re-ordering the columns to match the training data
data_encoded = data_encoded[expected_columns]

# Making predictions using the trained XGBoost model
y_pred = xgb_model.predict(data_encoded)
y_pred_proba = xgb_model.predict_proba(data_encoded)  # This gives probabilities for each class

# Prediction output
print("Prediction Results...")
print()
print(f"Prediction (Raw Output): {'Yes' if y_pred[0] == 1 else 'No'}")
print(f"Prediction Probability (Heart Disease): {y_pred_proba[0][1]:.4f}")
print(f"Prediction Probability (No Heart Disease): {y_pred_proba[0][0]:.4f}")
