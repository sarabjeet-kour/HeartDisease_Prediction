# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sqlite3
from sklearn.impute import SimpleImputer
import xgboost as xgb

app = Flask(__name__)

# Load the pre-trained XGBoost model
xgb_model = joblib.load('xgb_model.pkl')

# Define the route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Connect to the SQLite database to load data
    conn = sqlite3.connect('C:\\Users\\Sarabjeet Kour\\Database (1).db')
    data = pd.read_sql_query('SELECT * FROM Heart_disease', conn)
    conn.close()

    # Data Pre-processing (same as during training)
    categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory','Race', 
                            'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']

    # One-hot encode the categorical columns
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Convert boolean columns to integers (0 and 1)
    bool_columns = data_encoded.select_dtypes(include='bool').columns
    data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

    # Convert 'HeartDisease' to 1/0
    data_encoded['HeartDisease'] = data_encoded['HeartDisease'].map({'Yes': 1, 'No': 0})

    # Convert other columns to numeric (for columns like BMI, PhysicalHealth, etc.)
    data_encoded['BMI'] = pd.to_numeric(data_encoded['BMI'], errors='coerce')
    data_encoded['PhysicalHealth'] = pd.to_numeric(data_encoded['PhysicalHealth'], errors='coerce')
    data_encoded['MentalHealth'] = pd.to_numeric(data_encoded['MentalHealth'], errors='coerce')
    data_encoded['SleepTime'] = pd.to_numeric(data_encoded['SleepTime'], errors='coerce')

    # Handle missing values by imputing with median strategy
    imputer = SimpleImputer(strategy='median')
    data_encoded['BMI'] = imputer.fit_transform(data_encoded[['BMI']])
    data_encoded['PhysicalHealth'] = imputer.fit_transform(data_encoded[['PhysicalHealth']])

    # Ensure the columns match the expected format (add missing columns)
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

    # Add missing columns with default value of 0
    for col in expected_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Reorder columns to match the training data
    data_encoded = data_encoded[expected_columns]

    # Make predictions using the trained XGBoost model
    y_pred = xgb_model.predict(data_encoded)
    y_pred_proba = xgb_model.predict_proba(data_encoded)  # This gives probabilities for each class

    # Return the prediction result as JSON
    result = {
        "Prediction (Raw Output)": "Yes" if y_pred[0] == 1 else "No",
        "Prediction Probability (Heart Disease)": f"{y_pred_proba[0][1]:.4f}",
        "Prediction Probability (No Heart Disease)": f"{y_pred_proba[0][0]:.4f}"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
