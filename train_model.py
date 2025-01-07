# train_model.py

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Connecting to database.db
conn = sqlite3.connect('C:\\Users\\Sarabjeet Kour\\Database (1).db')
data = pd.read_sql_query('SELECT * FROM Heart_disease', conn)
conn.close()

print("Connected to database...")

# Data Pre-processing
categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory','Race', 
                        'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Converting all boolean columns to integers (0 and 1)
bool_columns = data_encoded.select_dtypes(include='bool').columns
data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

# Converting 'HeartDisease' from 'Yes'/'No' to 1/0
data_encoded['HeartDisease'] = data_encoded['HeartDisease'].map({'Yes': 1, 'No': 0})

# Converting columns to numeric
data_encoded['BMI'] = pd.to_numeric(data_encoded['BMI'], errors='coerce')
data_encoded['PhysicalHealth'] = pd.to_numeric(data_encoded['PhysicalHealth'], errors='coerce')
data_encoded['MentalHealth'] = pd.to_numeric(data_encoded['MentalHealth'], errors='coerce')
data_encoded['SleepTime'] = pd.to_numeric(data_encoded['SleepTime'], errors='coerce')

# Imputing missing values
imputer = SimpleImputer(strategy='median')
data_encoded['BMI'] = imputer.fit_transform(data_encoded[['BMI']])
data_encoded['PhysicalHealth'] = imputer.fit_transform(data_encoded[['PhysicalHealth']])

print()
print("Data preocessing completed successfully...")

# Splitting the data into features (X) and target (y)
X = data_encoded.drop(columns=['HeartDisease'])
y = data_encoded['HeartDisease']

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print()
print("Model training started using XGBOOST...")
print()

# Initializing the XGBoost classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')

# Training the model
xgb_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print()

# Saving the trained model
joblib.dump(xgb_model, 'xgb_model.pkl')
print("Model saved as 'xgb_model.pkl'")
