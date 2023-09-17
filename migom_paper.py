import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Load the dataset containing historical electricity consumption and related variables
# Replace 'your_dataset.csv' with the actual dataset file path
data = pd.read_csv('your_dataset.csv')

# Preprocess the data
# You may need to clean and transform the data based on the characteristics mentioned in the paper.

# Extract features and target variable
X = data.drop(columns=['Electricity_Consumption'])  # Features
y = data['Electricity_Consumption']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Create and train an XGBoost regressor
xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_regressor.fit(X_train, y_train)

# Make predictions using individual models
rf_predictions = rf_regressor.predict(X_test)
xgb_predictions = xgb_regressor.predict(X_test)

# Create an ensemble by averaging predictions
ensemble_predictions = (rf_predictions + xgb_predictions) / 2

# Calculate the root mean squared error (RMSE) to evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
print(f'Ensemble RMSE: {rmse}')

# Optionally, you can compare the performance of individual models (RF and XGBoost) as well
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

print(f'Random Forest RMSE: {rf_rmse}')
print(f'XGBoost RMSE: {xgb_rmse}')
