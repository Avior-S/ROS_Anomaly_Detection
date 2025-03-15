import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("/mnt/data/example.xlsx")  # Replace with your actual time series data

# Assuming "angular velocity z" is the time series variable
time_series_col = "angular velocity z"

# Step 1: Create Lag Features
def create_lag_features(data, column, lags=5):
    df_lagged = data.copy()
    for lag in range(1, lags + 1):
        df_lagged[f"{column}_lag_{lag}"] = df_lagged[column].shift(lag)
    return df_lagged.dropna()

df = create_lag_features(df, time_series_col, lags=5)

# Define features (past values) and target (next value)
features = [col for col in df.columns if "lag" in col]
target = time_series_col

X = df[features]
y = df[target]

# Step 2: Split into Train/Test (Train on Normal Data Only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 3: Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 4: Predict on Test Data
y_pred = xgb_model.predict(X_test)

# Step 5: Compute Residuals (Prediction Errors)
residuals = np.abs(y_test - y_pred)

# Step 6: Define Anomaly Threshold (e.g., 95th percentile)
threshold = np.percentile(residuals, 95)

# Flag anomalies
anomalies = residuals > threshold

# Print anomaly statistics
print(f"Anomaly threshold: {threshold:.4f}")
print(f"Detected anomalies: {anomalies.sum()} out of {len(y_test)}")

# Step 7: Plot Results
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label="Actual", color="blue")
plt.plot(y_test.index, y_pred, label="Predicted", color="green")
plt.scatter(y_test.index[anomalies], y_test[anomalies], color="red", label="Anomalies")
plt.xlabel("Time")
plt.ylabel(time_series_col)
plt.legend()
plt.title("Anomalies Detected Using XGBoost Regression")
plt.show()
