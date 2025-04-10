# This code is part of the Introduction to Machine learning course project 2
# It has been written using copilot to help with the implementation of the project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_excel("data/Concrete_Data.xls")  

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the index of the smallest y value
min_y_index = np.argmin(y)

# Print the corresponding scaled X values
print(f"Smallest y: {y[min_y_index]}, X_scaled for the smallest y: {X_scaled[min_y_index]}")


lambda_values = np.linspace(0, 4, 21)  # Defines 21 lambda values between 0 and 4
errors = []

# Perform k fold validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for lam in lambda_values:
    ridge = Ridge(alpha=lam)
    fold_errors = []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        ridge.fit(X_train, y_train)
        
        y_pred = ridge.predict(X_test)
        
        fold_errors.append(mean_squared_error(y_test, y_pred))
    
    errors.append(np.mean(fold_errors))

# Plot Generalization Error vs. Lambda
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, errors, marker='o', linestyle='-')
plt.xlabel("Lambda")
plt.ylabel("Generalization Error")
plt.title("Generalization Error vs. Regularization Parameter")
plt.grid(True)
plt.show()

# Best lambda
best_lambda = lambda_values[np.argmin(errors)]
print(f"Optimal lambda: {best_lambda}")

# Final model with best lambda
best_model = Ridge(alpha=best_lambda)
best_model.fit(X_scaled, y)

print(f"Model Coefficients: {best_model.coef_}")
print(f"Model intercept: {best_model.intercept_}")
print(f"Model: {best_model}")


y_pred = best_model.predict(X_scaled)

# Plot of prediction
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.7, color='blue', label="Predictions")
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label="Perfect Prediction")  # y = x line
plt.xlabel("Actual Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.title("Actual vs. Predicted Concrete Strength")
plt.legend()
plt.show()