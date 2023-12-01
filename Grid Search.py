import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
data = pd.read_excel(r"C:\Users\eugen\PycharmProjects\Hunger XGBoost\Data\Database_ver3.xlsx")

# Fill missing values with the median of the respective column
# Separate numeric and non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])
non_numeric_data = data.select_dtypes(exclude=[np.number])

# Fill NaN values in numeric columns with the median
numeric_filled = numeric_data.fillna(numeric_data.median())

# Combine them back (assuming you don't need to fill NaN in non-numeric columns)
data_filled = pd.concat([non_numeric_data, numeric_filled], axis=1)

# Check the first few rows of the filled data
print(data_filled.head())

xgb = XGBRegressor(
    colsample_bytree=0.6,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=3,
    n_estimators=500,
    objective='reg:squarederror',
    subsample=0.7,
    random_state=42,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1,  # L2 regularization
    gamma=0.5  # Complexity control
)
# Define your XGBRegressor
gb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'max_depth': [6, 8, 10],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.3, 0.5, 0.7],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 300, 500],
    'reg_alpha': [0.05, 0.1, 0.2],
    'reg_lambda': [0.5, 1, 1.5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

X = data_filled[["Population", "GDP_2017", "GDP", "SOFI", "Inflation", "GINI"]]
y = data_filled["Abs"]

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best estimator to make predictions
y_pred_best = grid_search.best_estimator_.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred_best)
mse = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_best)

# Print performance metrics
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)