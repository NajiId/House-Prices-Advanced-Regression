# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv('housing.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical variable
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# Split features and target variable
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Correlation analysis
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Train RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = forest.score(X_train_scaled, y_train)
test_score = forest.score(X_test_scaled, y_test)
print("Train Score:", train_score)
print("Test Score:", test_score)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the best model on test data
best_model = grid_search.best_estimator_
test_score_best = best_model.score(X_test_scaled, y_test)
print("Best Model Test Score:", test_score_best)
