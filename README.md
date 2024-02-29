# Housing Price Prediction with RandomForestRegressor

This project aims to predict housing prices using the RandomForestRegressor algorithm. It involves data preprocessing, model training, hyperparameter tuning, and evaluation.

## Data

The dataset used for this project is named `housing.csv`. It contains information about housing prices, including features such as longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, and ocean proximity.

## Preprocessing

1. Load the data.
2. Drop rows with missing values.
3. Encode categorical variable (`ocean_proximity`) using one-hot encoding.
4. Split features and target variable (`median_house_value`).
5. Split the data into train and test sets.
6. Standardize features using `StandardScaler`.

## Correlation Analysis

A correlation matrix is computed to understand the relationships between variables. A heatmap of the correlation matrix is plotted using `seaborn`.

## Model Training

1. Initialize a `RandomForestRegressor` model.
2. Train the model on the training data.
3. Evaluate the model on the training and test sets.

## Hyperparameter Tuning

Hyperparameters of the RandomForestRegressor model are tuned using `GridSearchCV`. The hyperparameters tuned include `n_estimators`, `min_samples_split`, and `max_depth`.

## Evaluation

The best model obtained from hyperparameter tuning is evaluated on the test set.

## Usage

To run the code:

1. Ensure you have the necessary libraries installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
2. Place the `housing.csv` file in the same directory as the Python script.
3. Run the script.

```python
python housing_prediction.py
