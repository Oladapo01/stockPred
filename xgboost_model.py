import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_xgboost_model(data, label_column, test_size=0.2, random_state=42):
    # Split data into features and target
    X = data.drop(columns=[label_column])
    y = data[label_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and fit the model
    model = xgb.XGBRegressor(objective='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 5,
                             alpha = 10,
                             n_estimators=1000
                             )

    # Fit the model
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return model, rmse, X_test, y_test, y_pred

def predict_xgboost(model, data):
    # Predict the test set
    y_pred = model.predict(data)

    return y_pred


def forecast_future_months(model, last_known_data,  months):
    future_predictions = []
    current_features = last_known_data.copy()

    for _ in range(months):
        # Predict the next month
        y_pred = model.predict(current_features)
        future_predictions.append(y_pred[0])

        # Create new features for the next prediction
        next_features = generate_features_for_next_prediction(current_features.iloc[-1])

        # Append the new features for the next prediction
        # Ensuring that 'current_features' remains a DataFrame
        current_features = pd.concat([current_features, pd.DataFrame([next_features])]).reset_index(drop=True)

    return future_predictions


def generate_features_for_next_prediction(last_row):
    # Create a new row for prediction with the necessary feature transformation
    new_row = last_row.copy()

    return new_row


