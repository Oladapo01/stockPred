import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_regressor_model(X_train, y_train, n_estimator=100):
    # Initialize the model
    model = RandomForestRegressor(n_estimators=n_estimator, random_state=0)

    # Fit the model
    model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    # Predict the price
    y_pred = model.predict(X_test)

    return y_pred

def evaluate(y_test, y_pred):
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse