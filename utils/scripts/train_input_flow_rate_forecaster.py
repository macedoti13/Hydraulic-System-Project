from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "data"))

# Paths
MODELS_PATH = project_root / "models"
CURATED_DATASETS_PATH = project_root / "data/curated_datasets"
WATER_CONSUMPTION_CURATED_PATH = CURATED_DATASETS_PATH / "water_consumption_curated.parquet"
MODEL_SAVING_PATH = MODELS_PATH / "input_flow_forecaster.pkl"

def create_training_samples(df, target_column='input_flow_rate', window_size=576):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[target_column].values[i-window_size:i])
        y.append(df[target_column].values[i])
    return np.array(X), np.array(y)

def set_model_training_pipeline() -> GridSearchCV:
    model = XGBRegressor(learning_rate=0.01, early_stopping_rounds=100)
    cv = TimeSeriesSplit(n_splits=5)
    params = {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.01]}
    clf = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
    return clf

def train_model(clf: GridSearchCV, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> GridSearchCV:
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    return clf

def calculate_error(y_test: np.ndarray, y_pred: np.ndarray):
    mae = round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 2)
    mse = round(mean_squared_error(y_true=y_test, y_pred=y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    r2 = round(r2_score(y_true=y_test, y_pred=y_pred), 2)
    return mae, mse, rmse, r2

def main():
    df = pd.read_parquet(WATER_CONSUMPTION_CURATED_PATH)
    
    # Split data into 2023 and 2024 sets
    df_train = df[df['year'] == 2023]
    df_test = df[df['year'] == 2024]
    
    # Create training samples
    X_train, y_train = create_training_samples(df_train)
    X_test, y_test = create_training_samples(df_test)
    
    clf = set_model_training_pipeline()
    grid_search_result = train_model(clf, X_train, y_train, X_test, y_test)
    
    # Get the best model from GridSearchCV
    best_model = grid_search_result.best_estimator_
    
    # Save the trained model
    with open(MODEL_SAVING_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mae, mse, rmse, r2 = calculate_error(y_test, y_pred)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

if __name__ == "__main__":
    main()
