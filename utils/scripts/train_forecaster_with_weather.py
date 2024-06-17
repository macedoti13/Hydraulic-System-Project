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
FORECASTING_DATASET_PATH = CURATED_DATASETS_PATH / "forecasting_dataset.parquet"
MODEL_SAVING_PATH = MODELS_PATH / "forecaster_with_weather.pkl"

# Functions
def set_training_data(df):
    columns_to_exclude = ["id", "timestamp"]
    training_df = df.drop(columns=columns_to_exclude)
    
    return training_df

def train_test_split(original_df):
    training_df = original_df.copy()
    train = training_df[training_df["year"] == 2023]
    test = training_df[training_df["year"] == 2024]

    # List of features for training (excluding "output_flow_rate")
    training_features = [col for col in training_df.columns if col != "output_flow_rate"]

    # Create X and y
    X_train = train[training_features]
    y_train = train["output_flow_rate"]
    X_test = test[training_features]
    y_test = test["output_flow_rate"]
    
    return X_train, y_train, X_test, y_test

def set_model_training_pipeline() -> GridSearchCV:
    model = XGBRegressor(n_estimators=100000, learning_rate=0.01, early_stopping_rounds=100)
    cv = TimeSeriesSplit(n_splits=5)
    params = {"n_estimators": [100, 500, 1000, 5000], "max_depth": [3, 5, 10, 14], "learning_rate": [0.01, 0.05, 0.1]}
    clf = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
    
    return clf

def train_model(clf: GridSearchCV, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> GridSearchCV:
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)
    return clf.best_estimator_

def calculate_error(y_test: pd.Series, y_pred: pd.Series):
    mae = round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 2)
    mse = round(mean_squared_error(y_true=y_test, y_pred=y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    r2 = round(r2_score(y_true=y_test, y_pred=y_pred), 2)
    
    return mae, mse, rmse, r2

def predict_with_model(clf: GridSearchCV, X: pd.DataFrame) -> pd.Series:
    y_pred = clf.predict(X)
    return pd.Series(y_pred)

def evaluate_model(y_test, X_test, clf):
    mae, mse, rmse, r2 = calculate_error(y_test, predict_with_model(clf, X_test))
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

def main():
    df = pd.read_parquet(FORECASTING_DATASET_PATH)
    training_df = set_training_data(df)
    X_train, y_train, X_test, y_test = train_test_split(training_df)
    clf = set_model_training_pipeline()
    clf = train_model(clf, X_train, y_train, X_test, y_test)
    pickle.dump(clf, open(MODEL_SAVING_PATH, 'wb'))
    print()
    evaluate_model(y_test, X_test, clf)
    
if __name__ == "__main__":
    main()
