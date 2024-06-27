import pandas as pd
import plotly.graph_objects as go
from utils import all_features, all_non_weather_features
import pickle

df1 = pd.read_parquet("../data/silver/training_dataset.parquet")
df2 = pd.read_parquet("../data/silver/water_consumption_silver.parquet")
df_test_weather = df1[(df1.year==2024)&(df1.inputed_row==False)][all_features]
df_test_no_weather = df1[(df1.year==2024)&(df1.inputed_row==False)][all_non_weather_features]
df_response = df1[(df1.year==2024)&(df1.inputed_row==False)]['target_1'].to_frame().reset_index(drop=True)
model_weather = pickle.load(open("../models/weather/xgb_weather_1h.pkl", "rb"))
model_no_weather = pickle.load(open("../models/no_weather/xgb_1h.pkl", "rb"))
predictions_weather = model_weather.predict(df_test_weather)
df_response["prediction_weather"] = predictions_weather
predictions_no_weather = model_no_weather.predict(df_test_no_weather)
df_response["prediction_no_weather"] = predictions_no_weather
df_response.to_parquet("../data/gold/model_prediction_evaluation.parquet")