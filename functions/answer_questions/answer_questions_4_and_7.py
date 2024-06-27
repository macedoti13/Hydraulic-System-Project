import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from utils import all_features, all_non_weather_features
import pandas as pd
import pickle
import os

def forecast_next_24_hours_output_flow_rate(year, month, day, hour, save_df=True):
    water_consumption_silver = pd.read_parquet(os.path.join(os.path.dirname(__file__),"../../data/silver/water_consumption_silver.parquet"))
    original_input_df = pd.read_parquet(os.path.join(os.path.dirname(__file__),"../../data/silver/training_dataset.parquet"))
    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    input_df = original_input_df[original_input_df.timestamp == timestamp]
    
    X = input_df[all_features]
    X_no_weather =  input_df[all_non_weather_features] 
    
    # no weather predictions
    predictions = []
    for i in range(1, 25):
        new_prediction = {}
        next_timestamp = timestamp + pd.Timedelta(hours=i)
        model = pickle.load(open(os.path.join(os.path.dirname(__file__),f"../../models/no_weather/xgb_{i}h.pkl"), "rb"))
        change_from_last_hour_output_flow_rate_mean = model.predict(X_no_weather)[0]
        new_prediction["timestamp"] = next_timestamp
        new_prediction["change_from_last_hour_output_flow_rate_mean"] = change_from_last_hour_output_flow_rate_mean
        predictions.append(new_prediction)
        
    # weather predictions
    weather_predictions = []
    for i in range(1, 25):
        new_prediction = {}
        next_timestamp = timestamp + pd.Timedelta(hours=i)
        model = pickle.load(open(os.path.join(os.path.dirname(__file__),f"../../models/weather/xgb_weather_{i}h.pkl"), "rb"))
        change_from_last_hour_output_flow_rate_mean = model.predict(X)[0]
        new_prediction["timestamp"] = next_timestamp
        new_prediction["change_from_last_hour_output_flow_rate_mean"] = change_from_last_hour_output_flow_rate_mean
        weather_predictions.append(new_prediction)
        
    predictions = pd.DataFrame(predictions)
    weather_predictions = pd.DataFrame(weather_predictions)
    merged_df = pd.merge(predictions, weather_predictions, on='timestamp', suffixes=('_no_weather', '_weather'))
    
    last_timestamp = merged_df.timestamp.iloc[0]
    first_timestamp = last_timestamp - pd.Timedelta(hours=72)
    timestamps = pd.date_range(start=first_timestamp, end=last_timestamp-pd.Timedelta(hours=1), freq='h')
    water_consumption_silver = water_consumption_silver[water_consumption_silver.timestamp.isin(timestamps)]
    water_consumption_silver = water_consumption_silver[["timestamp", "output_flow_rate_mean"]].rename(columns={"output_flow_rate_mean": "output_flow_rate_mean_no_weather"})
    water_consumption_silver['output_flow_rate_mean_weather'] = water_consumption_silver['output_flow_rate_mean_no_weather']
    water_consumption_silver['forecasted'] = False
    
    for _, row in merged_df.iterrows():
        timestamp = row["timestamp"]
        previous_timestamp = timestamp - pd.Timedelta(hours=1)
        last_output_flow_rate_mean = water_consumption_silver.loc[water_consumption_silver.timestamp == previous_timestamp, 'output_flow_rate_mean_no_weather'].values[0]
        this_hour_output_flow_rate_mean_no_weather = last_output_flow_rate_mean + row["change_from_last_hour_output_flow_rate_mean_no_weather"]
        this_hour_output_flow_rate_mean_weather = last_output_flow_rate_mean + row["change_from_last_hour_output_flow_rate_mean_weather"]
        new_row = {'timestamp': timestamp, 'output_flow_rate_mean_no_weather': this_hour_output_flow_rate_mean_no_weather, 'output_flow_rate_mean_weather': this_hour_output_flow_rate_mean_weather, 'forecasted': True}
        water_consumption_silver = pd.concat([water_consumption_silver, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    
    if not save_df:
        return water_consumption_silver
    
    water_consumption_silver.to_parquet(os.path.join(os.path.dirname(__file__),"../../data/gold/questions_4_and_7_answers.parquet"))

def main():
    forecast_next_24_hours_output_flow_rate(2023, 11, 13, 15)
    
if __name__ == "__main__":
    main()
