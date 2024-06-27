import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from answer_questions_4_and_7 import forecast_next_24_hours_output_flow_rate
import pandas as pd

pd.set_option('display.max_rows', None)

def simulate_emptying_reservoir(year, month, day, hour, save_df=True):
    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    original_df = pd.read_parquet(os.path.join(os.path.dirname(__file__),"../../data/silver/water_consumption_silver.parquet"))
    original_df = original_df[['timestamp', 'output_flow_rate_mean', 'reservoir_level_percentage_last']]
    original_df = original_df[original_df.timestamp <= timestamp].tail(72)    
    original_df = original_df.rename(columns={'output_flow_rate_mean': 'output_flow_rate_mean_no_weather', 'reservoir_level_percentage_last': 'reservoir_level_percentage_last_no_weather'})
    original_df['output_flow_rate_mean_weather'] = original_df['output_flow_rate_mean_no_weather']
    original_df['reservoir_level_percentage_last_weather'] = original_df['reservoir_level_percentage_last_no_weather']
    original_df['forecasted'] = False
    time_elapsed = 0
    
    while True:
        forecasted_df = forecast_next_24_hours_output_flow_rate(year, month, day, hour, save_df=False)
        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        
        for _ in range(24):
            next_timestamp = timestamp + pd.Timedelta(hours=1)
            next_hour_output_flow_rate_mean_no_weather = forecasted_df[forecasted_df.timestamp==next_timestamp].output_flow_rate_mean_no_weather.values[0]
            next_hour_output_flow_rate_mean_weather = forecasted_df[forecasted_df.timestamp==next_timestamp].output_flow_rate_mean_weather.values[0]
            liters_out_next_hour_weather = next_hour_output_flow_rate_mean_weather * 3600
            liters_out_next_hour_no_weather = next_hour_output_flow_rate_mean_no_weather * 3600
            last_reservoir_level_percentage_no_weather = original_df[original_df.timestamp == timestamp].reservoir_level_percentage_last_no_weather.values[0]
            last_reservoir_level_percentage_weather = original_df[original_df.timestamp == timestamp].reservoir_level_percentage_last_weather.values[0]
            next_reservoir_level_percentage_no_weather = last_reservoir_level_percentage_no_weather-((liters_out_next_hour_no_weather/1_000_000)*100)
            next_reservoir_level_percentage_weather = last_reservoir_level_percentage_weather-((liters_out_next_hour_weather/1_000_000)*100)
            new_row = {
                'timestamp': next_timestamp, 
                'output_flow_rate_mean_no_weather': next_hour_output_flow_rate_mean_no_weather, 
                'output_flow_rate_mean_weather': next_hour_output_flow_rate_mean_weather, 
                'reservoir_level_percentage_last_no_weather': next_reservoir_level_percentage_no_weather,
                'reservoir_level_percentage_last_weather': next_reservoir_level_percentage_weather,
                'forecasted': True
            }
            new_row = pd.DataFrame([new_row])
            original_df = pd.concat([original_df, new_row], ignore_index=True)
            timestamp = next_timestamp
            reservoir_level_percentage_no_weather = original_df.reservoir_level_percentage_last_no_weather.tail(1).values[0]
            reservoir_level_percentage_weather = original_df.reservoir_level_percentage_last_weather.tail(1).values[0]
            if reservoir_level_percentage_no_weather > 0 or reservoir_level_percentage_weather > 0:
                time_elapsed += 1
            
        if reservoir_level_percentage_no_weather < 0 or reservoir_level_percentage_weather < 0:
            break 
        
        year = original_df.iloc[-1].timestamp.year
        month = original_df.iloc[-1].timestamp.month
        day = original_df.iloc[-1].timestamp.day
        hour = original_df.iloc[-1].timestamp.hour
        
    first_empty_reservoir_index = original_df[original_df.reservoir_level_percentage_last_no_weather < 0].index[0]
    original_df = original_df.iloc[:first_empty_reservoir_index+1].tail(24).reset_index(drop=True)
    
    if not save_df:
        return original_df, time_elapsed
    
    original_df.to_parquet(os.path.join(os.path.dirname(__file__),"../../data/gold/question_5_answer.parquet"))

def main():
    simulate_emptying_reservoir(2023, 11, 13, 13, save_df=True)
    
if __name__ == "__main__":
    main()
