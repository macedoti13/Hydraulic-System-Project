import sys
from pathlib import Path

# Add the directory containing support_functions.py to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from utils.scripts.functions.support_functions import filter_df_for_forecasting_next_24_hours, create_features_vectorized, weather_features, seconds_to_hms
from xgboost import XGBRegressor
from datetime import timedelta
from typing import Optional
import pandas as pd
import random

def calculate_hourly_avg_flow_by_weekday_type(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    
    # Separate data into weekdays and weekends
    df_weekdays = df[df['weekday'] < 5]
    df_weekends = df[df['weekday'] >= 5]

    # Calculate average flow out per hour for weekdays and weekends
    flow_out_weekdays = df_weekdays.groupby('hour')['output_flow_rate'].mean().reset_index().rename(columns={'output_flow_rate': 'avg_weekday_output_flow'})
    flow_out_weekends = df_weekends.groupby('hour')['output_flow_rate'].mean().reset_index().rename(columns={'output_flow_rate': 'avg_weekend_output_flow'})
    
    # Merge the weekday and weekend data into a single DataFrame
    df_combined = pd.merge(flow_out_weekdays, flow_out_weekends, on='hour', how='outer')
    
    return df_combined

def calculate_daily_pump_usage_during_peak_and_offpeak_hours(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    peak_hours = (df["hour"] >= 18) & (df["hour"] <= 21)
    df["is_peak_hour"] = peak_hours
    
    df['pump_1_duration'] = df['pump_1'] * df['time_passed_seconds']
    df['pump_2_duration'] = df['pump_2'] * df['time_passed_seconds']
    
    daily_peak_usage = df[df['is_peak_hour']].groupby('date').agg({'pump_1_duration': 'sum', 'pump_2_duration': 'sum'})
    daily_off_peak_usage = df[~df['is_peak_hour']].groupby('date').agg({'pump_1_duration': 'sum', 'pump_2_duration': 'sum'})

    gmb_1_peak_avg = daily_peak_usage['pump_1_duration'].mean() / 60  
    gmb_1_off_peak_avg = daily_off_peak_usage['pump_1_duration'].mean() / 60  

    gmb_2_peak_avg = daily_peak_usage['pump_2_duration'].mean() / 60  
    gmb_2_off_peak_avg = daily_off_peak_usage['pump_2_duration'].mean() / 60  
    
    def convert_to_hours_and_minutes(minutes):
        if pd.isna(minutes):
            return "0 hours and 0 minutes"
        total_minutes = int(minutes)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours} hours and {minutes} minutes"
    
    data = {'pump': ['pump_1', 'pump_2'], 'average_time_used_peak_hours': [convert_to_hours_and_minutes(gmb_1_peak_avg), convert_to_hours_and_minutes(gmb_2_peak_avg)], 'average_time_used_offpeak_hours': [convert_to_hours_and_minutes(gmb_1_off_peak_avg), convert_to_hours_and_minutes(gmb_2_off_peak_avg)]}
    result_df = pd.DataFrame(data)
    
    return result_df

def forecast_next_24_hours_output_flow_rate(
        input_df: pd.DataFrame,  # receives water_consumption_curated.parquet dataset
        forecaster: XGBRegressor,
        input_flow_model: XGBRegressor,
        year: int, 
        month: int, 
        day: int, 
        hour: Optional[int] = None, 
        minute: Optional[int] = None,
        include_weather_features: bool = False
    ) -> pd.DataFrame:
    
    df = input_df.copy()
    
    # Filter dataset with the last 24 hours data from the specified date/time
    last_24_hours_data = filter_df_for_forecasting_next_24_hours(df, year, month, day, hour, minute)
    last_24_hours_data['forecasted'] = False
    original_last_24_hours_data = last_24_hours_data.copy()
    
    if len(last_24_hours_data) != 577:
        raise ValueError("The resulting DataFrame does not contain 576 rows.")
    
    first_row = last_24_hours_data.iloc[-1].to_dict()
    forecasted_values = [first_row]
    for i in range(576):
        # Create features for the new timestamp to be forecasteds
        features = pd.DataFrame(create_features_vectorized(last_24_hours_data).iloc[-1].drop(['id', 'timestamp', 'output_flow_rate'], errors='ignore'))
        
        # Remove weather-related features if include_weather_features is set to False
        if not include_weather_features:
            features = features.drop(weather_features, errors='ignore')
        
        # Forecast the output flow rate for the new timestamp using the forecaster model
        forecasted_output_flow_rate = forecaster.predict(features.values.reshape(1, -1))[0]
        if random.choice([True, False]):
            forecasted_output_flow_rate = forecasted_output_flow_rate * 1.05
        else:
            forecasted_output_flow_rate = forecasted_output_flow_rate * 0.95
        
        # Generate synthetic input_flow_rate for the new timestamp
        input_flow_features = last_24_hours_data['input_flow_rate'].tail(576).values.reshape(1, -1)
        next_input_flow = input_flow_model.predict(input_flow_features)[0]

        if random.choice([True, False]):
            next_input_flow = next_input_flow * 1.05
        else:
            next_input_flow = next_input_flow * 0.95
            
        if next_input_flow == 0:
            next_pump_1 = 0
            next_pump_2 = 0
        else:
            next_pump_1 = last_24_hours_data.iloc[-576]['pump_1']
            next_pump_2 = last_24_hours_data.iloc[-576]['pump_2']
            
        if next_pump_1 == 0 and next_pump_2 == 0:
            next_input_flow = 0.0
        
        # Derive other values based on the forecasted output and synthetic input
        next_timestamp = last_24_hours_data.iloc[-1]['timestamp'] + pd.Timedelta(seconds=150)
        next_reservoir_level_liters = last_24_hours_data.iloc[-1]['reservoir_level_liters'] + (next_input_flow * 150) - (forecasted_output_flow_rate * 150)
        next_reservoir_level_percentage = (next_reservoir_level_liters / 1000000) * 100
        next_total_liters_entered = next_input_flow * 150
        next_total_liters_out = forecasted_output_flow_rate * 150
        if next_total_liters_entered == 0:
            next_effective_liters_entered = 0.0
        elif next_total_liters_out > next_total_liters_entered:
            next_effective_liters_entered = 0.0
        else:
            next_effective_liters_entered = next_reservoir_level_liters - last_24_hours_data.iloc[-1]['reservoir_level_liters'] 
        next_pressure = last_24_hours_data['pressure'].iloc[-100:].mean() 
        
        if random.choice([True, False]):
            next_pressure = next_pressure * 1.05
        else:
            next_pressure = next_pressure * 0.95
        
        # Create the new row with all the values
        new_row = {
            'id': last_24_hours_data.iloc[-1]['id'] + 1,
            'timestamp': next_timestamp,
            'second': next_timestamp.second,
            'minute': next_timestamp.minute,
            'hour': next_timestamp.hour,
            'day': next_timestamp.day,
            'weekday': next_timestamp.weekday(),
            'week_of_year': next_timestamp.isocalendar().week,
            'month': next_timestamp.month,
            'year': next_timestamp.year,
            'time_passed_seconds': 150,
            'input_flow_rate': next_input_flow,
            'reservoir_level_percentage': next_reservoir_level_percentage,
            'reservoir_level_liters': next_reservoir_level_liters,
            'total_liters_entered': next_total_liters_entered,
            'effective_liters_entered': next_effective_liters_entered,
            'total_liters_out': next_total_liters_out,
            'output_flow_rate': forecasted_output_flow_rate,
            'pressure': next_pressure,
            'pump_1': next_pump_1,
            'pump_2': next_pump_2,
            'total_precip_mm': last_24_hours_data.iloc[-1]['total_precip_mm'],
            'station_pressure_mb': last_24_hours_data.iloc[-1]['station_pressure_mb'],
            'max_pressure_last_hour_mb': last_24_hours_data.iloc[-1]['max_pressure_last_hour_mb'],
            'min_pressure_last_hour_mb': last_24_hours_data.iloc[-1]['min_pressure_last_hour_mb'],
            'global_radiation_kj_m2': last_24_hours_data.iloc[-1]['global_radiation_kj_m2'],
            'air_temp_c': last_24_hours_data.iloc[-1]['air_temp_c'],
            'dew_point_temp_c': last_24_hours_data.iloc[-1]['dew_point_temp_c'],
            'max_temp_last_hour_c': last_24_hours_data.iloc[-1]['max_temp_last_hour_c'],
            'min_temp_last_hour_c': last_24_hours_data.iloc[-1]['min_temp_last_hour_c'],
            'max_dew_point_last_hour_c': last_24_hours_data.iloc[-1]['max_dew_point_last_hour_c'],
            'min_dew_point_last_hour_c': last_24_hours_data.iloc[-1]['min_dew_point_last_hour_c'],
            'max_humidity_last_hour_percentage': last_24_hours_data.iloc[-1]['max_humidity_last_hour_percentage'],
            'min_humidity_last_hour_percentage': last_24_hours_data.iloc[-1]['min_humidity_last_hour_percentage'],
            'relative_humidity_percentage': last_24_hours_data.iloc[-1]['relative_humidity_percentage'],
            'wind_direction_deg': last_24_hours_data.iloc[-1]['wind_direction_deg'],
            'max_wind_gust_m_s': last_24_hours_data.iloc[-1]['max_wind_gust_m_s'],
            'wind_speed_m_s': last_24_hours_data.iloc[-1]['wind_speed_m_s'],
            'forecasted': True
        }
        new_row_df = pd.DataFrame(new_row, index=[0])
        
        # Append the new row to the last_24_hours_data dataset
        last_24_hours_data = pd.concat([last_24_hours_data, new_row_df], ignore_index=True)
        original_last_24_hours_data = pd.concat([original_last_24_hours_data, new_row_df], ignore_index=True)
        
        # Keep only the last 576 rows
        last_24_hours_data = last_24_hours_data.iloc[1:]
        forecasted_values.append(new_row)
    
    # Create a DataFrame with forecasted values for the next 24 hours
    forecasted_df = pd.DataFrame(forecasted_values).round(2)
    
    return forecasted_df, original_last_24_hours_data[['hour', 'output_flow_rate', 'forecasted']].round(2)

def simulate_emptying(
    input_df: pd.DataFrame,  # receives water_consumption_curated.parquet dataset
    forecaster: XGBRegressor,
    input_flow_model: XGBRegressor,
    year: int, 
    month: int, 
    day: int, 
    hour: Optional[int] = None, 
    minute: Optional[int] = None,
    include_weather_features: bool = False,
) -> pd.DataFrame:
    
    df = input_df.copy()
    next_24_hours_forecasting, _ = forecast_next_24_hours_output_flow_rate(df, forecaster, input_flow_model, year, month, day, hour, minute, include_weather_features)
    reservoir_level = next_24_hours_forecasting.reservoir_level_liters.values[0]
    row = next_24_hours_forecasting.iloc[0, :]
    print(f"Initial reservoir level: {reservoir_level}")
    time_elapsed = 0
    simulations = []

    while reservoir_level > 0:
        simulation_df = pd.DataFrame([row] * 576).reset_index(drop=True)
        simulation_df['timestamp'] = [row['timestamp'] + timedelta(seconds=i * 150) for i in range(576)]
        simulation_df['output_flow_rate'] = next_24_hours_forecasting['output_flow_rate'].values[1:]
        for i, sim_row in simulation_df.iterrows():
            if i == 0:
                continue
            liters_out = sim_row['output_flow_rate'] * 150
            reservoir_level -= liters_out
            simulation_df.at[i, 'reservoir_level_liters'] = reservoir_level
            time_elapsed += 150
            
            if simulation_df.iloc[i, :]['reservoir_level_liters'] <= 0:
                simulation_df = simulation_df[simulation_df.index <= i]
                simulations.append(simulation_df)
                if len(simulations) > 0:
                    simulation_df = pd.concat(simulations, ignore_index=True)
                return simulation_df, seconds_to_hms(time_elapsed)
        
        row = simulation_df.iloc[-1, :]
        old_simulation_df = simulation_df.copy()
        simulations.append(old_simulation_df)
        
    return simulation_df, seconds_to_hms(time_elapsed)