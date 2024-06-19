from typing import Optional
import pandas as pd

def filter_df_for_forecasting_next_24_hours(
        input_df: pd.DataFrame, 
        year: int, 
        month: int, 
        day: int, 
        hour: Optional[int] = None, 
        minute: Optional[int] = None, 
    ) -> pd.DataFrame:
    
    df = input_df.copy()
    filtered_df = df[(df.year == year) & (df.month == month) & (df.day == day)]
    
    if hour is not None:
        new_filtered_df = filtered_df[filtered_df.hour == hour]
        if not new_filtered_df.empty:
            filtered_df = new_filtered_df
            
    if minute is not None:
        new_filtered_df = filtered_df[filtered_df.minute == minute]
        if not new_filtered_df.empty:
            filtered_df = new_filtered_df
    
    if filtered_df.empty:
        raise ValueError("No matching date/time found in the DataFrame.")
    
    index = filtered_df.index[0]
    
    if index < 575:
        raise ValueError("Not enough rows before the specified date/time to extract 576 rows.")
    
    start_index = index - 576
    useful_df = df.iloc[start_index:index+1]  
    
    if len(useful_df) != 577:
        raise ValueError("The resulting DataFrame does not contain 576 rows.")
    
    return useful_df  

def create_features_vectorized(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    window_24h = 576
    window_10h = int(10 * 60 * 60 / 150)
    window_1h = int(1 * 60 * 60 / 150)
    window_10m = int(10 * 60 / 150)
    windows = {"24_hours": window_24h, "10_hours": window_10h, "1_hour": window_1h, "10_minutes": window_10m}

    # Initialize a list for collecting DataFrames
    feature_dfs = []

    # Rolling calculations for each column and window
    for window_name, window_size in windows.items():
        features = {f"average_input_flow_rate_{window_name}": df["input_flow_rate"].rolling(window=window_size, min_periods=1).mean(), f"average_change_reservoir_level_percentage_{window_name}": df["reservoir_level_percentage"].rolling(window=window_size, min_periods=1).mean(), f"average_total_liters_entered_{window_name}": df["total_liters_entered"].rolling(window=window_size, min_periods=1).mean(), f"sum_total_liters_entered_last_{window_name}": df["total_liters_entered"].rolling(window=window_size, min_periods=1).sum(), f"average_effective_liters_entered_{window_name}": df["effective_liters_entered"].rolling(window=window_size, min_periods=1).mean(), f"sum_effective_liters_entered_last_{window_name}": df["effective_liters_entered"].rolling(window=window_size, min_periods=1).sum(), f"average_total_liters_out_last_{window_name}": df["total_liters_out"].rolling(window=window_size, min_periods=1).mean(), f"sum_total_liters_out_last_{window_name}": df["total_liters_out"].rolling(window=window_size, min_periods=1).sum(), f"average_output_flow_rate_last_{window_name}": df["output_flow_rate"].rolling(window=window_size, min_periods=1).mean(), f"average_pressure_last_{window_name}": df["pressure"].rolling(window=window_size, min_periods=1).mean()}
        weather_columns = ["total_precip_mm", "station_pressure_mb", "max_pressure_last_hour_mb", "min_pressure_last_hour_mb", "global_radiation_kj_m2", "air_temp_c", "dew_point_temp_c", "max_temp_last_hour_c", "min_temp_last_hour_c", "max_dew_point_last_hour_c", "min_dew_point_last_hour_c", "max_humidity_last_hour_percentage", "min_humidity_last_hour_percentage", "relative_humidity_percentage", "wind_direction_deg", "max_wind_gust_m_s", "wind_speed_m_s"]

        for col in weather_columns:
            features[f"average_{col}_last_{window_name}"] = df[col].rolling(window=window_size, min_periods=1).mean()

        feature_dfs.append(pd.DataFrame(features))

    # Last value calculations
    last_values = {"last_input_flow_rate": df["input_flow_rate"], "last_reservoir_level_percentage": df["reservoir_level_percentage"], "last_total_liters_entered": df["total_liters_entered"], "last_effective_liters_entered": df["effective_liters_entered"], "last_total_liters_out": df["total_liters_out"], "last_output_flow_rate": df["output_flow_rate"], "last_pressure": df["pressure"]}

    for col in weather_columns:
        last_values[f"last_{col}"] = df[col]

    last_values_df = pd.DataFrame(last_values)

    # Pump on time calculations
    pump_on_time = {"total_time_pump_1_was_on_last_24_hours": df["pump_1"].rolling(window=window_24h, min_periods=1).sum() * 150, "total_time_pump_2_was_on_last_24_hours": df["pump_2"].rolling(window=window_24h, min_periods=1).sum() * 150, "last_pump_1_status": df["pump_1"], "last_pump_2_status": df["pump_2"]}
    pump_on_time_df = pd.DataFrame(pump_on_time)

    # Date-related columns
    date_related = {"id": df["id"], "timestamp": df["timestamp"], "second": df["timestamp"].dt.second, "minute": df["timestamp"].dt.minute, "hour": df["timestamp"].dt.hour, "day": df["timestamp"].dt.day, "weekday": df["timestamp"].dt.weekday, "week_of_year": df["timestamp"].dt.isocalendar().week, "month": df["timestamp"].dt.month, "year": df["timestamp"].dt.year}
    date_related_df = pd.DataFrame(date_related)

    # Add target variable
    target = df["output_flow_rate"].rename("output_flow_rate")
    feature_data = pd.concat([pd.concat(feature_dfs, axis=1), last_values_df, pump_on_time_df, date_related_df, target], axis=1)
    feature_data = feature_data.iloc[576:].reset_index(drop=True)

    return feature_data

weather_features = [
        'average_total_precip_mm_last_24_hours', 'average_station_pressure_mb_last_24_hours',
        'average_max_pressure_last_hour_mb_last_24_hours', 'average_min_pressure_last_hour_mb_last_24_hours',
        'average_global_radiation_kj_m2_last_24_hours', 'average_air_temp_c_last_24_hours',
        'average_dew_point_temp_c_last_24_hours', 'average_max_temp_last_hour_c_last_24_hours',
        'average_min_temp_last_hour_c_last_24_hours', 'average_max_dew_point_last_hour_c_last_24_hours',
        'average_min_dew_point_last_hour_c_last_24_hours', 'average_max_humidity_last_hour_percentage_last_24_hours',
        'average_min_humidity_last_hour_percentage_last_24_hours', 'average_relative_humidity_percentage_last_24_hours',
        'average_wind_direction_deg_last_24_hours', 'average_max_wind_gust_m_s_last_24_hours',
        'average_wind_speed_m_s_last_24_hours', 'average_total_precip_mm_last_10_hours',
        'average_station_pressure_mb_last_10_hours', 'average_max_pressure_last_hour_mb_last_10_hours',
        'average_min_pressure_last_hour_mb_last_10_hours', 'average_global_radiation_kj_m2_last_10_hours',
        'average_air_temp_c_last_10_hours', 'average_dew_point_temp_c_last_10_hours',
        'average_max_temp_last_hour_c_last_10_hours', 'average_min_temp_last_hour_c_last_10_hours',
        'average_max_dew_point_last_hour_c_last_10_hours', 'average_min_dew_point_last_hour_c_last_10_hours',
        'average_max_humidity_last_hour_percentage_last_10_hours', 'average_min_humidity_last_hour_percentage_last_10_hours',
        'average_relative_humidity_percentage_last_10_hours', 'average_wind_direction_deg_last_10_hours',
        'average_max_wind_gust_m_s_last_10_hours', 'average_wind_speed_m_s_last_10_hours',
        'average_total_precip_mm_last_1_hour', 'average_station_pressure_mb_last_1_hour',
        'average_max_pressure_last_hour_mb_last_1_hour', 'average_min_pressure_last_hour_mb_last_1_hour',
        'average_global_radiation_kj_m2_last_1_hour', 'average_air_temp_c_last_1_hour',
        'average_dew_point_temp_c_last_1_hour', 'average_max_temp_last_hour_c_last_1_hour',
        'average_min_temp_last_hour_c_last_1_hour', 'average_max_dew_point_last_hour_c_last_1_hour',
        'average_min_dew_point_last_hour_c_last_1_hour', 'average_max_humidity_last_hour_percentage_last_1_hour',
        'average_min_humidity_last_hour_percentage_last_1_hour', 'average_relative_humidity_percentage_last_1_hour',
        'average_wind_direction_deg_last_1_hour', 'average_max_wind_gust_m_s_last_1_hour',
        'average_wind_speed_m_s_last_1_hour', 'average_total_precip_mm_last_10_minutes',
        'average_station_pressure_mb_last_10_minutes', 'average_max_pressure_last_hour_mb_last_10_minutes',
        'average_min_pressure_last_hour_mb_last_10_minutes', 'average_global_radiation_kj_m2_last_10_minutes',
        'average_air_temp_c_last_10_minutes', 'average_dew_point_temp_c_last_10_minutes',
        'average_max_temp_last_hour_c_last_10_minutes', 'average_min_temp_last_hour_c_last_10_minutes',
        'average_max_dew_point_last_hour_c_last_10_minutes', 'average_min_dew_point_last_hour_c_last_10_minutes',
        'average_max_humidity_last_hour_percentage_last_10_minutes', 'average_min_humidity_last_hour_percentage_last_10_minutes',
        'average_relative_humidity_percentage_last_10_minutes', 'average_wind_direction_deg_last_10_minutes',
        'average_max_wind_gust_m_s_last_10_minutes', 'average_wind_speed_m_s_last_10_minutes',
        'last_total_precip_mm', 'last_station_pressure_mb', 'last_max_pressure_last_hour_mb', 
        'last_min_pressure_last_hour_mb', 'last_global_radiation_kj_m2', 'last_air_temp_c',
        'last_dew_point_temp_c', 'last_max_temp_last_hour_c', 'last_min_temp_last_hour_c',
        'last_max_dew_point_last_hour_c', 'last_min_dew_point_last_hour_c', 'last_max_humidity_last_hour_percentage',
        'last_min_humidity_last_hour_percentage', 'last_relative_humidity_percentage', 'last_wind_direction_deg',
        'last_max_wind_gust_m_s', 'last_wind_speed_m_s'
    ]

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds