from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "data"))

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

def main():
    df = pd.read_parquet(project_root / "data" / "curated_datasets" / "water_consumption_curated.parquet")
    df2 = create_features_vectorized(df)
    df2.to_parquet(project_root / "data" / "curated_datasets" / "forecasting_dataset.parquet")
    
if __name__ == "__main__":
    main()