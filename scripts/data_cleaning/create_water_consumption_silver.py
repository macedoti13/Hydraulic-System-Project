import pandas as pd
import os

from create_water_consumption_silver_functions import fix_df, create_weather_dataset, create_combined_dataset, create_hourly_df

# paths
WATER_CONSUMPTION_BRONZE_PATH = os.path.join(os.path.dirname(__file__), "../../data/bronze/water_consumption.parquet")
WEATHER_2023_PATH = os.path.join(os.path.dirname(__file__), "../../data/bronze/weather_2023.parquet")
WEATHER_2024_PATH = os.path.join(os.path.dirname(__file__), "../../data/bronze/weather_2024.parquet")
WEATHER_2024_COMPLEMENTARY_PATH = os.path.join(os.path.dirname(__file__), "../../data/bronze/weather_2024_complementary.parquet")
WATER_CONSUMPTION_SILVER_SAVING_PATH = os.path.join(os.path.dirname(__file__), "../../data/silver/water_consumption_silver.parquet")

def main():
    
    # read the dataframes
    original_df = pd.read_parquet(WATER_CONSUMPTION_BRONZE_PATH)
    weather_2023 = pd.read_parquet(WEATHER_2023_PATH)
    weather_2024 = pd.read_parquet(WEATHER_2024_PATH)
    weather_2024_complemetary = pd.read_parquet(WEATHER_2024_COMPLEMENTARY_PATH)
    
    df = fix_df(original_df)
    weather_df = create_weather_dataset(weather_2023, weather_2024)
    combined_df = create_combined_dataset(df, weather_df, weather_2024_complemetary)
    hourly_df = create_hourly_df(combined_df)
    
    # save the dataframe
    hourly_df.to_parquet(WATER_CONSUMPTION_SILVER_SAVING_PATH)
    
if __name__ == "__main__":
    main()

