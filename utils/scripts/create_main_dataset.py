from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "data"))

BASEDFPATH = project_root / Path('data/original_datasets/water_consumption.parquet')
DATAPATH2023 = project_root / Path('data/original_datasets/weather_2023.parquet')
DATAPATH2024 = project_root / Path('data/original_datasets/weather_2024.parquet')
DATAPATH2024_COMP = project_root / Path('data/original_datasets/weather_2024_complementary.parquet')
COLUMN_MAPPING = {
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'total_precip_mm',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'station_pressure_mb',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'max_pressure_last_hour_mb',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'min_pressure_last_hour_mb',
    'RADIACAO GLOBAL (Kj/m²)': 'global_radiation_kj_m2',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'air_temp_c',
    'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'dew_point_temp_c',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'max_temp_last_hour_c',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'min_temp_last_hour_c',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'max_dew_point_last_hour_c',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'min_dew_point_last_hour_c',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'max_humidity_last_hour_percentage',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'min_humidity_last_hour_percentage',
    'UMIDADE RELATIVA DO AR, HORARIA (%)': 'relative_humidity_percentage',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'wind_direction_deg',
    'VENTO, RAJADA MAXIMA (m/s)': 'max_wind_gust_m_s',
    'VENTO, VELOCIDADE HORARIA (m/s)': 'wind_speed_m_s'
}

def impute_faulty_data(df):
    """
    Impute reservoir_level_percentage and input_flow_rate for rows with reservoir_level_percentage set to 0.0.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with imputed reservoir_level_percentage and input_flow_rate values.
    """
    df = df.copy()
    faulty_rows = df[df['reservoir_level_percentage'] == 0.0]

    for index, row in faulty_rows.iterrows():
        t_minus_1 = df.loc[:index-1, 'reservoir_level_percentage'].replace(0.0, pd.NA).last_valid_index()
        t_plus_1 = df.loc[index+1:, 'reservoir_level_percentage'].replace(0.0, pd.NA).first_valid_index()
        
        if pd.notna(t_minus_1) and pd.notna(t_plus_1):
            prev_level = df.at[t_minus_1, 'reservoir_level_percentage']
            next_level = df.at[t_plus_1, 'reservoir_level_percentage']
            if prev_level < next_level:
                imputed_level = round((prev_level + next_level) / 2, 2)
            else:
                imputed_level = round((prev_level + next_level) / 2, 2)

            df.at[index, 'reservoir_level_percentage'] = imputed_level
            
            prev_flow = df.at[t_minus_1, 'input_flow_rate']
            next_flow = df.at[t_plus_1, 'input_flow_rate']
            imputed_flow = round((prev_flow + next_flow) / 2, 2)
            df.at[index, 'input_flow_rate'] = imputed_flow

    return df

def fill_missing_pressure(df):
    """
    Fill missing pressure values using a moving average of the last 10 steps.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with filled pressure values.
    """
    df = df.copy()
    df['pressure'] = df['pressure'].replace(0.0, np.nan)
    df['pressure'] = df['pressure'].fillna(df['pressure'].rolling(window=10, min_periods=1).mean())
    df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
    df['pressure'] = df['pressure'].round(2)
    
    return df

def fix_pump_status(df):
    """
    Fix invalid pump status by turning on pump_2 where input_flow_rate is greater than 0 but both pumps are off.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with corrected pump statuses.
    """
    df = df.copy()
    invalid_pump_rows = df[(df['pump_1'] == 0) & (df['pump_2'] == 0) & (df['input_flow_rate'] > 0)]

    for index, row in invalid_pump_rows.iterrows():
        df.at[index, 'pump_2'] = 1

    return df

def turn_off_pumps(df):
    """
    Turn off both pumps where input_flow_rate is 0 but at least one pump is on.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with pumps turned off where necessary.
    """
    df = df.copy()
    invalid_rows = df[(df['input_flow_rate'] == 0) & ((df['pump_1'] == 1) | (df['pump_2'] == 1))]

    for index, row in invalid_rows.iterrows():
        df.at[index, 'pump_1'] = 0
        df.at[index, 'pump_2'] = 0

    return df

def create_necessary_columns(original_df):
    """
    Create additional necessary columns for analysis.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with additional columns.
    """
    df = original_df.copy()
    df['reservoir_level_liters'] = df['reservoir_level_percentage'] * 1_000_000 / 100
    df['time_passed_seconds'] = df['timestamp'].diff().dt.total_seconds()
    df['total_liters_entered'] = df['input_flow_rate'] * df['time_passed_seconds']
    df['effective_liters_entered'] = df['reservoir_level_liters'].diff()
    df['total_liters_out'] = df['total_liters_entered'] - df['effective_liters_entered']
    df['output_flow_rate'] = df['total_liters_out'] / df['time_passed_seconds']
    df.loc[df['effective_liters_entered'] < 0, 'effective_liters_entered'] = 0
    df = df.drop(0)
    
    return df


def adjust_flow_rates(df):
    """
    Adjust input_flow_rate so that total_liters_entered is not less than effective_liters_entered.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with calculated columns.

    Returns
    -------
    pd.DataFrame
        The DataFrame with adjusted input_flow_rate.
    """
    df = df.copy()
    problem_rows = df[df['total_liters_entered'] < df['effective_liters_entered']]
    
    for index, row in problem_rows.iterrows():
        df.at[index, 'input_flow_rate'] = df.at[index, 'effective_liters_entered'] / df.at[index, 'time_passed_seconds']
        df.at[index, 'total_liters_entered'] = df.at[index, 'effective_liters_entered']
        df.at[index, 'total_liters_out'] = 0
        df.at[index, 'output_flow_rate'] = 0
    
    return df

def fix_problematic_rows(df):
    """
    Apply all data cleaning functions to fix various issues in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    df = impute_faulty_data(df)
    df = fill_missing_pressure(df)
    df = fix_pump_status(df)
    df = turn_off_pumps(df)
    df = create_necessary_columns(df)
    df = adjust_flow_rates(df)
    df = df.round(2).reset_index().rename(columns={'index': 'id'})

    return df

def create_date_columns(original_df):
    """
    Create additional date-related columns from the 'timestamp' column.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original DataFrame with the raw data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with additional columns for second, minute, hour, day, weekday,
        week_of_year, month, and year.
    """
    df = original_df.copy()
    df['second'] = df['timestamp'].dt.second
    df['minute'] = df['timestamp'].dt.minute
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    
    return df


def create_weather_dataset():
    """
    Create a combined weather dataset from the 2023 and 2024 weather data files.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame containing weather data from both years.
    """
    df23, df24 = preprocess_weather_dataset(DATAPATH2023), preprocess_weather_dataset(DATAPATH2024)
    df_23_24_concat = pd.concat([df23, df24], axis=0)
    return df_23_24_concat


def preprocess_weather_dataset(path):
    """
    Preprocess a weather dataset from the given file path.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the weather data file.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame with renamed and formatted columns.
    """
    #df = pd.read_csv(path, delimiter=';', skiprows=8, encoding='latin1', decimal=',')
    df = pd.read_parquet(path)
    df.drop(df.columns[-1], axis=1, inplace=True)  # Last column is empty due to ";" at the end of each line
    df['Hora UTC'] = df['Hora UTC'].apply(lambda x: datetime.strptime(x, '%H%M %Z')) 
    df['Data'] = pd.to_datetime(df['Data'], format='%Y/%m/%d')
    df['hour'] = df['Hora UTC'].dt.hour
    df['day'] = df['Data'].dt.day
    df['month'] = df['Data'].dt.month
    df['year'] = df['Data'].dt.year
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    df.drop(columns=['Data', 'Hora UTC'], axis=1, inplace=True)

    return df


def fill_na_with_moving_average(original_df, window=10):
    """
    Fill missing values in a DataFrame using a moving average.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original DataFrame with the raw data.
    window : int, optional
        The window size for calculating the moving average (default is 10).

    Returns
    -------
    pd.DataFrame
        The DataFrame with missing values filled using a moving average.
    """
    df = original_df.copy()
    for column in df.columns:
        if df[column].isna().sum() > 0:
            df[column] = df[column].transform(lambda x: x.fillna(x.rolling(window, min_periods=1).mean()))
            # Forward fill any remaining NaNs
            df[column] = df[column].ffill()
            # Backward fill any remaining NaNs
            df[column] = df[column].bfill()
    return df


def create_curated_df(original_df, original_weather_df):
    """
    Create a curated DataFrame by merging the original data with weather data,
    filling missing values, and sorting the data.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original DataFrame with the raw data.
    original_weather_df : pd.DataFrame
        The weather DataFrame with the processed weather data.

    Returns
    -------
    pd.DataFrame
        The curated and merged DataFrame with filled missing values and sorted data.
    """
    df = original_df.copy()
    weather_df = original_weather_df.copy()
    
    merged_df = df.merge(weather_df, on=['hour', 'day', 'year', 'month'], how='left')
    comp_data = preprocess_weather_dataset(DATAPATH2024_COMP).query('1 <= day <= 11 and month == 3')
    merged_df = merged_df.sort_values(by=['day', 'month', 'hour', 'year'])
    comp_data = comp_data.sort_values(by=['day', 'month', 'hour', 'year'])

    # Set the index on the sorted DataFrames
    merged_df_indexed = merged_df.set_index(['day', 'month', 'hour', 'year'])
    comp_data_indexed = comp_data.set_index(['day', 'month', 'hour', 'year'])

    # Perform the update operation
    merged_df_indexed.update(comp_data_indexed)

    # Reset the index to return to the original structure
    full_df = merged_df_indexed.reset_index()
    filled_df = fill_na_with_moving_average(full_df)
    filled_df.sort_values(by='timestamp', inplace=True)
    filled_df = filled_df[[
        "id",
        "timestamp",
        "second",
        "minute",
        "hour",
        "day",
        "weekday",
        "week_of_year", 
        "month",
        "year",
        "time_passed_seconds",
        "input_flow_rate",
        "reservoir_level_percentage",
        "reservoir_level_liters",
        "total_liters_entered",
        "effective_liters_entered", 
        "total_liters_out",
        "output_flow_rate",
        "pressure",
        "pump_1",
        "pump_2",
        "total_precip_mm",
        "station_pressure_mb",
        "max_pressure_last_hour_mb",
        "min_pressure_last_hour_mb",
        "global_radiation_kj_m2",
        "air_temp_c",
        "dew_point_temp_c",
        "max_temp_last_hour_c",
        "min_temp_last_hour_c",
        "max_dew_point_last_hour_c",
        "min_dew_point_last_hour_c",
        "max_humidity_last_hour_percentage",
        "min_humidity_last_hour_percentage",
        "relative_humidity_percentage",
        "wind_direction_deg",
        "max_wind_gust_m_s",
        "wind_speed_m_s"
    ]].reset_index(drop=True)
    
    return filled_df

def set_up_base_df(base_df):
    original_df = base_df.copy()
    
    # Rename columns and replace pump status values
    original_df = original_df.rename(columns={
        "DATA/HORA": "timestamp",
        "VAZÃO ENTRADA (L/S)": "input_flow_rate",
        "NÍVEL RESERVATÓRIO (%)": "reservoir_level_percentage",
        "PRESSÃO (mca)": "pressure",
        "GMB 1 (10 OFF/ 90 ON)": "pump_1",
        "GMB 2(10 OFF/ 90 ON)": "pump_2",
    }).replace({
        "pump_1": {10: 0, 90: 1},
        "pump_2": {10: 0, 90: 1}
    })
    
    return original_df

def main():
    base_df = pd.read_parquet(BASEDFPATH)
    original_df = set_up_base_df(base_df)
    df = fix_problematic_rows(original_df)
    df = create_date_columns(df)
    weather_df = create_weather_dataset()
    filled_df = create_curated_df(df, weather_df)
    filled_df.to_parquet(project_root / Path('data/curated_datasets/water_consumption_curated_2.parquet'))

if __name__ == "__main__":
    main()
