import pandas as pd

def create_lag_features(input_df):
    df = input_df.copy()

    all_features = []
    non_weather_features = []

    lag_features = {}

    for lag in [1, 2, 3, 6, 12, 24, 26, 48, 72]:
        
        lag_features[f'{lag}_hours_ago_input_flow_rate_first'] = df['input_flow_rate_first'].shift(lag)
        lag_features[f'{lag}_hours_agot_input_flow_rate_last'] = df['input_flow_rate_last'].shift(lag)
        lag_features[f'{lag}_hours_ago_input_flow_rate_mean'] = df['input_flow_rate_mean'].shift(lag)
        lag_features[f'{lag}_hours_ago_reservoir_level_percentage_first'] = df['reservoir_level_percentage_first'].shift(lag)
        lag_features[f'{lag}_hours_ago_reservoir_level_percentage_last'] = df['reservoir_level_percentage_last'].shift(lag)
        lag_features[f'{lag}_hours_ago_reservoir_level_percentage_mean'] = df['reservoir_level_percentage_mean'].shift(lag)
        lag_features[f'{lag}_hours_ago_output_flow_rate_first'] = df['output_flow_rate_first'].shift(lag)
        lag_features[f'{lag}_hours_ago_output_flow_rate_last'] = df['output_flow_rate_last'].shift(lag)
        lag_features[f'{lag}_hours_ago_output_flow_rate_mean'] = df['output_flow_rate_mean'].shift(lag)
        lag_features[f'{lag}_hours_ago_pressure_first'] = df['pressure_first'].shift(lag)
        lag_features[f'{lag}_hours_ago_pressure_last'] = df['pressure_last'].shift(lag)
        lag_features[f'{lag}_hours_ago_pressure_mean'] = df['pressure_mean'].shift(lag)
        lag_features[f'{lag}_hours_agot_pump_1_duration_sum'] = df['pump_1_duration_sum'].shift(lag)
        lag_features[f'{lag}_hours_agot_pump_2_duration_sum'] = df['pump_2_duration_sum'].shift(lag)
        lag_features[f'{lag}_hours_ago_temperature'] = df['air_temp_c'].shift(lag)
        lag_features[f'{lag}_hours_ago_precipitation'] = df['total_precip_mm'].shift(lag)
        lag_features[f'{lag}_hours_ago_humidity'] = df['relative_humidity_percentage'].shift(lag)

        all_features.extend([
            f'{lag}_hours_ago_input_flow_rate_first', f'{lag}_hours_agot_input_flow_rate_last', f'{lag}_hours_ago_input_flow_rate_mean',
            f'{lag}_hours_ago_reservoir_level_percentage_first', f'{lag}_hours_ago_reservoir_level_percentage_last', f'{lag}_hours_ago_reservoir_level_percentage_mean',
            f'{lag}_hours_ago_output_flow_rate_first', f'{lag}_hours_ago_output_flow_rate_last', f'{lag}_hours_ago_output_flow_rate_mean',
            f'{lag}_hours_ago_pressure_first', f'{lag}_hours_ago_pressure_last', f'{lag}_hours_ago_pressure_mean',
            f'{lag}_hours_agot_pump_1_duration_sum', f'{lag}_hours_agot_pump_2_duration_sum',
            f'{lag}_hours_ago_temperature', f'{lag}_hours_ago_precipitation', f'{lag}_hours_ago_humidity'
        ])

        non_weather_features.extend([
            f'{lag}_hours_ago_input_flow_rate_first', f'{lag}_hours_agot_input_flow_rate_last', f'{lag}_hours_ago_input_flow_rate_mean',
            f'{lag}_hours_ago_reservoir_level_percentage_first', f'{lag}_hours_ago_reservoir_level_percentage_last', f'{lag}_hours_ago_reservoir_level_percentage_mean',
            f'{lag}_hours_ago_output_flow_rate_first', f'{lag}_hours_ago_output_flow_rate_last', f'{lag}_hours_ago_output_flow_rate_mean',
            f'{lag}_hours_ago_pressure_first', f'{lag}_hours_ago_pressure_last', f'{lag}_hours_ago_pressure_mean',
            f'{lag}_hours_agot_pump_1_duration_sum', f'{lag}_hours_agot_pump_2_duration_sum'
        ])

    # Use pd.concat to concatenate the new lag features into the dataframe
    df = pd.concat([df, pd.DataFrame(lag_features)], axis=1)

    return df, all_features, non_weather_features

def create_window_features(input_df):
    df = input_df.copy()
    
    all_features = []
    non_weather_features = []

    window_features = {}

    for window in [2, 3, 6, 12, 24, 36, 48, 72]:
        window_features[f'{window}_hours_rolling_input_flow_rate_diff'] = df['input_flow_rate_diff'].rolling(window).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else pd.NA)
        window_features[f'{window}_hours_rolling_output_flow_rate_diff'] = df['output_flow_rate_diff'].rolling(window).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else pd.NA)
        window_features[f'{window}_hours_rolling_reservoir_level_change'] = df['reservoir_level_change'].rolling(window).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else pd.NA)
        window_features[f'{window}_hours_rolling_pressure_change'] = df['pressure_change'].rolling(window).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else pd.NA)
        window_features[f'{window}_hours_rolling_change_from_last_hour_output_flow_rate_mean'] = df['change_from_last_hour_output_flow_rate_mean'].rolling(window).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == window else pd.NA)
        window_features[f'{window}_hours_rolling_temperature'] = df['air_temp_c'].rolling(window).mean()
        window_features[f'{window}_hours_rolling_precipitation'] = df['total_precip_mm'].rolling(window).mean()
        window_features[f'{window}_hours_rolling_humidity'] = df['relative_humidity_percentage'].rolling(window).mean()
        
        all_features.extend([
            f'{window}_hours_rolling_input_flow_rate_diff', f'{window}_hours_rolling_output_flow_rate_diff', f'{window}_hours_rolling_reservoir_level_change',
            f'{window}_hours_rolling_pressure_change', f'{window}_hours_rolling_change_from_last_hour_output_flow_rate_mean', f'{window}_hours_rolling_temperature',
            f'{window}_hours_rolling_precipitation', f'{window}_hours_rolling_humidity'
        ])

        non_weather_features.extend([
            f'{window}_hours_rolling_input_flow_rate_diff', f'{window}_hours_rolling_output_flow_rate_diff', f'{window}_hours_rolling_reservoir_level_change',
            f'{window}_hours_rolling_pressure_change', f'{window}_hours_rolling_change_from_last_hour_output_flow_rate_mean'
        ])
    
    # Use pd.concat to concatenate the new window features into the dataframe
    df = pd.concat([df, pd.DataFrame(window_features)], axis=1)
    
    return df, all_features, non_weather_features

def create_targets(input_df):
    df = input_df.copy()
    
    targets = []
    for i in range(1, 25):  
        df[f'target_{i}'] = df['change_from_last_hour_output_flow_rate_mean'].shift(-i)
        targets.append(f'target_{i}')
        
    return df, targets

def create_training_dataset(input_df):
    df = input_df.copy()
    
    df, all_lag_features, non_weather_lag_features = create_lag_features(df)
    df, all_window_features, non_weather_window_features = create_window_features(df)
    df, targets = create_targets(df)
    df.dropna(inplace=True)
    
    date_features = ['hour', 'day_of_week', 'week_of_year', 'year']
    all_features = date_features + all_lag_features + all_window_features + ['inputed_row']
    all_non_weather_features = date_features + non_weather_lag_features + non_weather_window_features + ['inputed_row']
    all_training_columns = ['timestamp'] + all_features + targets
    
    return df[all_training_columns].reset_index(drop=True), all_features, all_non_weather_features