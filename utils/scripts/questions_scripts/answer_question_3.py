from pathlib import Path
import pandas as pd
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root / "data"))

# Paths
QUESTIONS_DATASETS_PATH = project_root / "data/questions_datasets"
CURATED_DATASETS_PATH = project_root / "data/curated_datasets"
WATER_CONSUMPTION_CURATED_PATH = CURATED_DATASETS_PATH / "water_consumption_curated.parquet"
SAVING_PATH = QUESTIONS_DATASETS_PATH / "question_3_dataset.parquet"

def get_avg_use_per_bomb_in_minutes_corrected(original_df: pd.DataFrame) -> pd.DataFrame:
    df = original_df.copy()
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create date column
    df['date'] = df['timestamp'].dt.date
    
    # Calculate peak hours
    peak_hours = (df["hour"] >= 18) & (df["hour"] <= 21)
    df["is_peak_hour"] = peak_hours
    
    # Calculate duration of pumps
    df['pump_1_duration'] = df['pump_1'] * df['time_passed_seconds']
    df['pump_2_duration'] = df['pump_2'] * df['time_passed_seconds']
    
    # Sum water bombs usage time per day and hour
    daily_peak_usage = df[df['is_peak_hour']].groupby('date').agg({'pump_1_duration': 'sum', 'pump_2_duration': 'sum'})
    daily_off_peak_usage = df[~df['is_peak_hour']].groupby('date').agg({'pump_1_duration': 'sum', 'pump_2_duration': 'sum'})

    # Calculate water bombs average usage time per day in minutes
    gmb_1_peak_avg = daily_peak_usage['pump_1_duration'].mean() / 60  # convert seconds to minutes
    gmb_1_off_peak_avg = daily_off_peak_usage['pump_1_duration'].mean() / 60  # convert seconds to minutes

    gmb_2_peak_avg = daily_peak_usage['pump_2_duration'].mean() / 60  # convert seconds to minutes
    gmb_2_off_peak_avg = daily_off_peak_usage['pump_2_duration'].mean() / 60  # convert seconds to minutes
    
    def convert_to_hours_and_minutes(minutes):
        if pd.isna(minutes):
            return "0 hours and 0 minutes"
        total_minutes = int(minutes)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours} hours and {minutes} minutes"
    
    data = {
        'pump': ['pump_1', 'pump_2'],
        'average_time_used_peak_hours': [
            convert_to_hours_and_minutes(gmb_1_peak_avg), 
            convert_to_hours_and_minutes(gmb_2_peak_avg)
        ],
        'average_time_used_offpeak_hours': [
            convert_to_hours_and_minutes(gmb_1_off_peak_avg), 
            convert_to_hours_and_minutes(gmb_2_off_peak_avg)
        ]
    }
    
    result_df = pd.DataFrame(data)
    
    return result_df

def main():
    df = pd.read_parquet(WATER_CONSUMPTION_CURATED_PATH)
    question_3_dataset = get_avg_use_per_bomb_in_minutes_corrected(df)
    question_3_dataset.to_parquet(SAVING_PATH)
    
if __name__ == "__main__":
    main()