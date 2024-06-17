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
SAVING_PATH = QUESTIONS_DATASETS_PATH / "question_2_dataset.parquet"

def get_average_flow_out_across_day(original_df: pd.DataFrame) -> pd.DataFrame:
    df = original_df.copy()
    
    # Separate data into weekdays and weekends
    df_weekdays = df[df['weekday'] < 5]
    df_weekends = df[df['weekday'] >= 5]

    # Calculate average flow out per hour for weekdays and weekends
    flow_out_weekdays = df_weekdays.groupby('hour')['output_flow_rate'].mean().reset_index().rename(columns={'output_flow_rate': 'avg_weekday_output_flow'})
    flow_out_weekends = df_weekends.groupby('hour')['output_flow_rate'].mean().reset_index().rename(columns={'output_flow_rate': 'avg_weekend_output_flow'})
    
    # Merge the weekday and weekend data into a single DataFrame
    df_combined = pd.merge(flow_out_weekdays, flow_out_weekends, on='hour', how='outer')
    
    return df_combined
def main():
    df = pd.read_parquet(WATER_CONSUMPTION_CURATED_PATH)
    question_2_dataset = get_average_flow_out_across_day(df)
    question_2_dataset.to_parquet(SAVING_PATH)
    
if __name__ == "__main__":
    main()