import pandas as pd
import os
    
def answer_question_3():
    df = pd.read_parquet(os.path.join(os.path.dirname(__file__),"../../data/silver/water_consumption_silver.parquet"))
    df = df[['timestamp', 'day', 'hour', 'pump_1_duration_sum', 'pump_2_duration_sum', 'inputed_row']]
    df['date'] = df.timestamp.dt.date
    peak_hours = (df["hour"] >= 18) & (df["hour"] <= 21)
    df["is_peak_hour"] = peak_hours
    daily_peak_usage = df[(df.inputed_row==False)&(df['is_peak_hour'])].groupby('date').agg({'pump_1_duration_sum': 'sum', 'pump_2_duration_sum': 'sum'})
    daily_off_peak_usage = df[(df.inputed_row==False)&(~df['is_peak_hour'])].groupby('date').agg({'pump_1_duration_sum': 'sum', 'pump_2_duration_sum': 'sum'})
    gmb_1_peak_avg = daily_peak_usage['pump_1_duration_sum'].mean() / 60
    gmb_1_off_peak_avg = daily_off_peak_usage['pump_1_duration_sum'].mean() / 60
    gmb_2_peak_avg = daily_peak_usage['pump_2_duration_sum'].mean() / 60
    gmb_2_off_peak_avg = daily_off_peak_usage['pump_2_duration_sum'].mean() / 60 
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
    data = pd.DataFrame(data)
    data.to_parquet(os.path.join(os.path.dirname(__file__),"../../data/gold/question_3_answer.parquet"))
    
def main():
    answer_question_3()
    
if __name__ == '__main__':
    main()