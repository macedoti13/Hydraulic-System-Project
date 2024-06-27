import pandas as pd
import os

# functions
from model_training_functions import train_xgb_model, save_model, print_error

# paths
TRAINING_DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../data/silver/training_dataset.parquet")
MODEL_WITH_WEATHER_SAVING_PATH = os.path.join(os.path.dirname(__file__), "../../models/weather/xgb_weather_9h.pkl")
MODEL_SAVING_PATH = os.path.join(os.path.dirname(__file__), "../../models/no_weather/xgb_9h.pkl")

def main():
    
    # read the dataframe
    df = pd.read_parquet(TRAINING_DATASET_PATH)
    
    # train the models
    xgb_with_weather = train_xgb_model(df, with_weather=True, target='target_9')
    xgb_without_weather = train_xgb_model(df, with_weather=False, target='target_9')
    
    # save the models
    save_model(xgb_with_weather, MODEL_WITH_WEATHER_SAVING_PATH)
    save_model(xgb_without_weather, MODEL_SAVING_PATH)
    
    # print the results
    print("\nWith weather:")
    print_error(xgb_with_weather, df, with_weather=True)
    print()
    print("Without weather:")
    print_error(xgb_without_weather, df, with_weather=False)
    
    
if __name__ == "__main__":
    main()