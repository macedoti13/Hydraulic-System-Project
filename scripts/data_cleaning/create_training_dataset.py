import pandas as pd
import os

# functions
from create_training_dataset_functions import *

# paths
WATER_CONSUMPTION_SILVER_PATH = os.path.join(os.path.dirname(__file__), "../../data/silver/water_consumption_silver.parquet")
TRAINING_DATASET_SAVING_PATH = os.path.join(os.path.dirname(__file__), "../../data/silver/training_dataset.parquet")

def main():
    
    # read the dataframe
    df = pd.read_parquet(WATER_CONSUMPTION_SILVER_PATH)
    
    # process the dataframe
    df, _, _ = create_training_dataset(df)
    
    # save the dataframe
    df.to_parquet(TRAINING_DATASET_SAVING_PATH)
    
if __name__ == "__main__":
    main()