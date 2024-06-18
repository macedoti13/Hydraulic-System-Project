from flask import Flask, render_template, request
import pandas as pd
import pickle
import sys
import os

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load datasets
water_consumption = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'curated_datasets', 'water_consumption_curated.parquet'))
forecasting_dataset = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'curated_datasets', 'forecasting_dataset.parquet'))

# load forecasters
output_flow_forecaster = pickle.load(open(os.path.join(BASE_DIR, 'models', 'forecaster.pkl'), 'rb'))
output_flow_forecaster_with_weather = pickle.load(open(os.path.join(BASE_DIR, 'models', 'forecaster_with_weather.pkl'), 'rb'))
input_flow_forecaster = pickle.load(open(os.path.join(BASE_DIR, 'models', 'input_flow_forecaster.pkl'), 'rb'))

# import from the utils module here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.scripts.functions.main_functions import simulate_emptying, forecast_next_24_hours_output_flow_rate

# generate app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for the static plots page
@app.route('/static-plots')
def static_plots():
    return render_template('static_plots.html')

# Route for the forecasting plots page
@app.route('/forecasting-plots', methods=['GET', 'POST'])
def foresting_plots():
    if request.method == 'POST':
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])

        # create all dataframes that require the user input
        output_flow_24_hour_forecast_df = forecast_next_24_hours_output_flow_rate(
            water_consumption, output_flow_forecaster, input_flow_forecaster, year, month, day, hour, minute, False
        )
        
        output_flow_24_hour_forecast_with_weather_df = forecast_next_24_hours_output_flow_rate(
            water_consumption, output_flow_forecaster_with_weather, input_flow_forecaster, year, month, day, hour, minute, True
        )
        
        emptying_simulation_df = simulate_emptying(
            water_consumption, output_flow_forecaster, input_flow_forecaster, year, month, day, hour, minute, False
        )
        
        # TO DO: create dataframe for pump status optimization here (not yet implemented in the utils module)

        # create the plots that will be displayed on the forecasting plots page here
        
        return render_template('forecasting_plots.html')
    
    return render_template('forecasting_plots.html')

# Route for the model performance page
@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html')

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
