from flask import Flask, render_template, request
import pandas as pd
import sys
import os

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import datasets
water_consumption_path = os.path.join(BASE_DIR, 'data', 'curated_datasets', 'water_consumption_curated.parquet')
forecasting_dataset_paht = os.path.join(BASE_DIR, 'data', 'curated_datasets', 'forecasting_dataset.parquet')
water_consumption = pd.read_parquet(water_consumption_path)
forecasting_dataset = pd.read_parquet(forecasting_dataset_paht)

# import from the utils module here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
@app.route('/forecasting-plots')
def user_input():
    return render_template('forecasting_plots.html')

# Route for the model performance page
@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    timestamp = request.form['timestamp']
    # Add your logic to handle the timestamp and generate plots here
    return render_template('user_input.html', timestamp=timestamp)

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
