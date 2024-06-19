from flask import Flask, render_template, request
import pandas as pd
import pickle
import sys
import os
import plotly.graph_objects as go

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load datasets
water_consumption = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'curated_datasets', 'water_consumption_curated.parquet'))
forecasting_dataset = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'curated_datasets', 'forecasting_dataset.parquet'))
question_2_dataset = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'questions_datasets', 'question_2_dataset.parquet'))

# load forecasters
output_flow_forecaster = pickle.load(open(os.path.join(BASE_DIR, 'models', 'forecaster.pkl'), 'rb'))
output_flow_forecaster_with_weather = pickle.load(open(os.path.join(BASE_DIR, 'models', 'forecaster_with_weather.pkl'), 'rb'))
input_flow_forecaster = pickle.load(open(os.path.join(BASE_DIR, 'models', 'input_flow_forecaster.pkl'), 'rb'))

# import from the utils module here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.scripts.functions.main_functions import simulate_emptying, forecast_next_24_hours_output_flow_rate

# plots functions
def generate_question_2_plot_1(question_2_dataset):
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=question_2_dataset['hour'], y=question_2_dataset['avg_weekday_output_flow'], mode='lines+markers', name='Dia Útil', line=dict(color='blue', width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=question_2_dataset['hour'], y=question_2_dataset['avg_weekend_output_flow'], mode='lines+markers', name='Fim de Semana', line=dict(color='red', width=2), marker=dict(size=5)))

    fig.update_layout(
        xaxis_title='Hora do Dia',
        yaxis_title='Fluxo Médio de Saída (L/S)',
        template='plotly_white',
        plot_bgcolor='rgb(240,240,240)', 
        xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f'{h}' for h in range(24)], gridcolor='rgba(0.05,0.05,0.05,0.05)'),
        yaxis=dict(tickformat=',.0f',gridcolor='rgba(0.05,0.05,0.5,0.05)'),
        legend=dict(x=0.02, y=0.98, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )
    fig.update_traces(marker=dict(symbol='circle'))    
    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_question_2_plot_2(question_2_dataset):
    fig = go.Figure(data=[
        go.Bar(name='Dia Útil', x=question_2_dataset['hour'], y=question_2_dataset['avg_weekday_output_flow'], marker_color='blue'),
        go.Bar(name='Fim de Semana', x=question_2_dataset['hour'], y=question_2_dataset['avg_weekend_output_flow'], marker_color='red')
    ])

    # Update layout
    fig.update_layout(
        title='Comparação do Fluxo Médio de Saída por Hora',
        xaxis_title='Hora do Dia',
        yaxis_title='Fluxo Médio de Saída (L/S)',
        barmode='group',
        template='plotly_white',
        plot_bgcolor='rgb(240,240,240)',  
        xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f'{h}' for h in range(24)], gridcolor='rgba(0.0,0.0,0.0,0.0)'),
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(0.10,0.10,0.10,0.10)'  ),
        legend=dict(x=0.02, y=0.98, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )  
    plot_html = fig.to_html(full_html=False)
    return plot_html


# generate app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for the static plots page
@app.route('/static-plots')
def static_plots():
    # Generate the plots that will be displayed on the static plots page
    question_1_plot = generate_question_2_plot_1(question_2_dataset)
    question_2_plot = generate_question_2_plot_2(question_2_dataset)
    question_3_plot = generate_question_2_plot_2(question_2_dataset)  # Add your function to generate plot 3
    question_4_plot = generate_question_2_plot_2(question_2_dataset)  # Add your function to generate plot 4

    return render_template('static_plots.html', plot_html=question_1_plot, plot_2_html=question_2_plot, plot_3_html=question_3_plot, plot_4_html=question_4_plot)

# Route for the forecasting plots page
@app.route('/forecasting-plots', methods=['GET', 'POST'])
def forecasting_plots():
    if request.method == 'POST':
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])

        # create all dataframes that require the user input
        output_flow_24_hour_forecast_question_2_dataset = forecast_next_24_hours_output_flow_rate(
            water_consumption, output_flow_forecaster, input_flow_forecaster, year, month, day, hour, minute, False
        )
        
        output_flow_24_hour_forecast_with_weather_question_2_dataset = forecast_next_24_hours_output_flow_rate(
            water_consumption, output_flow_forecaster_with_weather, input_flow_forecaster, year, month, day, hour, minute, True
        )
        
        emptying_simulation_question_2_dataset = simulate_emptying(
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
