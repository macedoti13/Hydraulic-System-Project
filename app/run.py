from flask import Flask, render_template, request
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# functions to generate the data used by the plots
from functions.answer_questions.answer_questions_4_and_7 import forecast_next_24_hours_output_flow_rate
from functions.answer_questions.answer_question_5 import simulate_emptying_reservoir
from functions.answer_questions.answer_question_8 import find_best_pump_schedule

# functions to generate the plots
from functions.plots.question_2_plots import generate_question_2_plot_1, generate_question_2_plot_2
from functions.plots.question_3_plots import generate_question_3_plot_1, generate_question_3_plot_2
from functions.plots.questions_4_and_7_plots import create_questions_4_and_7_plot_1
from functions.plots.question_5_plots import create_question_5_plot
from functions.plots.question_6_plots import generate_question_6_plots
from functions.plots.question_8_plot import plot_pump_schedule

# load datasets
water_consumption_silver = pd.read_parquet(os.path.join(os.path.dirname(__file__), "../data/silver/water_consumption_silver.parquet"))
question_2_dataset = pd.read_parquet(os.path.join(os.path.dirname(__file__), "../data/gold/question_2_answer.parquet"))
question_3_dataset = pd.read_parquet(os.path.join(os.path.dirname(__file__), "../data/gold/question_3_answer.parquet"))

# app 
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static-plots')
def static_plots():
    
    # question 2 plots
    question_2_plot_1 = generate_question_2_plot_1(question_2_dataset)
    question_2_plot_2 = generate_question_2_plot_2(question_2_dataset)
    
    # question 3 plots
    question_3_plot_1 = generate_question_3_plot_1(question_3_dataset)
    question_3_plot_2 = generate_question_3_plot_2(question_3_dataset)
    
    # question 6 plots
    question_6_plot_1, question_6_plot_2, question_6_plot_3 = generate_question_6_plots(water_consumption_silver)
    
    # render template
    return render_template(
        'static_plots.html',
        plot_html=question_2_plot_1,
        plot_2_html=question_2_plot_2,
        plot_3_html=question_3_plot_1,
        plot_4_html=question_3_plot_2,
        plot_5_html=question_6_plot_1,
        plot_6_html=question_6_plot_2,
        plot_7_html=question_6_plot_3
    )

@app.route('/forecasting-plots', methods=['GET', 'POST'])
def forecasting_plots():
    
    # wait user input
    if request.method == "POST":
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        
        # generate the data from the user input
        output_flow_prediction_dataset = forecast_next_24_hours_output_flow_rate(year, month, day, hour, save_df=False)
        simulation_dataset, _ = simulate_emptying_reservoir(year, month, day, hour, save_df=False)
        best_schedule, best_reward = find_best_pump_schedule(year, month, day, hour)
        
        # forecast the next 24 hours output flow rate and simulate the emptying of the reservoir
        question_4_and_7_plot_1 = create_questions_4_and_7_plot_1(output_flow_prediction_dataset)
        question_5_plot = create_question_5_plot(simulation_dataset)
        question_8_plot = plot_pump_schedule(best_schedule)
        
        # render template with the plots
        return render_template(
            'forecasting_plots.html',
            plot_html=question_4_and_7_plot_1,
            plot_2_html=question_5_plot,
            plot_3_html=question_8_plot
        )
        
    # render the template with the form to get the user input
    return render_template('forecasting_plots.html')
        

@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html')

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)
    
if __name__ == "__main__":
    main() 