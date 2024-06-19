import pandas as pd
import os
import plotly.graph_objects as go

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# load datasets
question_2_dataset = pd.read_parquet(os.path.join(BASE_DIR, '..', 'data', 'questions_datasets', 'question_2_dataset.parquet'))
question_3_dataset = pd.read_parquet(os.path.join(BASE_DIR, '..', 'data', 'questions_datasets', 'question_3_dataset.parquet'))

# helper functions
def convert_to_minutes(time_str):
    hours, minutes = 0, 0
    if "hours" in time_str:
        hours = int(time_str.split(" hours")[0])
        minutes = int(time_str.split(" and ")[1].split(" minutes")[0])
    else:
        minutes = int(time_str.split(" minutes")[0])
    return hours * 60 + minutes

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


def generate_question_3_plot_1(question_3_dataset):
    question_3_dataset["average_time_used_peak_hours_minutes"] = question_3_dataset["average_time_used_peak_hours"].apply(convert_to_minutes)
    question_3_dataset["average_time_used_offpeak_hours_minutes"] = question_3_dataset["average_time_used_offpeak_hours"].apply(convert_to_minutes)

    total_peak_minutes = 4 * 60
    total_offpeak_minutes = 20 * 60
    
    question_3_dataset["proportion_peak_hours"] = question_3_dataset["average_time_used_peak_hours_minutes"] / total_peak_minutes
    question_3_dataset["proportion_offpeak_hours"] = question_3_dataset["average_time_used_offpeak_hours_minutes"] / total_offpeak_minutes
    
    fig = go.Figure(data=[
        go.Bar(name='Horário de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['proportion_peak_hours'], text=question_3_dataset['proportion_peak_hours'].apply(lambda x: f"{x:.2%}"), marker_color='#FF5733'),
        go.Bar(name='Fora de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['proportion_offpeak_hours'], text=question_3_dataset['proportion_offpeak_hours'].apply(lambda x: f"{x:.2%}"), marker_color='#33C4FF')
    ])

    fig.update_layout(title='Proporção de Tempo de Uso das Bombas em Horário de Ponta e Fora de Ponta', xaxis_title='Bombas', yaxis_title='Proporção de Tempo de Uso', barmode='group', yaxis=dict(tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], ticktext=['0%', '20%', '40%', '60%', '80%', '100%']))
    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_question_3_plot_2(question_3_dataset):
    question_3_dataset["average_time_used_peak_hours_minutes"] = question_3_dataset["average_time_used_peak_hours"].apply(convert_to_minutes)
    question_3_dataset["average_time_used_offpeak_hours_minutes"] = question_3_dataset["average_time_used_offpeak_hours"].apply(convert_to_minutes)

    fig = go.Figure(data=[
        go.Bar(name='Horário de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['average_time_used_peak_hours_minutes'], text=question_3_dataset['average_time_used_peak_hours'], marker_color='#FF5733'),
        go.Bar(name='Fora de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['average_time_used_offpeak_hours_minutes'], text=question_3_dataset['average_time_used_offpeak_hours'], marker_color='#33C4FF', base=question_3_dataset['average_time_used_peak_hours_minutes'])
    ])

    fig.update_layout(title='Tempo Médio de Uso das Bombas em Horário de Ponta e Fora de Ponta', xaxis_title='Bombas', yaxis_title='Tempo Médio de Uso (minutos)', barmode='stack', yaxis=dict(tickvals=[0, 60, 120, 180, 240, 300, 360, 420, 480], ticktext=['0', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h']))
    plot_html = fig.to_html(full_html=False)
    return plot_html