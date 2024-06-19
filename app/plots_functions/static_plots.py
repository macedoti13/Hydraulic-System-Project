import pandas as pd
import os
import plotly.graph_objects as go

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# load datasets
question_2_dataset = pd.read_parquet(os.path.join(BASE_DIR, '..', 'data', 'questions_datasets', 'question_2_dataset.parquet'))
question_3_dataset = pd.read_parquet(os.path.join(BASE_DIR, '..', 'data', 'questions_datasets', 'question_3_dataset.parquet'))

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
    pass


def generate_question_3_plot_2(question_3_dataset):
    pass
