import os
import pandas as pd
import plotly.graph_objects as go

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)


def select_first_and_middle(group):
    rows = group.iloc[::7]
    return rows

def create_question_4_plot(df):
    real_data = df[df['forecasted'] == False].groupby('hour', group_keys=False).apply(select_first_and_middle)
    forecasted_data = df[df['forecasted'] == True].groupby('hour', group_keys=False).apply(select_first_and_middle)
    real_data_complete = pd.concat([real_data, forecasted_data]).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_data_complete.index[real_data_complete['forecasted'] == False],y=real_data_complete['output_flow_rate'][real_data_complete['forecasted'] == False],mode='lines+markers',name='Ultimas 24 horas',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=real_data_complete.index[real_data_complete['forecasted'] == True],y=real_data_complete['output_flow_rate'][real_data_complete['forecasted'] == True],mode='lines+markers',name='Previsao das proximas 24 horas',line=dict(color='red')))
    fig.update_layout(title={'text': 'Previsao da Saida de Agua', 'x': 0.5, 'xanchor': 'center'},xaxis_title='',yaxis_title='Saida de Agua (L/s)',legend_title='Type')
    fig.update_traces(marker=dict(symbol='circle'))
    plot_html = fig.to_html(full_html=False)
    return plot_html


def create_question_5_plot(df):
    df['time'] = pd.to_datetime(df[['day', 'hour', 'minute']].astype(str).agg('-'.join, axis=1), format='%d-%H-%M')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'],y=df['reservoir_level_liters'],mode='lines+markers',name='Reservoir Level',line=dict(color='blue')))
    fig.update_layout(title={'text': 'Nivel do Reservatorio caindo com o tempo', 'x': 0.5, 'xanchor': 'center'},xaxis_title='Horarion',yaxis_title='Litros no Reservatorio',legend_title='Type')
    plot_html = fig.to_html(full_html=False)
    return plot_html
