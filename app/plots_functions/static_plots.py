import os
import plotly.graph_objects as go

# Construct the absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

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
    fig.update_layout(title={'text': 'Comparação do Fluxo Médio de Saída por Hora', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Hora do Dia', yaxis_title='Fluxo Médio de Saída (L/S)', template='plotly_white',plot_bgcolor='rgb(240,240,240)', xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f'{h}' for h in range(24)], gridcolor='rgba(0.05,0.05,0.05,0.05)'), yaxis=dict(tickformat=',.0f',gridcolor='rgba(0.05,0.05,0.5,0.05)'), legend=dict(x=0.02, y=0.98, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'))
    fig.update_traces(marker=dict(symbol='circle'))    
    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_question_2_plot_2(question_2_dataset):
    fig = go.Figure(data=[go.Bar(name='Dia Útil', x=question_2_dataset['hour'], y=question_2_dataset['avg_weekday_output_flow'], marker_color='blue'), go.Bar(name='Fim de Semana', x=question_2_dataset['hour'], y=question_2_dataset['avg_weekend_output_flow'], marker_color='red')])
    fig.update_layout(title={'text': 'Comparação do Fluxo Médio de Saída por Hora', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Hora do Dia', yaxis_title='Fluxo Médio de Saída (L/S)', barmode='group', template='plotly_white', plot_bgcolor='rgb(240,240,240)', xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f'{h}' for h in range(24)], gridcolor='rgba(0.0,0.0,0.0,0.0)'), yaxis=dict(tickformat=',.0f', gridcolor='rgba(0.10,0.10,0.10,0.10)'  ), legend=dict(x=0.02, y=0.98, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'))  
    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_question_3_plot_1(question_3_dataset):
    question_3_dataset["average_time_used_peak_hours_minutes"] = question_3_dataset["average_time_used_peak_hours"].apply(convert_to_minutes)
    question_3_dataset["average_time_used_offpeak_hours_minutes"] = question_3_dataset["average_time_used_offpeak_hours"].apply(convert_to_minutes)
    total_peak_minutes = 4 * 60
    total_offpeak_minutes = 20 * 60    
    question_3_dataset["proportion_peak_hours"] = question_3_dataset["average_time_used_peak_hours_minutes"] / total_peak_minutes
    question_3_dataset["proportion_offpeak_hours"] = question_3_dataset["average_time_used_offpeak_hours_minutes"] / total_offpeak_minutes

    fig = go.Figure(data=[go.Bar(name='Horário de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['proportion_peak_hours'], text=question_3_dataset['proportion_peak_hours'].apply(lambda x: f"{x:.2%}"), marker_color='#FF5733'), go.Bar(name='Fora de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['proportion_offpeak_hours'], text=question_3_dataset['proportion_offpeak_hours'].apply(lambda x: f"{x:.2%}"), marker_color='#33C4FF')])
    fig.update_layout(title={'text': 'Proporção de Tempo de Uso das Bombas', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Bombas', yaxis_title='Proporção de Tempo de Uso', barmode='group', yaxis=dict(tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], ticktext=['0%', '20%', '40%', '60%', '80%', '100%']))
    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_question_3_plot_2(question_3_dataset):
    question_3_dataset["average_time_used_peak_hours_minutes"] = question_3_dataset["average_time_used_peak_hours"].apply(convert_to_minutes)
    question_3_dataset["average_time_used_offpeak_hours_minutes"] = question_3_dataset["average_time_used_offpeak_hours"].apply(convert_to_minutes)

    fig = go.Figure(data=[go.Bar(name='Horário de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['average_time_used_peak_hours_minutes'], text=question_3_dataset['average_time_used_peak_hours'], marker_color='#FF5733'), go.Bar(name='Fora de Ponta', x=question_3_dataset['pump'], y=question_3_dataset['average_time_used_offpeak_hours_minutes'], text=question_3_dataset['average_time_used_offpeak_hours'], marker_color='#33C4FF', base=question_3_dataset['average_time_used_peak_hours_minutes'])])
    fig.update_layout(title={'text': 'Tempo Médio de Uso das Bombas', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Bombas', yaxis_title='Tempo Médio de Uso (minutos)', barmode='stack', yaxis=dict(tickvals=[0, 60, 120, 180, 240, 300, 360, 420, 480], ticktext=['0', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h']))
    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_question_6_plots(question_6_dataset):
    correlation_air_temp = question_6_dataset['output_flow_rate'].corr(question_6_dataset['air_temp_c'])
    correlation_total_precip_mm = question_6_dataset['output_flow_rate'].corr(question_6_dataset['total_precip_mm'])
    correlation_relative_humidity = question_6_dataset['output_flow_rate'].corr(question_6_dataset['relative_humidity_percentage'])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=question_6_dataset['air_temp_c'], y=question_6_dataset['output_flow_rate'], mode='markers', marker_color='purple'))
    fig1.update_layout(title={'text': f'Correlação entre Temperatura do Ar e Consumo de Água (r={correlation_air_temp:.2f})', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Temperatura do Ar (°C)', yaxis_title='Consumo de Água (L/s)')
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=question_6_dataset['total_precip_mm'], y=question_6_dataset['output_flow_rate'], mode='markers', marker_color='red'))
    fig2.update_layout(title={'text': f'Correlação entre Precipitacao e Consumo de Água (r={correlation_total_precip_mm:.2f})', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Precipitacao (mm)', yaxis_title='Consumo de Água (L/s)')
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=question_6_dataset['relative_humidity_percentage'], y=question_6_dataset['output_flow_rate'], mode='markers', marker_color='green'))
    fig3.update_layout(title={'text': f'Correlação entre Humidade Relativa e Consumo de Água (r={correlation_relative_humidity:.2f})', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Humidade Relativa (%)', yaxis_title='Consumo de Água (L/s)')
    
    plot1_html, plot2_html, plot3_html = fig1.to_html(full_html=False), fig2.to_html(full_html=False), fig3.to_html(full_html=False)
    return plot1_html, plot2_html, plot3_html