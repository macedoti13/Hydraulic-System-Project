import plotly.graph_objects as go

def create_questions_4_and_7_plot_1(input_df):
    df = input_df.copy()
    real_data = df[df.forecasted==False]
    forecasted_data = df[df.forecasted==True]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_data.timestamp, y=real_data.output_flow_rate_mean_no_weather, mode='lines+markers', name='Últimas 24 horas', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecasted_data.timestamp, y=forecasted_data.output_flow_rate_mean_no_weather, mode='lines+markers', name='Previsão do próximo dia (sem temperatura)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=forecasted_data.timestamp, y=forecasted_data.output_flow_rate_mean_weather, mode='lines+markers', name='Previsão do próximo dia (com temperatura)', line=dict(color='green')))
    fig.update_layout(title={'text': 'Previsão da Saída de Água', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Horario', yaxis_title='Saída de Água (L/s)', legend_title='Type')
    plot_html = fig.to_html(full_html=False)
    return plot_html