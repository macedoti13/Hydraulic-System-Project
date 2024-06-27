import plotly.graph_objects as go

def generate_question_6_plots(question_6_dataset):
    correlation_air_temp = question_6_dataset['output_flow_rate_mean'].corr(question_6_dataset['air_temp_c'])
    correlation_total_precip_mm = question_6_dataset['output_flow_rate_mean'].corr(question_6_dataset['total_precip_mm'])
    correlation_relative_humidity = question_6_dataset['output_flow_rate_mean'].corr(question_6_dataset['relative_humidity_percentage'])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=question_6_dataset['air_temp_c'], y=question_6_dataset['output_flow_rate_mean'], mode='markers', marker_color='purple'))
    fig1.update_layout(title={'text': f'Correlação entre Temperatura do Ar e Consumo de Água (r={correlation_air_temp:.2f})', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Temperatura do Ar (°C)', yaxis_title='Consumo de Água (L/s)')
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=question_6_dataset['total_precip_mm'], y=question_6_dataset['output_flow_rate_mean'], mode='markers', marker_color='red'))
    fig2.update_layout(title={'text': f'Correlação entre Precipitacao e Consumo de Água (r={correlation_total_precip_mm:.2f})', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Precipitacao (mm)', yaxis_title='Consumo de Água (L/s)')
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=question_6_dataset['relative_humidity_percentage'], y=question_6_dataset['output_flow_rate_mean'], mode='markers', marker_color='green'))
    fig3.update_layout(title={'text': f'Correlação entre Humidade Relativa e Consumo de Água (r={correlation_relative_humidity:.2f})', 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Humidade Relativa (%)', yaxis_title='Consumo de Água (L/s)')
    
    plot1_html, plot2_html, plot3_html = fig1.to_html(full_html=False), fig2.to_html(full_html=False), fig3.to_html(full_html=False)
    return plot1_html, plot2_html, plot3_html