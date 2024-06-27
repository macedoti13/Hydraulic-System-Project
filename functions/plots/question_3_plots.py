import plotly.graph_objects as go

def convert_to_minutes(time_str):
    hours, minutes = 0, 0
    if "hours" in time_str:
        hours = int(time_str.split(" hours")[0])
        minutes = int(time_str.split(" and ")[1].split(" minutes")[0])
    else:
        minutes = int(time_str.split(" minutes")[0])
    return hours * 60 + minutes

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