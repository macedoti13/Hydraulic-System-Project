import plotly.graph_objects as go

def plot_pump_schedule(schedule):
    horas = list(range(1, 25))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=horas,
        y=schedule,
        mode='lines+markers',
        name='Status da Bomba',
        line=dict(color='royalblue', width=2),
        marker=dict(color='royalblue', size=8)
    ))
    
    fig.update_layout(
        title={
            'text': 'Programação das Bombas nas Próximas 24 Horas',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Hora do Dia',
        yaxis_title='Status da Bomba (1=Ligada, 0=Desligada)',
        xaxis=dict(
            tickmode='array',
            tickvals=horas,
            ticktext=[f'{h}h' for h in horas]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Desligada', 'Ligada']
        ),
        template='plotly_white',
        showlegend=True,
        title_font=dict(size=15, family='Arial', color='DarkSlateGrey'),
        xaxis_title_font=dict(size=12, family='Arial', color='DarkSlateGrey'),
        yaxis_title_font=dict(size=12, family='Arial', color='DarkSlateGrey'),
        legend=dict(
            title='Legenda',
            font=dict(size=12, family='Arial', color='DarkSlateGrey')
        ),
        plot_bgcolor='#e0f7fa'
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    
    fig_html = fig.to_html(full_html=False)
    return fig_html