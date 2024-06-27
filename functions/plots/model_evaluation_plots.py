import plotly.graph_objects as go 

def plot_comparison_with_weather(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['target_1'], mode='lines+markers', name='Real', line=dict(color='blue', width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['prediction_weather'], mode='lines+markers', name='Previsão com Clima', line=dict(color='red', width=2), marker=dict(size=5)))
    fig.update_layout(
        title={'text': 'Comparação da mudanca na saida de agua com Previsão com Clima', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Índice',
        yaxis_title='Valor',
        template='plotly_white',
        plot_bgcolor='rgb(240,240,240)',
        xaxis=dict(gridcolor='rgba(0.05,0.05,0.05,0.05)'),
        yaxis=dict(gridcolor='rgba(0.05,0.05,0.5,0.05)'),
        legend=dict(x=0.02, y=0.98, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )
    fig.update_traces(marker=dict(symbol='circle'))    
    fig_html = fig.to_html(full_html=False)
    return fig_html

def plot_comparison_no_weather(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['target_1'], mode='lines+markers', name='Real', line=dict(color='blue', width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['prediction_no_weather'], mode='lines+markers', name='Previsão sem Clima', line=dict(color='green', width=2), marker=dict(size=5)))
    fig.update_layout(
        title={'text': 'Comparação da mudanca na saida de agua com Previsão sem Clima', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Índice',
        yaxis_title='Valor',
        template='plotly_white',
        plot_bgcolor='rgb(240,240,240)',
        xaxis=dict(gridcolor='rgba(0.05,0.05,0.05,0.05)'),
        yaxis=dict(gridcolor='rgba(0.05,0.05,0.5,0.05)'),
        legend=dict(x=0.02, y=0.98, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )
    fig.update_traces(marker=dict(symbol='circle'))    
    fig_html = fig.to_html(full_html=False)
    return fig_html
