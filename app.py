import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import random
from datetime import timedelta
from callbacks import read_data, outlier_score as outlier_score_, list_devices
from functools import cache


@cache
def get_user_dataframe(device_id):
    df = read_data(device_id)
    expected = {"true", "pred", "base", "error", "Timestamp",
                "ShiftAmountHours", "DeviceID"}
    got = set(df.columns)
    assert got == expected, f"missing: {expected - got}, extra: {got - expected}"

    df["PredictionFor"] = df["Timestamp"] + pd.to_timedelta(df["ShiftAmountHours"], unit="hours")
    pivot_df = df.pivot(index='PredictionFor', columns='ShiftAmountHours', values='error')
    err_min, err_max = pivot_df.min(skipna=True).min(), pivot_df.max(skipna=True).max()
    outlier_sc = outlier_score_(pivot_df)
    outlier_sc_min, outlier_sc_max = outlier_sc.min(), outlier_sc.max()
    return df, pivot_df, outlier_sc, (err_min, err_max), (outlier_sc_min, outlier_sc_max)


def filter_df(pivot_df, outlier_score, outlier_score_range):
    outlier_score_min, outlier_score_max = sorted(outlier_score_range)
    valid_rows = outlier_score[(outlier_score >= outlier_score_min) & (outlier_score <= outlier_score_max)].index
    filtered_pivot_df = pivot_df.loc[valid_rows]
    y_vals_dt = pd.Series(filtered_pivot_df.index.get_level_values("PredictionFor"))
    if not (len(y_vals_dt) <= 1000):  # NOTE: a lot of timestamp labels on y-axis significantly slows down the rendering of heatmap
        y_vals_dt = y_vals_dt[:: max(1, (len(y_vals_dt) // 1000))]
    return filtered_pivot_df, y_vals_dt


DEVICES = list_devices()

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3("Overlapping Forecasts Visualization"),

    html.Div([
        html.Label("Select DeviceID:"),
        dcc.Dropdown(
            id='device-dropdown',
            options=[{'label': d, 'value': d} for d in DEVICES],
            value=DEVICES[0],
            clearable=False,
            style={'width': '300px'}
        ),
    ], style={'width': '40%', 'margin': '10px'}),

    html.Div([
        html.Label("Heatmap color gradient:"),
        dcc.RangeSlider(
            id='error-range-slider',
            step=0.1,
            marks=None,
            tooltip={"placement": "bottom"}
        )
    ], style={'width': '40%', 'margin': '10px'}),

    html.Div([
        html.Label("Outlier selection:"),
        dcc.RangeSlider(
            id='outlier-score-range-slider',
            step=0.1,
            marks=None,
            tooltip={"placement": "bottom"}
        )
    ], style={'width': '40%', 'margin': '10px'}),

    html.Div([
        dcc.Graph(id='scatter-plot', style={'width': '600px'})
    ], style={'height': '600px', "width": "300px", 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='heatmap', style={'width': '1200px'})
    ], style={'height': '600px', 'overflowY': 'scroll', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='single-prediction-plot', style={'width': '600px'})
    ], style={'display': 'inline-block', 'verticalAlign': 'top'})
])


@app.callback(
    Output('error-range-slider', 'min'),
    Output('error-range-slider', 'max'),
    Output('error-range-slider', 'value'),
    Output('outlier-score-range-slider', 'min'),
    Output('outlier-score-range-slider', 'max'),
    Output('outlier-score-range-slider', 'value'),
    Input('device-dropdown', 'value')
)
def set_sliders(device_id):
    _, _, _, error_range, outlier_score_range = get_user_dataframe(device_id)
    return (
        error_range[0],
        error_range[1],
        [error_range[0], error_range[1]],
        outlier_score_range[0],
        outlier_score_range[1],
        [outlier_score_range[0], outlier_score_range[1]]
    )


@app.callback(
    Output('heatmap', 'figure'),
    Input('device-dropdown', 'value'),
    Input('error-range-slider', 'value'),
    Input('outlier-score-range-slider', 'value'),
)
def update_heatmap(device_id, error_range, outlier_score_range):
    _, pivot_df, outlier_score, _, _ = get_user_dataframe(device_id)

    zmin, zmax = sorted(error_range)
    filtered_pivot_df, y_vals_dt = filter_df(pivot_df, outlier_score, outlier_score_range)

    x_vals = filtered_pivot_df.columns
    y_vals = filtered_pivot_df.index
    z_vals = filtered_pivot_df.values

    fig = go.Figure(
        data=go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            colorscale=[[0, 'green'], [1, 'red']],
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            hovertemplate='<br>date=%{y}<br>shift=%{x}<br>error=%{z}<extra></extra>'
        )
    )
    fig.update_layout(
        title="Error Heatmap",
        xaxis_title="ShiftAmountHours",
        yaxis_title="PredictionFor",
        height=max(600, 15 * len(y_vals)),
        width=1200
    )
    fig.update_yaxes(
        type="category",
        tickmode="array",
        tickvals=y_vals_dt,
        ticktext=[ts.strftime("%Y-%m-%d %H:%M") for ts in y_vals_dt],
        autorange="reversed",
    )
    fig.update_xaxes(
        tickmode='array',
        side="top",
        tickvals=x_vals,
        ticktext=[str(x).rjust(2, "0") for x in x_vals]
    )
    return fig


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('device-dropdown', 'value'),
    Input('outlier-score-range-slider', 'value'),
)
def update_scatter(device_id, outlier_score_range):
    _, pivot_df, outlier_score, _, _ = get_user_dataframe(device_id)

    _, y_vals_dt = filter_df(pivot_df, outlier_score, outlier_score_range)
    scatter_x = [random.uniform(-0.1, 0.1) for _ in range(len(y_vals_dt))]

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=scatter_x,
        y=y_vals_dt,
        mode='markers',
        text=[ts.strftime("%Y-%m-%d %H:%M") for ts in y_vals_dt],
        hovertemplate='date=%{text}<extra></extra>'
    ))
    scatter_fig.update_layout(title="Outlier density (jitted)", height=600, width=300)
    scatter_fig.update_xaxes(
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2,
        showline=False,
        showticklabels=False,
    )
    scatter_fig.update_yaxes(
        autorange="reversed",
        showgrid=False
    )
    return scatter_fig


@app.callback(
    Output('single-prediction-plot', 'figure'),
    Input('heatmap', 'clickData'),
    Input('device-dropdown', 'value')
)
def update_single_prediction_plot(clickData, device_id):
    df, _, _, _, _ = get_user_dataframe(device_id)

    fig = go.Figure().update_layout(title="Prediction", height=600, width=600)
    if not clickData or 'points' not in clickData or not clickData['points']:
        return fig

    shift = clickData['points'][0]['x']
    prediction_for = clickData['points'][0]['y']
    timestamp = pd.to_datetime(prediction_for) - timedelta(hours=shift)
    prediction = df[df['Timestamp'] == timestamp]
    if not prediction.empty:
        x = prediction["Timestamp"] + pd.to_timedelta(prediction["ShiftAmountHours"], unit="hours")
        fig.update_layout(title=f"Prediction for {timestamp}", height=600, width=600)
        fig.add_trace(go.Scatter(x=x, y=prediction['pred'], mode='lines', name='Predicted'))
        fig.add_trace(go.Scatter(x=x, y=prediction['true'], mode='lines', name='True'))
        fig.add_trace(go.Scatter(x=x, y=prediction['base'], mode='lines', name='Baseline'))
        x_line = x.iloc[int(shift)]
        fig.add_shape(
            type='line',
            x0=x_line, x1=x_line,
            y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='black', dash='dash')
        )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
