from datetime import datetime
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

class Dashboard:
    def __init__(self):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
            ]
        )
        self.app.config.suppress_callback_exceptions = True
        
        self.current_predictions = None
        self.current_alerts = []
        self.last_update = None
        
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                html.H1("Airfield Weather Prediction System"),
                html.Div([
                    html.Div(id='last-update-time'),
                    dcc.Interval(
                        id='main-update',
                        interval=300000,  
                        n_intervals=0
                    )
                ])
            ], className='header'),
            
            # Main content area
            html.Div([
                # Left panel - Maps and radar
                html.Div([
                    # Weather Map
                    html.Div([
                        html.H3("Weather Radar Overlay"),
                        dcc.Graph(id='weather-map'),
                        dcc.RadioItems(
                            id='map-layer-selector',
                            options=[
                                {'label': 'Radar', 'value': 'radar'},
                                {'label': 'Satellite', 'value': 'satellite'},
                                {'label': 'Wind', 'value': 'wind'}
                            ],
                            value='radar'
                        )
                    ], className='map-container'),
                    
                    # Storm Movement Prediction
                    html.Div([
                        html.H3("Storm Movement Prediction"),
                        dcc.Graph(id='storm-movement')
                    ], className='storm-movement-container')
                ], className='left-panel'),
                
                # Right panel - Alerts and details
                html.Div([
                    # Risk Level Indicators
                    html.Div([
                        html.H3("Current Risk Levels"),
                        html.Div([
                            dcc.Graph(id='wind-risk-gauge'),
                            dcc.Graph(id='storm-risk-gauge')
                        ], className='risk-gauges')
                    ]),
                    
                    # Active Alerts
                    html.Div([
                        html.H3("Active Alerts"),
                        html.Div(id='alerts-panel', className='alerts-container')
                    ]),
                    
                    # Detailed Metrics
                    html.Div([
                        html.H3("Key Weather Metrics"),
                        dcc.Tabs([
                            dcc.Tab(label='Wind', children=[
                                dcc.Graph(id='wind-detail-chart')
                            ]),
                            dcc.Tab(label='Pressure', children=[
                                dcc.Graph(id='pressure-detail-chart')
                            ]),
                            dcc.Tab(label='Visibility', children=[
                                dcc.Graph(id='visibility-detail-chart')
                            ])
                        ])
                    ])
                ], className='right-panel')
            ], className='main-content'),
            
            # Footer with forecast explanation
            html.Div([
                html.H3("Forecast Analysis"),
                html.Div(id='forecast-explanation', className='forecast-explanation')
            ], className='footer')
        ], className='dashboard-container')
        
    def setup_callbacks(self):
        """
        Set up all dashboard callbacks
        """
        @self.app.callback(
            [Output('wind-risk-gauge', 'figure'),
             Output('storm-risk-gauge', 'figure')],
            Input('main-update', 'n_intervals')
        )
        def update_risk_indicators(n):
            try:
                # Example risk levels
                wind_risk = 65
                storm_risk = 45
                
                def create_gauge(value, title):
                    color = "green" if value < 33 else "yellow" if value < 66 else "red"
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': title},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 33], 'color': "green"},
                                {'range': [33, 66], 'color': "yellow"},
                                {'range': [66, 100], 'color': "red"}
                            ]
                        }
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    return fig
                
                return [
                    create_gauge(wind_risk, "Wind Risk"),
                    create_gauge(storm_risk, "Storm Risk")
                ]
            except Exception as e:
                print(f"Error in update_risk_indicators: {str(e)}")
                # Return empty figures on error
                return [go.Figure(), go.Figure()]
            
        @self.app.callback(
            Output('weather-map', 'figure'),
            [Input('main-update', 'n_intervals'),
             Input('map-layer-selector', 'value')]
        )
        def update_weather_map(n, layer):
            try:
                # Example map with VAAH airfield location (Ahmedabad, India)
                fig = go.Figure()
                
                # Add base marker for the airfield
                fig.add_trace(go.Scattermapbox(
                    lat=[23.0667],
                    lon=[72.6167],
                    mode='markers+text',
                    marker=dict(size=14, color='red'),
                    text=['VAAH'],
                    textposition="top right",
                    name='Airfield'
                ))
                
                # Add layer-specific data
                if layer == 'radar':
                    # Example radar data (circular pattern around airfield)
                    radius = 0.1  # Roughly 10km
                    theta = np.linspace(0, 2*np.pi, 50)
                    lats = 23.0667 + radius * np.cos(theta)
                    lons = 72.6167 + radius * np.sin(theta)
                    fig.add_trace(go.Scattermapbox(
                        lat=lats.tolist(),
                        lon=lons.tolist(),
                        mode='lines',
                        line=dict(width=2, color='blue'),
                        name='Radar'
                    ))
                
                elif layer == 'satellite':
                    # Example satellite cloud cover (random points)
                    np.random.seed(0)
                    num_points = 50
                    lats = np.random.uniform(22.9667, 23.1667, num_points)
                    lons = np.random.uniform(72.5167, 72.7167, num_points)
                    fig.add_trace(go.Scattermapbox(
                        lat=lats.tolist(),
                        lon=lons.tolist(),
                        mode='markers',
                        marker=dict(size=8, color='white'),
                        name='Clouds'
                    ))
                
                elif layer == 'wind':
                    # Example wind arrows
                    arrow_lats = [23.0667 + 0.05, 23.0667 - 0.05]
                    arrow_lons = [72.6167 + 0.05, 72.6167 - 0.05]
                    fig.add_trace(go.Scattermapbox(
                        lat=arrow_lats,
                        lon=arrow_lons,
                        mode='lines+markers',
                        line=dict(width=2, color='green'),
                        marker=dict(size=10, symbol='arrow', angle=45),
                        name='Wind'
                    ))
                
                fig.update_layout(
                    mapbox=dict(
                        style='open-street-map',
                        center=dict(lat=23.0667, lon=72.6167),
                        zoom=11
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=400,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255,255,255,0.8)'
                    )
                )
                return fig
            except Exception as e:
                print(f"Error in update_weather_map: {str(e)}")
                # Return empty figure on error
                return go.Figure()
            
        @self.app.callback(
            [Output('wind-detail-chart', 'figure'),
             Output('pressure-detail-chart', 'figure'),
             Output('visibility-detail-chart', 'figure')],
            Input('main-update', 'n_intervals')
        )
        def update_detail_charts(n):
            try:
                # Example time series data
                times = [f"{i:02d}:00" for i in range(24)]
                wind_speeds = np.random.normal(15, 5, 24).tolist()
                pressure_values = np.random.normal(1013, 2, 24).tolist()
                visibility_values = np.random.normal(10, 2, 24).tolist()
                
                def create_timeseries(y_values, title, y_label):
                    fig = go.Figure(data=go.Scatter(
                        x=times,
                        y=y_values,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                    fig.update_layout(
                        title=dict(
                            text=title,
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title='Time (UTC)',
                        yaxis_title=y_label,
                        height=250,
                        margin=dict(l=50, r=20, t=40, b=30),
                        paper_bgcolor='white',
                        plot_bgcolor='rgba(240,240,240,0.5)',
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                            tickangle=45,
                            nticks=12
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)'
                        )
                    )
                    return fig
                
                return [
                    create_timeseries(wind_speeds, "Wind Speed", "km/h"),
                    create_timeseries(pressure_values, "Pressure", "hPa"),
                    create_timeseries(visibility_values, "Visibility", "km")
                ]
            except Exception as e:
                print(f"Error in update_detail_charts: {str(e)}")
                # Return empty figures on error
                return [go.Figure(), go.Figure(), go.Figure()]
            
        @self.app.callback(
            Output('alerts-panel', 'children'),
            Input('main-update', 'n_intervals')
        )
        def update_alerts(n):
            # Example alerts
            alerts = [
                {"level": "warning", "text": "Strong winds expected in next 2 hours"},
                {"level": "info", "text": "Visibility reducing due to haze"}
            ]
            
            return [
                html.Div(
                    alert["text"],
                    className=f'alert alert-{alert["level"]}'
                ) for alert in alerts
            ]
            
        @self.app.callback(
            Output('forecast-explanation', 'children'),
            Input('main-update', 'n_intervals')
        )
        def update_forecast_explanation(n):
            # Example forecast explanation
            return html.P([
                "Current Analysis: ",
                html.Br(),
                "• Moderate wind conditions with speeds around 15-20 km/h",
                html.Br(),
                "• Good visibility above 8 km",
                html.Br(),
                "• No significant weather threats in the next 3 hours"
            ])
            
    def update_data(self, predictions: dict, alerts: list):
        """
        Update dashboard with new predictions and alerts
        """
        self.current_predictions = predictions
        self.current_alerts = alerts
        self.last_update = datetime.now()
    
    def run_server(self, debug=True):
        """
        Start the dashboard server
        """
        self.app.run(debug=debug)
