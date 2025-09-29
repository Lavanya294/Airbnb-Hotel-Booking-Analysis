import dash
from dash import dcc, html
import dash.dependencies as dd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================
# Load & Clean Data
# =========================
df = pd.read_csv("Airbnb_Data.csv", low_memory=False)

# Clean price
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df = df[df['price'] < 1000]  # remove extreme outliers

# Convert numeric columns
df['minimum nights'] = pd.to_numeric(df['minimum nights'], errors='coerce')
df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
df['availability 365'] = pd.to_numeric(df['availability 365'], errors='coerce')

# Fix spelling in neighbourhood group
df['neighbourhood group'] = df['neighbourhood group'].str.title()
df['neighbourhood group'] = df['neighbourhood group'].replace({
    "Brookln": "Brooklyn",
    "Manhatan": "Manhattan"
})

# =========================
# Styles
# =========================
COLORS = {
    'background': '#f8f9fa',
    'card': '#ffffff',
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'text': '#2c3e50',
    'border': '#dee2e6'
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'padding': '20px',
    'borderRadius': '10px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
    'marginBottom': '20px'
}

HEADER_STYLE = {
    'textAlign': 'center',
    'color': COLORS['primary'],
    'marginBottom': '30px',
    'padding': '20px',
    'backgroundColor': COLORS['card'],
    'borderRadius': '10px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
}

# =========================
# Initialize App
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Airbnb Analytics Dashboard"

# =========================
# Layout
# =========================
app.layout = html.Div([
    html.Div([
        html.H1("🏠 Airbnb Analytics Dashboard", style=HEADER_STYLE),
    ]),

    dcc.Tabs(id="tabs", value='tab1', 
             style={'marginBottom': '20px'},
             children=[
        dcc.Tab(label='📊 Overview', value='tab1'),
        dcc.Tab(label='💰 Pricing Analysis', value='tab2'),
        dcc.Tab(label='👥 Host Insights', value='tab3'),
        dcc.Tab(label='📅 Availability', value='tab4'),
        dcc.Tab(label='🔮 Price Predictor', value='tab5'),
    ]),

    html.Div(id='tabs-content')
], style={'padding': '20px', 'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# =========================
# Tab Functions
# =========================

def tab_overview():
    return html.Div([
        html.Div([
            html.H2("Overview Dashboard", style={'color': COLORS['primary']}),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Neighbourhood Group:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="overview_neighbourhood",
                        options=[{'label': 'All', 'value': 'All'}] + 
                                [{"label": i, "value": i} for i in sorted(df['neighbourhood group'].dropna().unique())],
                        value="All",
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Room Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="overview_room_type",
                        options=[{'label': 'All', 'value': 'All'}] + 
                                [{"label": i, "value": i} for i in df['room type'].dropna().unique()],
                        value="All",
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            
            # Stats Cards
            html.Div(id='overview_stats', style={'marginBottom': '20px'}),
            
            # Graphs
            html.Div([
                dcc.Graph(id='overview_neighbourhood_chart', style={'marginBottom': '20px'}),
                dcc.Graph(id='overview_room_chart')
            ])
        ], style=CARD_STYLE)
    ])

def tab_pricing():
    return html.Div([
        html.Div([
            html.H2("Pricing Analysis", style={'color': COLORS['primary']}),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Neighbourhood Group:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="price_dropdown",
                        options=[{"label": i, "value": i} for i in sorted(df['neighbourhood group'].dropna().unique())],
                        value=df['neighbourhood group'].dropna().iloc[0],
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Room Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="price_room_type",
                        options=[{'label': 'All', 'value': 'All'}] + 
                                [{"label": i, "value": i} for i in df['room type'].dropna().unique()],
                        value="All",
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            
            # Price Stats
            html.Div(id='price_stats', style={'marginBottom': '20px'}),
            
            dcc.Graph(id="price_hist", style={'marginBottom': '20px'}),
            dcc.Graph(id="map_price")
        ], style=CARD_STYLE)
    ])

def tab_hosts():
    return html.Div([
        html.Div([
            html.H2("Host Insights", style={'color': COLORS['primary']}),
            
            # Filter
            html.Div([
                html.Label("Neighbourhood Group:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id="host_neighbourhood",
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{"label": i, "value": i} for i in sorted(df['neighbourhood group'].dropna().unique())],
                    value="All",
                    clearable=False
                )
            ], style={'width': '48%', 'marginBottom': '20px'}),
            
            dcc.Graph(id='host_chart', style={'marginBottom': '20px'}),
            dcc.Graph(id='host_scatter')
        ], style=CARD_STYLE)
    ])

def tab_availability():
    return html.Div([
        html.Div([
            html.H2("Availability Analysis", style={'color': COLORS['primary']}),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Neighbourhood Group:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="avail_neighbourhood",
                        options=[{'label': 'All', 'value': 'All'}] + 
                                [{"label": i, "value": i} for i in sorted(df['neighbourhood group'].dropna().unique())],
                        value="All",
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Room Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="avail_room_type",
                        options=[{'label': 'All', 'value': 'All'}] + 
                                [{"label": i, "value": i} for i in df['room type'].dropna().unique()],
                        value="All",
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            
            dcc.Graph(id='avail_hist', style={'marginBottom': '20px'}),
            dcc.Graph(id='avail_box')
        ], style=CARD_STYLE)
    ])

def tab_predictor():
    return html.Div([
        html.Div([
            html.H2("Price Predictor with Regression", style={'color': COLORS['primary']}),
            html.P("Predict Airbnb prices based on number of reviews using linear regression"),
            
            # Input Section
            html.Div([
                html.Div([
                    html.Label("Neighbourhood Group:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="pred_neighbourhood",
                        options=[{"label": i, "value": i} for i in sorted(df['neighbourhood group'].dropna().unique())],
                        value=df['neighbourhood group'].dropna().iloc[0],
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Number of Reviews:", style={'fontWeight': 'bold'}),
                    dcc.Input(
                        id='pred_reviews_input',
                        type='number',
                        value=50,
                        min=0,
                        style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': f'1px solid {COLORS["border"]}'}
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            
            # Prediction Result
            html.Div(id='prediction_result', style={'marginBottom': '20px'}),
            
            # Regression Chart
            dcc.Graph(id='regression_chart')
        ], style=CARD_STYLE)
    ])

# =========================
# Helper Function
# =========================
def filter_data(data, neighbourhood, room_type=None):
    dff = data.copy()
    if neighbourhood != 'All':
        dff = dff[dff['neighbourhood group'] == neighbourhood]
    if room_type and room_type != 'All':
        dff = dff[dff['room type'] == room_type]
    return dff

def create_stat_card(title, value, icon="📊"):
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '30px', 'marginRight': '15px'}),
            html.Div([
                html.P(title, style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'}),
                html.H3(value, style={'margin': '0', 'color': COLORS['primary']})
            ])
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'backgroundColor': COLORS['background'],
        'padding': '15px',
        'borderRadius': '8px',
        'flex': '1',
        'marginRight': '10px'
    })

# =========================
# Callback for Tabs
# =========================
@app.callback(dd.Output('tabs-content', 'children'),
              dd.Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab1':
        return tab_overview()
    elif tab == 'tab2':
        return tab_pricing()
    elif tab == 'tab3':
        return tab_hosts()
    elif tab == 'tab4':
        return tab_availability()
    elif tab == 'tab5':
        return tab_predictor()

# =========================
# Overview Tab Callbacks
# =========================
@app.callback(
    [dd.Output('overview_stats', 'children'),
     dd.Output('overview_neighbourhood_chart', 'figure'),
     dd.Output('overview_room_chart', 'figure')],
    [dd.Input('overview_neighbourhood', 'value'),
     dd.Input('overview_room_type', 'value')]
)
def update_overview(neighbourhood, room_type):
    dff = filter_data(df, neighbourhood, room_type)
    
    # Stats
    stats = html.Div([
        create_stat_card("Total Listings", f"{len(dff):,}", "🏠"),
        create_stat_card("Avg Price", f"${dff['price'].mean():.2f}", "💰"),
        create_stat_card("Avg Reviews", f"{dff['number of reviews'].mean():.0f}", "⭐"),
        create_stat_card("Avg Availability", f"{dff['availability 365'].mean():.0f} days", "📅")
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
    
    # Charts
    fig1 = px.histogram(
        dff, x="neighbourhood group",
        title="Listings by Neighbourhood Group",
        color="neighbourhood group",
        labels={"neighbourhood group": "Neighbourhood"}
    )
    fig1.update_layout(showlegend=False, plot_bgcolor='white')
    
    fig2 = px.pie(
        dff, names="room type",
        title="Distribution of Room Types",
        hole=0.4
    )
    
    return stats, fig1, fig2

# =========================
# Pricing Tab Callbacks
# =========================
@app.callback(
    [dd.Output('price_stats', 'children'),
     dd.Output("price_hist", "figure"),
     dd.Output("map_price", "figure")],
    [dd.Input("price_dropdown", "value"),
     dd.Input("price_room_type", "value")]
)
def update_price_tab(neighbourhood, room_type):
    dff = filter_data(df, neighbourhood, room_type)
    dff_clean = dff.dropna(subset=['lat', 'long', 'price'])
    
    # Stats
    stats = html.Div([
        create_stat_card("Listings", f"{len(dff):,}", "🏠"),
        create_stat_card("Min Price", f"${dff['price'].min():.2f}", "💵"),
        create_stat_card("Avg Price", f"${dff['price'].mean():.2f}", "💰"),
        create_stat_card("Max Price", f"${dff['price'].max():.2f}", "💎")
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
    
    # Histogram
    fig1 = px.histogram(
        dff.dropna(subset=['price']), x="price", nbins=50,
        title=f"Price Distribution in {neighbourhood}",
        labels={"price": "Price ($)"}
    )
    fig1.update_layout(plot_bgcolor='white')
    
    # Map
    if len(dff_clean) > 0:
        dff_clean['size_val'] = dff_clean['minimum nights'].fillna(1).abs().clip(lower=1)
        fig2 = px.scatter_mapbox(
            dff_clean, lat="lat", lon="long", color="price",
            size="size_val", hover_name="NAME",
            hover_data={'lat': False, 'long': False, 'size_val': False, 'price': True, 'minimum nights': True},
            mapbox_style="carto-positron", zoom=10,
            title=f"Map of Airbnb Listings in {neighbourhood}",
            color_continuous_scale="Viridis"
        )
    else:
        fig2 = px.scatter_mapbox(
            mapbox_style="carto-positron", zoom=10,
            title=f"No location data available for {neighbourhood}"
        )
    
    return stats, fig1, fig2

# =========================
# Hosts Tab Callbacks
# =========================
@app.callback(
    [dd.Output('host_chart', 'figure'),
     dd.Output('host_scatter', 'figure')],
    [dd.Input('host_neighbourhood', 'value')]
)
def update_hosts(neighbourhood):
    dff = filter_data(df, neighbourhood)
    
    # Top hosts
    host_counts = dff['host id'].value_counts().head(10).reset_index()
    host_counts.columns = ['host id', 'listings']
    
    fig1 = px.bar(
        host_counts, x="host id", y="listings",
        title="Top 10 Hosts by Number of Listings",
        labels={"host id": "Host ID", "listings": "Listings"}
    )
    fig1.update_layout(plot_bgcolor='white')
    
    # Scatter
    fig2 = px.scatter(
        dff, x="number of reviews", y="price",
        title="Price vs. Number of Reviews",
        color="neighbourhood group", opacity=0.6,
        trendline="ols"
    )
    fig2.update_layout(plot_bgcolor='white')
    
    return fig1, fig2

# =========================
# Availability Tab Callbacks
# =========================
@app.callback(
    [dd.Output('avail_hist', 'figure'),
     dd.Output('avail_box', 'figure')],
    [dd.Input('avail_neighbourhood', 'value'),
     dd.Input('avail_room_type', 'value')]
)
def update_availability(neighbourhood, room_type):
    dff = filter_data(df, neighbourhood, room_type)
    
    fig1 = px.histogram(
        dff, x="availability 365", nbins=50,
        title="Availability Distribution (Days in a Year)",
        color="neighbourhood group",
        labels={"availability 365": "Available Days"}
    )
    fig1.update_layout(plot_bgcolor='white')
    
    fig2 = px.box(
        dff, x="neighbourhood group", y="availability 365",
        title="Availability by Neighbourhood Group",
        color="neighbourhood group"
    )
    fig2.update_layout(showlegend=False, plot_bgcolor='white')
    
    return fig1, fig2

# =========================
# Price Predictor Callbacks
# =========================
@app.callback(
    [dd.Output('prediction_result', 'children'),
     dd.Output('regression_chart', 'figure')],
    [dd.Input('pred_neighbourhood', 'value'),
     dd.Input('pred_reviews_input', 'value')]
)
def update_predictor(neighbourhood, reviews_input):
    dff = df[df['neighbourhood group'] == neighbourhood].copy()
    dff_clean = dff.dropna(subset=['number of reviews', 'price'])
    
    if len(dff_clean) < 2:
        return html.Div("Not enough data for prediction"), {}
    
    # Train model
    X = dff_clean[['number of reviews']].values
    y = dff_clean['price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict
    predicted_price = model.predict([[reviews_input]])[0]
    r_squared = model.score(X, y)
    
    # Result card
    result = html.Div([
        html.Div([
            html.Div([
                html.H3("Predicted Price", style={'color': COLORS['secondary'], 'margin': '0'}),
                html.H1(f"${predicted_price:.2f}", style={'color': COLORS['accent'], 'margin': '10px 0'}),
                html.P(f"For {reviews_input} reviews in {neighbourhood}", style={'color': '#7f8c8d', 'margin': '0'}),
                html.P(f"R² Score: {r_squared:.3f}", style={'color': '#7f8c8d', 'margin': '5px 0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': COLORS['background'], 'borderRadius': '8px'})
        ])
    ])
    
    # Regression chart
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=dff_clean['number of reviews'],
        y=dff_clean['price'],
        mode='markers',
        name='Actual Listings',
        marker=dict(color='lightblue', size=8, opacity=0.6)
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=x_range.flatten(),
        y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=3)
    ))
    
    # Prediction point
    fig.add_trace(go.Scatter(
        x=[reviews_input],
        y=[predicted_price],
        mode='markers',
        name='Your Prediction',
        marker=dict(color='green', size=15, symbol='star')
    ))
    
    fig.update_layout(
        title=f"Price vs Reviews Regression for {neighbourhood}",
        xaxis_title="Number of Reviews",
        yaxis_title="Price ($)",
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    return result, fig

# =========================
# Run App
# =========================
# Expose server for deployment
server = app.server

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Use Render PORT or fallback to 8050 locally
    app.run(host='0.0.0.0', port=port, debug=True)