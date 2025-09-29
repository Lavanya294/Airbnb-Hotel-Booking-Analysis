import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Airbnb Analytics Dashboard",
    page_icon="🏠",
    layout="wide"
)

# =========================
# Load & Clean Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Airbnb_Data.csv", low_memory=False)
    # ... (data cleaning code is the same as before) ...
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
    df = df[df['price'] < 1000]
    df['minimum nights'] = pd.to_numeric(df['minimum nights'], errors='coerce')
    df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
    df['availability 365'] = pd.to_numeric(df['availability 365'], errors='coerce')
    df['neighbourhood group'] = df['neighbourhood group'].str.title().replace({"Brookln": "Brooklyn", "Manhatan": "Manhattan"})
    return df

df = load_data()

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filter Options ⚙️")

neighbourhood = st.sidebar.selectbox(
    "Select Neighbourhood Group",
    options=['All'] + sorted(df['neighbourhood group'].dropna().unique()),
    help="Filter data by a specific neighbourhood group."
)

room_type = st.sidebar.selectbox(
    "Select Room Type",
    options=['All'] + list(df['room type'].dropna().unique()),
    help="Filter data by a specific room type."
)

# Filter data based on sidebar selection
dff = df.copy()
if neighbourhood != 'All':
    dff = dff[dff['neighbourhood group'] == neighbourhood]
if room_type != 'All':
    dff = dff[dff['room type'] == room_type]

# =========================
# Main App
# =========================
st.title("🏠 Airbnb Analytics Dashboard")
st.markdown("An interactive dashboard to explore Airbnb listings in NYC.")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "💰 Pricing Analysis", 
    "👥 Host Insights", 
    "📅 Availability", 
    "🔮 Price Predictor"
])

# =========================
# Tab 1: Overview
# =========================
with tab1:
    st.header("Dashboard Overview")

    # Display stats using st.metric in styled columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(dff):,}", help="Total number of listings based on current filters.")
    col2.metric("Average Price", f"${dff['price'].mean():.2f}", help="Average price per night.")
    col3.metric("Average Reviews", f"{dff['number of reviews'].mean():.0f}", help="Average number of reviews per listing.")
    col4.metric("Average Availability", f"{dff['availability 365'].mean():.0f} days", help="Average number of days a listing is available per year.")
    
    st.markdown("---")

    # Display charts
    chart1, chart2 = st.columns(2)
    with chart1:
        fig_neighbourhood = px.histogram(dff, x="neighbourhood group", title="Listings by Neighbourhood", color="neighbourhood group")
        fig_neighbourhood.update_layout(showlegend=False)
        st.plotly_chart(fig_neighbourhood, use_container_width=True)

    with chart2:
        fig_room_type = px.pie(dff, names="room type", title="Distribution of Room Types", hole=0.4)
        st.plotly_chart(fig_room_type, use_container_width=True)

# =========================
# Tab 2: Pricing Analysis
# =========================
with tab2:
    st.header("💰 Pricing Analysis")
    
    # Histogram
    st.plotly_chart(px.histogram(
        dff.dropna(subset=['price']), x="price", nbins=50,
        title=f"Price Distribution"
    ), use_container_width=True)
    
    # Map
    dff_map = dff.dropna(subset=['lat', 'long', 'price']).copy()
    if not dff_map.empty:
        dff_map['size_val'] = dff_map['minimum nights'].fillna(1).abs().clip(lower=1)
        fig_map = px.scatter_map(
            dff_map, lat="lat", lon="long", color="price",
            size="size_val", hover_name="NAME",
            hover_data={'lat': False, 'long': False, 'price': True, 'minimum nights': True},
            zoom=10, title="Map of Airbnb Listings", color_continuous_scale="Viridis"
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No location data available for the selected filters.")

# =========================
# Tab 3: Host Insights
# =========================
with tab3:
    st.header("👥 Host Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        host_counts = dff['host id'].value_counts().nlargest(10).reset_index()
        host_counts.columns = ['host id', 'listings']
        fig_top_hosts = px.bar(host_counts, x="host id", y="listings", title="Top 10 Hosts by Listings")
        st.plotly_chart(fig_top_hosts, use_container_width=True)
    
    with col2:
        fig_price_reviews = px.scatter(dff, x="number of reviews", y="price", title="Price vs. Number of Reviews", trendline="ols")
        st.plotly_chart(fig_price_reviews, use_container_width=True)

# =========================
# Tab 4: Availability
# =========================
with tab4:
    st.header("📅 Availability Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_avail_hist = px.histogram(dff, x="availability 365", nbins=50, title="Availability Distribution (Days/Year)")
        st.plotly_chart(fig_avail_hist, use_container_width=True)

    with col2:
        fig_avail_box = px.box(dff, x="neighbourhood group", y="availability 365", title="Availability by Neighbourhood", color="neighbourhood group")
        fig_avail_box.update_layout(showlegend=False)
        st.plotly_chart(fig_avail_box, use_container_width=True)

# =========================
# Tab 5: Price Predictor
# =========================
with tab5:
    st.header("🔮 Price Predictor")
    st.write("Predict prices based on reviews using a simple linear regression model for the selected neighbourhood.")
    
    # Use the globally filtered neighbourhood or allow override if 'All' is selected
    pred_neighbourhood_options = sorted(df['neighbourhood group'].dropna().unique())
    pred_neighbourhood = neighbourhood if neighbourhood != 'All' else st.selectbox("Select a Neighbourhood to Predict In", pred_neighbourhood_options)
    
   # New Code (fixed)
    pred_reviews = st.slider("Number of Reviews", 0, int(dff['number of reviews'].max()), 50, 10)
    dff_pred = df[df['neighbourhood group'] == pred_neighbourhood].dropna(subset=['number of reviews', 'price'])
    
    if len(dff_pred) < 10:
        st.error(f"Not enough data for '{pred_neighbourhood}' to make a reliable prediction.")
    else:
        # Train model and predict
        model = LinearRegression().fit(dff_pred[['number of reviews']], dff_pred['price'])
        predicted_price = model.predict([[pred_reviews]])[0]
        r_squared = model.score(dff_pred[['number of reviews']], dff_pred['price'])
        
        st.metric(
            label=f"Predicted Price in {pred_neighbourhood}",
            value=f"${predicted_price:.2f}",
            delta=f"R² Score: {r_squared:.2f}",
            delta_color="off"
        )

        # Regression chart
        fig_reg = go.Figure()
        x_range = np.linspace(0, dff_pred['number of reviews'].max(), 100).reshape(-1, 1)
        fig_reg.add_trace(go.Scatter(x=dff_pred['number of reviews'], y=dff_pred['price'], mode='markers', name='Actual Listings', marker=dict(opacity=0.5)))
        fig_reg.add_trace(go.Scatter(x=x_range.flatten(), y=model.predict(x_range), mode='lines', name='Regression Line', line=dict(color='red')))
        fig_reg.add_trace(go.Scatter(x=[pred_reviews], y=[predicted_price], mode='markers', name='Your Prediction', marker=dict(color='green', size=15, symbol='star')))
        fig_reg.update_layout(title=f"Price vs Reviews Regression", xaxis_title="Number of Reviews", yaxis_title="Price ($)")
        st.plotly_chart(fig_reg, use_container_width=True)