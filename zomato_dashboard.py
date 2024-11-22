import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Zomato Restaurant Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        width: 100%;
    }
    h1 {
        color: #ff4b4b;
    }
    h2 {
        color: #ff725c;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🍽️ Zomato Restaurant Analysis Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        # Try to load from the current directory first (for local development)
        df = pd.read_csv('zomato.csv')
    except FileNotFoundError:
        try:
            # If not found, try loading from the data directory (for deployment)
            df = pd.read_csv('data/zomato.csv')
        except FileNotFoundError:
            st.error("Could not find zomato.csv in either the current directory or the data directory.")
            st.stop()
    
    # Basic data cleaning
    df.dropna(inplace=True)
    
    # Convert rate to numeric, handling 'NEW' and other non-numeric values
    df['rate'] = df['rate'].apply(lambda x: str(x).split('/')[0] if isinstance(x, str) and '/' in str(x) else x)
    df['rate'] = pd.to_numeric(df['rate'].replace(['NEW', '-'], np.nan), errors='coerce')
    
    # Convert cost to numeric, handling commas
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'].astype(str).str.replace(',',''), errors='coerce')
    
    return df

# Load data with error handling
try:
    df = load_data()
    if df.empty:
        st.error("Error: Dataset is empty")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Location filter
locations = sorted(df['location'].unique())
selected_location = st.sidebar.multiselect(
    "Select Locations",
    locations,
    default=locations[:5]
)

# Cuisine filter
cuisines = sorted(df['cuisines'].unique())
selected_cuisines = st.sidebar.multiselect(
    "Select Cuisines",
    cuisines,
    default=cuisines[:5]
)

# Price range filter
price_range = st.sidebar.slider(
    "Price Range (for two people)",
    int(df['approx_cost(for two people)'].min()),
    int(df['approx_cost(for two people)'].max()),
    (0, 5000)
)

# Filter data
filtered_df = df[
    (df['location'].isin(selected_location)) &
    (df['cuisines'].isin(selected_cuisines)) &
    (df['approx_cost(for two people)'].between(price_range[0], price_range[1]))
]

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Restaurants", len(filtered_df))
with col2:
    avg_rating = filtered_df['rate'].mean()
    st.metric("Average Rating", f"{avg_rating:.1f}")
with col3:
    avg_cost = filtered_df['approx_cost(for two people)'].mean()
    st.metric("Average Cost for Two", f"₹{avg_cost:.0f}")
with col4:
    online_ordering = (filtered_df['online_order'] == 'Yes').mean() * 100
    st.metric("Online Ordering", f"{online_ordering:.1f}%")

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location Analysis")
    location_counts = filtered_df['location'].value_counts().head(10)
    fig = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        labels={'x': 'Location', 'y': 'Number of Restaurants'},
        title="Top 10 Locations by Number of Restaurants"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💳 Online Ordering Analysis")
    online_order_counts = filtered_df['online_order'].value_counts()
    fig = px.pie(
        values=online_order_counts.values,
        names=online_order_counts.index,
        title="Online Ordering Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Rating Analysis
st.subheader("⭐ Rating Distribution")
fig = px.histogram(
    filtered_df,
    x='rate',
    nbins=20,
    title="Distribution of Ratings"
)
st.plotly_chart(fig, use_container_width=True)

# Price vs Rating Analysis
st.subheader("💰 Price vs Rating Analysis")
fig = px.scatter(
    filtered_df,
    x='approx_cost(for two people)',
    y='rate',
    title="Price vs Rating Correlation",
    trendline="ols"
)
st.plotly_chart(fig, use_container_width=True)

# Calculate correlation
correlation = filtered_df['approx_cost(for two people)'].corr(filtered_df['rate'])

# Restaurant type analysis
rest_type_ratings = filtered_df.groupby('rest_type').agg({
    'rate': ['mean', 'count']
}).reset_index()
rest_type_ratings.columns = ['rest_type', 'avg_rating', 'count']
rest_type_ratings = rest_type_ratings[rest_type_ratings['count'] > 10].sort_values('avg_rating', ascending=False)

# Location analysis
location_ratings = filtered_df.groupby('location')['rate'].mean().sort_values(ascending=False)

# Get top cuisine and location
top_cuisine = filtered_df['cuisines'].mode().iloc[0] if not filtered_df.empty else "N/A"
top_location = filtered_df['location'].mode().iloc[0] if not filtered_df.empty else "N/A"

# Insights
st.markdown("## 📊 Key Insights")

st.markdown("""
### 📈 Statistical Analysis
   - Average rating across selected areas: {:.2f}
   - Most common cuisine type: {}
   - Price and rating correlation: {:.2f}
   - Diverse price points across restaurant types

### 💡 Recommendations
1. For Customers:
   - Best rated areas for dining: {}
   - Best value restaurants: {}

2. For Restaurant Owners:
   - Consider online ordering to stay competitive
   - Focus on quality as higher prices don't guarantee better ratings
   - Popular locations for new ventures: {}

3. For Investors:
   - High-potential segments: {}
   - Growing market for {} cuisine
   - Consider {} for new restaurant investments
""".format(
    avg_rating,
    top_cuisine,
    correlation,
    ', '.join(location_ratings.head(3).index) if not location_ratings.empty else "No data available",
    ', '.join(rest_type_ratings['rest_type'].head(3)) if not rest_type_ratings.empty else "No data available",
    ', '.join(filtered_df['location'].value_counts().head(3).index) if not filtered_df.empty else "No data available",
    ', '.join(rest_type_ratings.head(3)['rest_type']) if not rest_type_ratings.empty else "No data available",
    top_cuisine,
    top_location
))
