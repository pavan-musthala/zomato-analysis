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
    
    # Standardize restaurant types
    rest_type_mapping = {
        # Cafes and Bakeries
        'Cafe, Bakery': 'Cafe & Bakery',
        'Bakery, Cafe': 'Cafe & Bakery',
        'Cafe, Quick Bites': 'Cafe',
        'Cafe, Casual Dining': 'Cafe',
        'Cafe, Dessert Parlor': 'Cafe',
        'Cafe, Lounge': 'Cafe & Lounge',
        'Bakery, Quick Bites': 'Bakery',
        'Bakery, Dessert Parlor': 'Bakery & Desserts',
        
        # Bars and Pubs
        'Bar, Lounge': 'Bar & Lounge',
        'Bar, Quick Bites': 'Bar',
        'Lounge, Bar': 'Bar & Lounge',
        'Pub, Cafe': 'Pub',
        'Pub, Casual Dining': 'Pub',
        'Bar, Casual Dining': 'Bar & Dining',
        'Bar, Pub': 'Bar & Pub',
        'Pub, Bar': 'Bar & Pub',
        
        # Microbreweries
        'Microbrewery, Bar': 'Microbrewery',
        'Microbrewery, Casual Dining': 'Microbrewery',
        'Microbrewery, Lounge': 'Microbrewery',
        'Microbrewery, Pub': 'Microbrewery',
        'Casual Dining, Microbrewery': 'Microbrewery',
        
        # Casual Dining
        'Casual Dining, Cafe': 'Casual Dining',
        'Casual Dining, Lounge': 'Casual Dining & Lounge',
        'Casual Dining, Pub': 'Casual Dining',
        'Casual Dining, Irani Cafee': 'Casual Dining',
        'Lounge, Casual Dining': 'Casual Dining & Lounge',
        
        # Fine Dining
        'Fine Dining, Bar': 'Fine Dining',
        'Fine Dining, Lounge': 'Fine Dining',
        'Fine Dining, Microbrewery': 'Fine Dining',
        
        # Quick Service
        'Quick Bites, Bakery': 'Quick Bites',
        'Quick Bites, Beverage Shop': 'Quick Bites',
        'Quick Bites, Food Court': 'Quick Bites',
        
        # Beverage Shops
        'Beverage Shop, Cafe': 'Beverage Shop',
        'Beverage Shop, Dessert Parlor': 'Beverage Shop',
        'Beverage Shop, Quick Bites': 'Beverage Shop',
        
        # Dessert Places
        'Dessert Parlor, Bakery': 'Dessert Parlor',
        'Dessert Parlor, Beverage Shop': 'Dessert Parlor',
        'Dessert Parlor, Cafe': 'Dessert Parlor',
        'Dessert Parlor, Kiosk': 'Dessert Parlor',
        'Dessert Parlor, Quick Bites': 'Dessert Parlor',
        'Dessert Parlour': 'Dessert Parlor',
        
        # Food Courts
        'Food Court, Casual Dining': 'Food Court',
        'Food Court, Quick Bites': 'Food Court',
        
        # Clubs and Lounges
        'Club, Casual Dining': 'Club',
        'Lounge, Cafe': 'Lounge & Cafe',
        'Lounge, Microbrewery': 'Lounge',
        
        # Delivery and Takeaway
        'Takeaway, Delivery': 'Takeaway',
        'Delivery': 'Takeaway'
    }
    
    df['rest_type'] = df['rest_type'].replace(rest_type_mapping)
    
    # Drop rows with NaN values after conversion
    df = df.dropna(subset=['rate', 'approx_cost(for two people)'])
    
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
locations = ['All'] + sorted(df['location'].unique().tolist())
selected_location = st.sidebar.selectbox('Select Location', locations)

# Restaurant type filter
rest_types = ['All'] + sorted(df['rest_type'].unique().tolist())
selected_rest_type = st.sidebar.selectbox('Select Restaurant Type', rest_types)

# Price range filter
price_range = st.sidebar.slider(
    'Price Range (for two people)',
    int(df['approx_cost(for two people)'].min()),
    int(df['approx_cost(for two people)'].max()),
    (int(df['approx_cost(for two people)'].min()), int(df['approx_cost(for two people)'].max()))
)

# Filter data based on selections
filtered_df = df.copy()
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
if selected_rest_type != 'All':
    filtered_df = filtered_df[filtered_df['rest_type'] == selected_rest_type]
filtered_df = filtered_df[
    (filtered_df['approx_cost(for two people)'] >= price_range[0]) &
    (filtered_df['approx_cost(for two people)'] <= price_range[1])
]

# Check if filtered data is empty
if len(filtered_df) == 0:
    st.error(f"""
    ### No Data Available! 
    
    The current filter combination returns no results:
    - Location: {selected_location}
    - Restaurant Type: {selected_rest_type}
    - Price Range: ₹{price_range[0]} to ₹{price_range[1]}
    
    Please try different filter combinations.
    """)
    st.stop()

# Show filter summary
st.info(f"""
**Current Filter Settings:**
- Location: {selected_location}
- Restaurant Type: {selected_rest_type}
- Price Range: ₹{price_range[0]} to ₹{price_range[1]}
- Number of restaurants in this selection: {len(filtered_df):,}
""")

# Main dashboard
# Row 1 - Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Restaurants", len(filtered_df))
with col2:
    st.metric("Average Rating", f"{filtered_df['rate'].mean():.2f}")
with col3:
    st.metric("Average Cost for Two", f"₹{filtered_df['approx_cost(for two people)'].mean():.2f}")
with col4:
    st.metric("Online Order Available", 
              f"{(filtered_df['online_order'] == 'Yes').mean()*100:.1f}%")

st.markdown("---")

# Row 2 - Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Rating Distribution")
    fig = px.histogram(filtered_df, x='rate', nbins=20,
                      title='Distribution of Restaurant Ratings')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    avg_rating = filtered_df['rate'].mean()
    median_rating = filtered_df['rate'].median()
    most_common_rating = filtered_df['rate'].mode().iloc[0]
    st.markdown(f"""
    **Key Insights:**
    - Average Rating: {avg_rating:.2f}
    - Median Rating: {median_rating:.2f}
    - Most Common Rating: {most_common_rating:.2f}
    - Distribution shows how restaurants are rated by customers, with most falling in the middle range.
    """)

with col2:
    st.subheader("💰 Cost vs Rating")
    fig = px.scatter(filtered_df, x='approx_cost(for two people)', y='rate',
                    title='Cost vs Rating Correlation')
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation
    correlation = filtered_df['rate'].corr(filtered_df['approx_cost(for two people)'])
    st.markdown(f"""
    **Key Insights:**
    - Correlation Coefficient: {correlation:.2f}
    - {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} relationship between cost and rating
    - Higher cost doesn't necessarily guarantee better ratings
    """)

st.markdown("---")

# Row 3 - More Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏪 Top Restaurant Types")
    rest_type_counts = filtered_df['rest_type'].value_counts().head(10)
    fig = px.bar(rest_type_counts, title='Top 10 Restaurant Types')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary
    total_types = len(filtered_df['rest_type'].unique())
    top_type = rest_type_counts.index[0]
    top_type_percent = (rest_type_counts.iloc[0] / len(filtered_df)) * 100
    st.markdown(f"""
    **Key Insights:**
    - Total Restaurant Types: {total_types}
    - Most Common Type: {top_type}
    - {top_type} represents {top_type_percent:.1f}% of all restaurants
    - Shows the diversity of dining options available
    """)
    
with col2:
    st.subheader("💳 Online Ordering Analysis")
    online_order_counts = filtered_df['online_order'].value_counts()
    fig = px.pie(values=online_order_counts.values, 
                 names=online_order_counts.index,
                 title='Online Order Availability')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary
    online_percent = (online_order_counts.get('Yes', 0) / len(filtered_df)) * 100
    st.markdown(f"""
    **Key Insights:**
    - {online_percent:.1f}% of restaurants offer online ordering
    - {'Majority' if online_percent > 50 else 'Minority'} of restaurants have adopted online ordering
    - Shows the digital transformation in the restaurant industry
    """)

# Row 4 - Location Analysis
st.subheader("📍 Location Analysis")
col1, col2 = st.columns(2)

with col1:
    # Average rating by location
    location_ratings = filtered_df.groupby('location')['rate'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(location_ratings, title='Top 10 Locations by Average Rating')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary
    top_location = location_ratings.index[0]
    top_location_rating = location_ratings.iloc[0]
    st.markdown(f"""
    **Key Insights:**
    - Best Rated Area: {top_location} ({top_location_rating:.2f})
    - Shows geographical distribution of restaurant quality
    - Helps identify prime locations for dining
    """)

with col2:
    # Restaurant types by average cost
    rest_type_cost = filtered_df.groupby('rest_type').agg({
        'approx_cost(for two people)': ['mean', 'count']
    }).reset_index()
    rest_type_cost.columns = ['rest_type', 'avg_cost', 'count']
    rest_type_cost = rest_type_cost.sort_values('avg_cost', ascending=False).head(10)
    
    fig = px.bar(rest_type_cost, x='rest_type', y='avg_cost',
                 title='Top 10 Restaurant Types by Average Cost')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary
    most_expensive = rest_type_cost.iloc[0]
    avg_cost_overall = filtered_df['approx_cost(for two people)'].mean()
    st.markdown(f"""
    **Key Insights:**
    - Most Expensive Type: {most_expensive['rest_type']} (₹{most_expensive['avg_cost']:.2f})
    - Overall Average Cost: ₹{avg_cost_overall:.2f}
    - Shows price variations across different restaurant types
    """)

# Footer with overall insights
st.markdown("---")
st.subheader("📊 Overall Dashboard Insights")
st.markdown(f"""
- This dashboard analyzes {len(filtered_df)} restaurants across different locations
- Price range varies from ₹{filtered_df['approx_cost(for two people)'].min():.0f} to ₹{filtered_df['approx_cost(for two people)'].max():.0f} for two people
- {len(filtered_df['location'].unique())} unique locations are covered
- {len(filtered_df['rest_type'].unique())} different types of restaurants are analyzed
""")

# Additional insights
st.markdown("### 📊 Restaurant Type Distribution")
rest_type_dist = filtered_df['rest_type'].value_counts().head(10)
fig = px.pie(values=rest_type_dist.values,
             names=rest_type_dist.index,
             title='Top 10 Restaurant Types Distribution',
             hole=0.4)
fig.update_layout(
    showlegend=True,
    legend_title="Restaurant Type"
)
st.plotly_chart(fig, use_container_width=True)

# Add summary for restaurant type distribution
top_rest_type = rest_type_dist.index[0]
top_rest_count = rest_type_dist.values[0]
top_rest_percent = (top_rest_count / len(filtered_df)) * 100
st.markdown(f"""
**Key Insights - Restaurant Type Distribution:**
- Most common type: {top_rest_type} ({top_rest_count} restaurants)
- {top_rest_type} represents {top_rest_percent:.1f}% of all restaurants
- Top 10 types account for {(rest_type_dist.sum() / len(filtered_df) * 100):.1f}% of total restaurants
- Shows the diversity and concentration of restaurant types in the market
""")

# Row 5 - Restaurant Types Analysis
st.subheader("🏪 Restaurant Types Analysis")
col1, col2 = st.columns(2)

with col1:
    # Restaurant types by average rating
    rest_type_ratings = filtered_df.groupby('rest_type').agg({
        'rate': ['mean', 'count']
    }).reset_index()
    rest_type_ratings.columns = ['rest_type', 'avg_rating', 'count']
    rest_type_ratings = rest_type_ratings[rest_type_ratings['count'] > 10].sort_values('avg_rating', ascending=False)

    fig = px.bar(rest_type_ratings.head(10),
                x='rest_type',
                y='avg_rating',
                color='avg_rating',
                text=rest_type_ratings.head(10)['avg_rating'].round(2),
                title='Top 10 Restaurant Types by Average Rating',
                color_continuous_scale='RdYlGn')
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Restaurant Type",
        yaxis_title="Average Rating",
        xaxis_tickangle=45,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary for restaurant types by rating
    top_rated_type = rest_type_ratings.iloc[0]
    st.markdown(f"""
    **Key Insights - Restaurant Types by Rating:**
    - Highest rated type: {top_rated_type['rest_type']} (Rating: {top_rated_type['avg_rating']:.2f})
    - Average rating across all types: {filtered_df['rate'].mean():.2f}
    - {len(rest_type_ratings)} restaurant types have more than 10 reviews
    - Shows which restaurant types consistently perform better in terms of customer satisfaction
    """)

with col2:
    # Restaurant types by average cost
    rest_type_cost = filtered_df.groupby('rest_type').agg({
        'approx_cost(for two people)': ['mean', 'count']
    }).reset_index()
    rest_type_cost.columns = ['rest_type', 'avg_cost', 'count']
    rest_type_cost = rest_type_cost[rest_type_cost['count'] > 10].sort_values('avg_cost', ascending=False)

    fig = px.bar(rest_type_cost.head(10),
                x='rest_type',
                y='avg_cost',
                color='avg_cost',
                text=rest_type_cost.head(10)['avg_cost'].round(0),
                title='Top 10 Restaurant Types by Average Cost',
                color_continuous_scale='RdYlGn')
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Restaurant Type",
        yaxis_title="Average Cost for Two (₹)",
        xaxis_tickangle=45,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary for restaurant types by cost
    most_expensive = rest_type_cost.iloc[0]
    least_expensive = rest_type_cost.iloc[-1]
    st.markdown(f"""
    **Key Insights - Restaurant Types by Cost:**
    - Most expensive type: {most_expensive['rest_type']} (₹{most_expensive['avg_cost']:.0f})
    - Least expensive type: {least_expensive['rest_type']} (₹{least_expensive['avg_cost']:.0f})
    - Price difference: {most_expensive['avg_cost'] - least_expensive['avg_cost']:.0f}₹
    - Shows the price segmentation across different dining experiences
    """)

# Comprehensive Overall Summary
st.markdown("---")
st.markdown("## 📈 Comprehensive Analysis Summary")

# Calculate key metrics
total_restaurants = len(filtered_df)
avg_rating = filtered_df['rate'].mean()
avg_cost = filtered_df['approx_cost(for two people)'].mean()
online_ordering_pct = (filtered_df['online_order'] == 'Yes').mean() * 100
top_cuisine = filtered_df['cuisines'].str.split(',').explode().value_counts().index[0]
price_range = (filtered_df['approx_cost(for two people)'].min(), filtered_df['approx_cost(for two people)'].max())

st.markdown(f"""
### 🎯 Key Findings
1. **Market Size and Diversity**
   - Analyzed {total_restaurants:,} restaurants across {len(filtered_df['location'].unique())} locations
   - {len(filtered_df['rest_type'].unique())} distinct restaurant types
   - Most popular cuisine: {top_cuisine}

2. **Customer Experience**
   - Average rating: {avg_rating:.2f}/5.0
   - {online_ordering_pct:.1f}% restaurants offer online ordering
   - Price range: ₹{price_range[0]:.0f} to ₹{price_range[1]:.0f}
   - Average cost for two: ₹{avg_cost:.0f}

3. **Business Insights**
   - {top_rest_type} is the most common restaurant type ({top_rest_percent:.1f}% of market)
   - {most_expensive['rest_type']} restaurants command highest average prices
   - {top_location} has the highest average rating of {top_location_rating:.2f}

4. **Market Trends**
   - {'Strong' if online_ordering_pct > 70 else 'Growing' if online_ordering_pct > 40 else 'Limited'} adoption of online ordering
   - Price and rating show a {correlation:.2f} correlation
""")

# Footer
st.markdown("---")
st.markdown("### 📝 Dashboard Notes")
st.markdown("""
- Data is based on Zomato restaurant listings
- All ratings are on a 5-point scale
- Costs are in Indian Rupees (₹)
- Analysis considers only restaurants with sufficient data (>10 reviews)
- Dashboard is interactive - use filters to explore specific segments
""")
