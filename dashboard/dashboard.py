import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px

# Streamlit Page Configuration
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

# Title of the Dashboard
st.title('Bike Sharing Data Analysis')

# Sidebar for user input
st.sidebar.header("User Input")
selected_view = st.sidebar.selectbox("Choose Analysis Type", ["Overview", "Seasonal and Weather Analysis", "Time Series Forecasting", "Clustering Analysis"])

# Load Dataset
@st.cache
def load_data():
    main_data = pd.read_csv('/path_to_your_data/main_data.csv')  # Use the file that has already been merged
    return main_data

main_data = load_data()

# Show Raw Data if selected
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader("Main Data")
    st.write(main_data.head())

# --- Page 1: Overview Analysis ---
if selected_view == "Overview":
    st.header('Overview Analysis')

    # **3.1 Distribution of Bike Usage**
    st.subheader('Distribution of Daily Bike Usage')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(main_data['cnt'], kde=True, color="skyblue", bins=30, ax=ax)
    ax.set_title('Distribution of Daily Bike Usage')
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # **3.2 Seasonal and Weather Effects**
    st.subheader('Bike Usage Based on Season and Weather')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='season', y='cnt', data=main_data, palette='Set2', ax=ax)
    ax.set_title('Bike Usage by Season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='weathersit', y='cnt', data=main_data, palette='Set3', ax=ax)
    ax.set_title('Bike Usage by Weather')
    ax.set_xlabel('Weather')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

# --- Page 2: Seasonal and Weather Analysis ---
elif selected_view == "Seasonal and Weather Analysis":
    st.header('Seasonal and Weather Analysis')

    # **3.2 Detailed Analysis of Bike Usage Based on Season**
    st.subheader('Detailed Bike Usage by Season')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='season', y='cnt', data=main_data, palette='Set2', ax=ax)
    ax.set_title('Seasonal Bike Usage')
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

    # **3.3 Impact of Weather on Bike Usage**
    st.subheader('Impact of Weather on Bike Usage')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='weathersit', y='cnt', data=main_data, palette='Set3', ax=ax)
    ax.set_title('Bike Usage Based on Weather Conditions')
    ax.set_xlabel('Weather')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

    # **Interactive Plot**
    st.subheader("Interactive Heatmap: Bike Usage by Season and Weather")
    season_weather_heatmap = main_data.pivot_table(values='cnt', index='season', columns='weathersit', aggfunc='sum')
    fig = px.imshow(season_weather_heatmap, labels={'x': 'Weather', 'y': 'Season', 'color': 'Total Users'}, title="Heatmap of Bike Usage by Season and Weather")
    st.plotly_chart(fig)

# --- Page 3: Time Series Forecasting ---
elif selected_view == "Time Series Forecasting":
    st.header('Time Series Forecasting for Bike Usage')

    st.subheader('Forecasting Future Bike Usage with ARIMA')
    main_data['dteday'] = pd.to_datetime(main_data['dteday'])
    main_data.set_index('dteday', inplace=True)
    daily_usage = main_data['cnt'].resample('D').sum()

    # Splitting data for training and testing
    train = daily_usage[:int(0.8 * len(daily_usage))]
    test = daily_usage[int(0.8 * len(daily_usage)):]

    # ARIMA model
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecasting
    forecast = model_fit.forecast(steps=len(test))

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(daily_usage.index, daily_usage, label='Historical Data')
    ax.plot(test.index, forecast, label='Forecast', color='orange')
    ax.set_title('Forecasted Bike Usage with ARIMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Users')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Interactive Line Chart with Plotly
    st.subheader('Interactive Time Series Forecasting')
    fig = px.line(daily_usage, x=daily_usage.index, y=daily_usage, labels={'y': 'Number of Users', 'x': 'Date'}, title="Daily Bike Usage")
    st.plotly_chart(fig)

# --- Page 4: Clustering Analysis ---
elif selected_view == "Clustering Analysis":
    st.header('Clustering Analysis of Bike Usage')

    st.subheader('Clustering Bike Usage Based on Features')
    features = main_data[['temp', 'season', 'weathersit', 'hr', 'cnt']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    main_data['cluster'] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='temp', y='cnt', hue='cluster', data=main_data, palette='Set2', s=100, alpha=0.7, ax=ax)
    ax.set_title('Clustering Bike Usage Based on Temperature and Users')
    ax.set_xlabel('Temperature (Celsius)')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

    # 3D Clustering Visualization
    st.subheader('3D Clustering Visualization')
    fig = px.scatter_3d(main_data, x='temp', y='cnt', z='hr', color='cluster', title="3D Clustering of Bike Usage", labels={'x': 'Temperature', 'y': 'Users', 'z': 'Hour'})
    st.plotly_chart(fig)

# --- Conclusion ---
st.subheader("Conclusion")
st.write("""
1. **Key Influencing Factors for Bike Usage**: Temperature and season play a major role in determining bike usage. Clear and sunny weather is a significant factor in increasing the number of users.
2. **Peak Bike Usage Times**: Bike usage shows peaks in the early morning and evening hours, coinciding with the commute times. Warmer weather and pleasant seasons also contribute to higher usage.
3. **Clustering Insights**: Clustering analysis reveals that bike usage patterns can be segmented based on temperature and the number of users, allowing for better understanding of user behavior.
4. **Forecasting Future Usage**: Using ARIMA, we can predict future bike usage, which helps in planning for bike availability and maintenance.
""")
