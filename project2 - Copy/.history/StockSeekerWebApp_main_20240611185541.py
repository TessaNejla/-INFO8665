#Import Required Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
#Suppress Warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#Load Custom CSS for Styling
# Function to load local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Provide the path to the style.css file
style_css_path = "path/to/your/style.css"  # Update the path as needed
local_css(style_css_path)
#Sidebar Setup
# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Stock tickers combo box
st.sidebar.subheader("STOCK SEEKER WEB APP")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

# Date range selection
start_date = st.sidebar.date_input("Start date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.now())

# Analysis type selection
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

# Additional information selection
st.sidebar.subheader("Display Additional Information")
selected_options = {
    "Stock Actions": st.sidebar.checkbox("Stock Actions"),
    "Quarterly Financials": st.sidebar.checkbox("Quarterly Financials"),
    "Institutional Shareholders": st.sidebar.checkbox("Institutional Shareholders"),
    "Quarterly Balance Sheet": st.sidebar.checkbox("Quarterly Balance Sheet"),
    "Quarterly Cashflow": st.sidebar.checkbox("Quarterly Cashflow"),
    "Analysts Recommendation": st.sidebar.checkbox("Analysts Recommendation"),
    "Predicted Prices": st.sidebar.checkbox("Predicted Prices")
}

# Buttons for analysis and summary
button_clicked = st.sidebar.button("Analyze")
summary_clicked = st.sidebar.button("OsTron")
