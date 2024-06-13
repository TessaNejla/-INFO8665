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
#Function to Display Stock Analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")

    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        st.plotly_chart(fig)
        
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages',
                          xaxis_title='Date',
                          yaxis_title='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        st.plotly_chart(fig)
        
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig)
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        st.plotly_chart(fig)
#Function to Display Additional Information
        