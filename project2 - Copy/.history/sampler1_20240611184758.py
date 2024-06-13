import panel as pn
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Function to fetch stock data
def fetch_stock_data(ticker, start, end):
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period='1d', start=start, end=end)
    return df

# Function to display stock analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_df = fetch_stock_data(selected_stock, start_date, end_date)
    if analysis_type == "Pivot Points":
        display_pivot_points(stock_df)
    elif analysis_type == "Chaikin Oscillator":
        display_chaikin_oscillator(stock_df)
    elif analysis_type == "Stochastic Oscillator":
        display_stochastic_oscillator(stock_df)

# Function to display pivot points (dummy example)
def display_pivot_points(stock_df):
    pn.pane.Markdown("## Pivot Points Analysis").servable()
    pn.pane.DataFrame(stock_df).servable()

# Function to display Chaikin Oscillator (dummy example)
def display_chaikin_oscillator(stock_df):
    pn.pane.Markdown("## Chaikin Oscillator Analysis").servable()
    pn.pane.DataFrame(stock_df).servable()

# Function to display Stochastic Oscillator (dummy example)
def display_stochastic_oscillator(stock_df):
    pn.pane.Markdown("## Stochastic Oscillator Analysis").servable()
    pn.pane.DataFrame(stock_df).servable()

# Setting up the Panel application layout
pn.extension()

# Widgets for user interaction
stock_selector = pn.widgets.MultiSelect(name="Select stock tickers...", options=popular_tickers)
analysis_selector = pn.widgets.Select(name="Select Analysis Type", options=["Pivot Points", "Chaikin Oscillator", "Stochastic Oscillator"])
start_date_picker = pn.widgets.DatePicker(name="Start Date", value=datetime(2020, 1, 1))
end_date_picker = pn.widgets.DatePicker(name="End Date", value=datetime.today())

# Placeholder for displaying analysis results
output_area = pn.Column()

# Callback function to update the analysis based on user input
def update_analysis(event):
    output_area.clear()  # Clear the previous output
    for ticker in stock_selector.value:
        display_stock_analysis(ticker, analysis_selector.value, start_date_picker.value, end_date_picker.value)

# Link the callback function to user inputs
stock_selector.param.watch(update_analysis, 'value')
analysis_selector.param.watch(update_analysis, 'value')
start_date_picker.param.watch(update_analysis, 'value')
end_date_picker.param.watch(update_analysis, 'value')

# Defining the layout template
app = pn.template.FastListTemplate(
    site="Stock Market Analysis and Prediction Web App",
    title="STOCK SEEKER WEB APP",
    sidebar=[stock_selector, analysis_selector, start_date_picker, end_date_picker],
    main=[output_area]
)

# Serve the Panel application
app.servable()