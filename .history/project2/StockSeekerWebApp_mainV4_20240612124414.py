import panel as pn
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

# Enable Panel extensions
pn.extension('plotly', 'tabulator')

# Load custom CSS for styling
css_file_path = r"C:\Users\Admin\Documents\MLAI\-INFO8665\project2\docs\assets\style.css"
pn.config.raw_css.append(open(css_file_path).read())

# Define widgets
tickers = ['AAPL', 'META', 'NVDA', 'NFLX']
ticker_select = pn.widgets.MultiSelect(name='Select Stock Tickers', options=tickers, size=4)

start_date_picker = pn.widgets.DatePicker(name='Start Date', value=datetime(2020, 1, 1))
end_date_picker = pn.widgets.DatePicker(name='End Date', value=datetime.now())

analysis_type_select = pn.widgets.Select(name='Select Analysis Type', options=[
    "Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

additional_info = pn.widgets.MultiChoice(name='Display Additional Information', options=[
    "Stock Actions", "Quarterly Financials", "Institutional Shareholders", "Quarterly Balance Sheet", "Quarterly Cashflow", "Analysts Recommendation", "Predicted Prices"])

# Buttons
analyze_button = pn.widgets.Button(name='Analyze', button_type='primary')
summary_button = pn.widgets.Button(name='OsTron', button_type='success')
oschero_button = pn.widgets.Button(name='OscHero', button_type='primary')
oschero_graph_select = pn.widgets.Select(name='Select Graph', options=[
    "Candlestick Chart with Pivots",
    "Chaikin Oscillator",
    "Stochastic Oscillator",
    "MACD",
    "RSI"
])

# Define the callback function for the OscHero button
def oschero_button_click(event):
    selected_graph = oschero_graph_select.value
    selected_stock = ticker_select.value[0]  # Assuming single selection for demo
    start_date = start_date_picker.value
    end_date = end_date_picker.value

    if selected_graph == "Candlestick Chart with Pivots":
        display_candlestick_with_pivots(selected_stock, start_date, end_date)
    elif selected_graph == "Chaikin Oscillator":
        display_chaikin_oscillator(selected_stock, start_date, end_date)
    elif selected_graph == "Stochastic Oscillator":
        display_stochastic_oscillator(selected_stock, start_date, end_date)
    elif selected_graph == "MACD":
        display_macd(selected_stock, start_date, end_date)
    elif selected_graph == "RSI":
        display_rsi(selected_stock, start_date, end_date)

# Assign the callback function to the OscHero button
oschero_button.on_click(oschero_button_click)

# Add the OscHero button and graph select dropdown to your main layout
main_layout = pn.Column(
    pn.Row(ticker_select, start_date_picker, end_date_picker, analysis_type_select, additional_info),
    pn.Row(analyze_button, summary_button, oschero_button, oschero_graph_select)
)

def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    output_pane = pn.Column()

    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        output_pane.append(fig)
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        output_pane.append(fig)
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages', xaxis_title='Date', yaxis_title='Price')
        output_pane.append(fig)
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        output_pane.append(fig)
    elif analysis_type == "Correlation Heatmap":
        df_selected_stock = stock_df['Close']
        corr = df_selected_stock.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        output_pane.append(fig)
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        output_pane.append(fig)
    
    return output_pane

def display_additional_information(selected_stock, selected_options):
    info_panes = []
    stock_data = yf.Ticker(selected_stock)
    
    for option in selected_options:
        if option == "Stock Actions":
            data = stock_data.actions
            title = "Stock Actions"
        elif option == "Quarterly Financials":
            data = stock_data.quarterly_financials
            title = "Quarterly Financials"
        elif option == "Institutional Shareholders":
            data = stock_data.institutional_holders
            title = "Institutional Shareholders"
        elif option == "Quarterly Balance Sheet":
            data = stock_data.quarterly_balance_sheet
            title = "Quarterly Balance Sheet"
        elif option == "Quarterly Cashflow":
            data = stock_data.quarterly_cashflow
            title = "Quarterly Cashflow"
        elif option == "Analysts Recommendation":
            data = stock_data.recommendations

        pane = pn.pane.DataFrame(data, name=title)
        info_panes.append(pane)
    
    return pn.Tabs(*[(pane.name, pane) for pane in info_panes])
