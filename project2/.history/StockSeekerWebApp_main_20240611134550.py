import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import panel as pn

# Ensure the panel extension for matplotlib is loaded
pn.extension('matplotlib')

# Define the main template
template = pn.template.MaterialTemplate(title="Stock Market Analysis and Prediction Web App")

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Sidebar Widgets
ticker_selector = pn.widgets.MultiSelect(name='Select Stock Tickers', options=popular_tickers, size=len(popular_tickers))
date_range_picker = pn.widgets.DateRangePicker(
    name='Select Date Range',
    start=pd.Timestamp('2020-01-01'),
    end=pd.Timestamp.now()
)
analysis_type_selector = pn.widgets.Select(name='Select Analysis Type', options=["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])
additional_info = {
    "Stock Actions": pn.widgets.Checkbox(name="Stock Actions"),
    "Quarterly Financials": pn.widgets.Checkbox(name="Quarterly Financials"),
    "Institutional Shareholders": pn.widgets.Checkbox(name="Institutional Shareholders"),
    "Quarterly Balance Sheet": pn.widgets.Checkbox(name="Quarterly Balance Sheet"),
    "Quarterly Cashflow": pn.widgets.Checkbox(name="Quarterly Cashflow"),
    "Analysts Recommendation": pn.widgets.Checkbox(name="Analysts Recommendation"),
    "Predicted Prices": pn.widgets.Checkbox(name="Predicted Prices")  
}
analyze_button = pn.widgets.Button(name='Analyze', button_type='primary')

# Layout for Sidebar
template.sidebar.append(ticker_selector)
template.sidebar.append(date_range_picker)
template.sidebar.append(analysis_type_selector)
template.sidebar.append(pn.pane.Markdown("### Additional Information"))
for widget in additional_info.values():
    template.sidebar.append(widget)
template.sidebar.append(analyze_button)

# Function to display stock analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        return fig
        
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        return fig
        
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
        return fig
        
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        return fig
        
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        return fig
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        return fig

# Function to display additional information
def display_additional_information(selected_stock, selected_options):
    panes = []
    for option, widget in selected_options.items():
        if widget.value:
            if option == "Stock Actions":
                display_action = yf.Ticker(selected_stock).actions
                if not display_action.empty:
                    panes.append(pn.pane.DataFrame(display_action, name=f"{selected_stock} - {option}"))
                else:
                    panes.append(pn.pane.Markdown(f"No data available for {option}"))
            elif option == "Quarterly Financials":
                display_fin 