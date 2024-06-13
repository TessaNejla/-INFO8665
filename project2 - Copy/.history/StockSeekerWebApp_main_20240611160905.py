import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import panel as pn
from datetime import datetime

# Specify start and end dates as datetime.date objects
start_date = datetime(2023, 1, 1).date()
end_date = datetime(2024, 1, 1).date()

# Create DateRangePicker widget with specified value
date_range_picker = pn.widgets.DateRangePicker(
    name='Select Date Range',
    value=(start_date, end_date)  # Start and end dates as datetime objects
)
# Ensure the panel extension for matplotlib is loaded
pn.extension('matplotlib')

# Define the main template
template = pn.template.MaterialTemplate(title="Stock Market Analysis and Prediction Web App")

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Sidebar Widgets
ticker_selector = pn.widgets.MultiSelect(name='Select Stock Tickers', options=popular_tickers, size=len(popular_tickers))
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
        fig = plt.figure()
        plt.plot(stock_df.index, stock_df['Close'])
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'{selected_stock} Closing Prices')
        return fig
        
    # Add other analysis types here...

# Main function to execute when 'Analyze' button is clicked
def analyze_button_click(event):
    selected_stock = ticker_selector.value
    analysis_type = analysis_type_selector.value
    start_date, end_date = date_range_picker.value_as_datetime
    additional_info_selected = {key: widget.value for key, widget in additional_info.items()}
    
    analysis_result = display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
    
    # Display the stock analysis result
    template.main[0][0] = pn.Column(
        pn.pane.Markdown(f"### {selected_stock} - {analysis_type} Analysis"),
        pn.pane.Matplotlib(analysis_result),
        # Add additional information display here...
    )

# Event handler for 'Analyze' button click
analyze_button.on_click(analyze_button_click)

# Display the app
template.servable()