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
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

# Enable Panel extensions
pn.extension('plotly', 'tabulator')

# Define widgets
tickers = ['AAPL', 'META', 'NVDA', 'NFLX']
ticker_select = pn.widgets.MultiSelect(name='Select Stock Tickers', options=tickers, size=4)
start_date_picker = pn.widgets.DatePicker(name='Start Date', value=datetime(2020, 1, 1))
end_date_picker = pn.widgets.DatePicker(name='End Date', value=datetime.now())
analysis_type_select = pn.widgets.Select(name='Select Analysis Type', options=[
    "Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"
])
additional_info = pn.widgets.MultiChoice(name='Display Additional Information', options=[
    "Stock Actions", "Quarterly Financials", "Institutional Shareholders", "Quarterly Balance Sheet", 
    "Quarterly Cashflow", "Analysts Recommendation", "Predicted Prices"
])

# Buttons
analyze_button = pn.widgets.Button(name='Analyze', button_type='primary')
summary_button = pn.widgets.Button(name='OsTron', button_type='success')

# Functions
def fetch_stock_data(selected_stock, start_date, end_date):
    return yf.Ticker(selected_stock).history(start=start_date, end=end_date)

def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_df = fetch_stock_data(selected_stock, start_date, end_date)
    output_pane = pn.Column()

    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages', xaxis_title='Date', yaxis_title='Price')
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
    elif analysis_type == "Correlation Heatmap":
        df_selected_stock = stock_df['Close']
        corr = df_selected_stock.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
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
            title = "Analysts Recommendation"
        elif option == "Predicted Prices":
            info_panes.append(display_predicted_prices(selected_stock, start_date_picker.value, end_date_picker.value))
            continue
        
        if not data.empty:
            info_panes.append(pn.Column(f"## {selected_stock} - {title}", pn.widgets.DataFrame(data)))
        else:
            info_panes.append(pn.pane.Markdown(f"## {selected_stock} - {title}\n\nNo data available."))
    
    return pn.Column(*info_panes)

def display_predicted_prices(selected_stock, start_date, end_date):
    df = yf.download(selected_stock, start=start_date, end=end_date)
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
    fig.update_layout(title=f'{selected_stock} Predicted Prices', xaxis_title='Date', yaxis_title='Close Price')
    
    return pn.Column(fig)

# Main Layout
def main_layout():
    return pn.Column(
        pn.Row(ticker_select, start_date_picker, end_date_picker),
        pn.Row(analysis_type_select, analyze_button),
        pn.Row(additional_info, summary_button),
        pn.layout.Divider(),
        pn.pane.Markdown("## Analysis and Additional Information"),
        pn.Column()
    )

# Event Handling
def analyze_button_click(event):
    selected_stock = ticker_select.value[0] if ticker_select.value else tickers[0]
    analysis_type = analysis_type_select.value
    start_date = start_date_picker.value
    end_date = end_date_picker.value
    output_pane = display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
    layout[3] = output_pane

def summary_button_click(event):
    selected_stock = ticker_select.value[0] if ticker_select.value else tickers[0]
    selected_options = additional_info.value
    info_pane = display_additional_information(selected_stock, selected_options)
    layout[3] = info_pane

analyze_button.on_click(analyze_button_click)
summary_button.on_click(summary_button_click)

# Serve the layout
layout = main_layout()
layout.servable()
Final Notes: