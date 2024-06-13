import panel as pn
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
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
oschero_button = pn.widgets.Button(name='OscHero', button_type='warning')

# Function to fetch stock data
def fetch_stock_data(selected_stock, start_date, end_date):
    return yf.Ticker(selected_stock).history(start=start_date, end=end_date)

# Function to display basic stock analysis
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
        # Need to update to compare multiple stocks if required
        fig = go.Figure()
        fig.update_layout(title='Correlation Heatmap - Function not yet implemented')
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
    
    output_pane.append(fig)
    return output_pane

# Function to display additional stock information
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

# Function to display predicted prices using LSTM model
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

# Helper function to calculate Chaikin Oscillator
def calculate_chaikin_oscillator(data):
    data['ADL'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
    data['Chaikin_Oscillator'] = ta.ema(data['ADL'], length=3) - ta.ema(data['ADL'], length=10)
    return data

# Helper function to calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, period=14):
    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max()
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H

2/2




contunue for response
ChatGPT
python
Copy code
14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

# Function to display Stochastic Oscillator and Price chart
def chart_stochastic_oscillator_and_price(ticker, df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ma50'], mode='lines', name='50-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ma200'], mode='lines', name='200-Day MA'))
    fig.update_layout(title=f'{ticker} Closing Price with Moving Averages', xaxis_title='Date', yaxis_title='Price')
    fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D'))
    fig.update_layout(yaxis2=dict(anchor='x', overlaying='y', side='right'))
    return fig

# Function to display advanced analysis
def display_advanced_analysis(selected_stock, start_date, end_date):
    stock_df = fetch_stock_data(selected_stock, start_date, end_date)
    
    # Calculate indicators
    stock_df = calculate_chaikin_oscillator(stock_df)
    stock_df = calculate_stochastic_oscillator(stock_df)
    
    # Plot Stochastic Oscillator and Price
    chart = chart_stochastic_oscillator_and_price(selected_stock, stock_df)
    
    # Display the chart
    pn.panel(chart).servable()

# Function to handle analysis type selection
def handle_analysis_type(event):
    analysis_type = analysis_type_select.value
    selected_stock = ticker_select.value[0]
    start_date = start_date_picker.value
    end_date = end_date_picker.value
    
    if event.obj.name == 'Analyze':
        output.clear()
        output.append(display_stock_analysis(selected_stock, analysis_type, start_date, end_date))
    elif event.obj.name == 'OsTron':
        output.clear()
        output.append(display_advanced_analysis(selected_stock, start_date, end_date))

# Set button callbacks
analyze_button.on_click(handle_analysis_type)
summary_button.on_click(handle_analysis_type)

# Layout
widgets = pn.WidgetBox(ticker_select, start_date_picker, end_date_picker, analysis_type_select, analyze_button, summary_button, oschero_button)
output = pn.Column()

# Display the app
app = pn.Row(widgets, output)
app.servable()