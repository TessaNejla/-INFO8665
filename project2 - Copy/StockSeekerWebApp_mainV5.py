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
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
# Ensure that the app's directory is on sys.path
import sys
sys.path.insert(0, '/home/your_username')  # Modify this path as per your setup

# Activate your virtual environment
activate_this = '/home/your_username/.virtualenvs/panel_env/bin/activate_this.py'
with open(activate_this) as f:
    exec(f.read(), {'__file__': activate_this})

# Import panel serve function
from panel.io.server import get_server

# Import your script
from dashboard import dashboard  # Assuming your script defines 'dashboard'

# Serve the panel dashboard
server = get_server(dashboard, port=8000, address='127.0.0.1', show=False)

# Start the server
server.start()

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
# Columns for each button output
analysis_column = pn.Column()
ostron_column = pn.Column()
oschero_column = pn.Column()
# Define the callback function for the OscHero button
# Define the callback function for the Analyze button
def analyze_button_click(event):
    selected_stock = ticker_select.value[0]  # Assuming single selection for demo
    start_date = start_date_picker.value
    end_date = end_date_picker.value

    analysis_pane = display_stock_analysis(selected_stock, analysis_type_select.value, start_date, end_date)
    analysis_column.clear()
    analysis_column.append(analysis_pane)

analyze_button.on_click(analyze_button_click)

# Define the callback function for the OsTron button
def ostron_button_click(event):
    selected_stock = ticker_select.value[0]  # Assuming single selection for demo
    start_date = start_date_picker.value
    end_date = end_date_picker.value

    ostron_column.clear()  # Clear previous content

    # Display additional information if selected
    if "Predicted Prices" in additional_info.value:
        lstm_pane = lstm_predict_stock_price(selected_stock, start_date, end_date)
        ostron_column.append(lstm_pane)
    else:
        additional_info_pane = display_additional_information(selected_stock, additional_info.value)
        ostron_column.append(additional_info_pane)

summary_button.on_click(ostron_button_click)

# Define the callback function for the OscHero button
def oschero_button_click(event):
    selected_graph = oschero_graph_select.value
    selected_stock = ticker_select.value[0]  # Assuming single selection for demo
    start_date = start_date_picker.value
    end_date = end_date_picker.value

    oschero_column.clear()

    if selected_graph == "Candlestick Chart with Pivots":
        oschero_column.append(display_candlestick_with_pivots(selected_stock, start_date, end_date))
    elif selected_graph == "Chaikin Oscillator":
        oschero_column.append(display_chaikin_oscillator(selected_stock, start_date, end_date))
    elif selected_graph == "Stochastic Oscillator":
        oschero_column.append(display_stochastic_oscillator(selected_stock, start_date, end_date))
    elif selected_graph == "MACD":
        oschero_column.append(display_macd(selected_stock, start_date, end_date))
    elif selected_graph == "RSI":
        oschero_column.append(display_rsi(selected_stock, start_date, end_date))

oschero_button.on_click(oschero_button_click)

# # Add the OscHero button and graph select dropdown to your main layout
# main_layout = pn.Row(
#     pn.Column(
#         ticker_select, start_date_picker, end_date_picker, analysis_type_select, additional_info, 
#         analyze_button, summary_button, oschero_button, oschero_graph_select
#     ),
#     pn.Column(
#         pn.pane.Markdown("### Analysis Column"),
#         analysis_column
#     ),
#     pn.Column(
#         pn.pane.Markdown("### OsTron Column"),
#         ostron_column
#     ),
#     pn.Column(
#         pn.pane.Markdown("### OscHero Column"),
#         oschero_column
#     )
# )

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

# Function to handle LSTM prediction
def lstm_predict_stock_price(ticker, start_date, end_date):
    # Fetch historical stock data
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period='1d', start=start_date, end=end_date)
    data = df.filter(['Close'])

    # Prepare data for LSTM
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    training_data_len = int(np.ceil(len(dataset) * .8))
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []

    look_back = 60
    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create testing data
    test_data = scaled_data[training_data_len - look_back:, :]
    x_test, y_test = [], dataset[training_data_len:, :]

    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the results
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    fig = plt.figure(figsize=(12, 6))
    plt.title(f'{ticker} LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    return fig

# Define the callback function for the OsTron button
# Define the callback function for the OsTron button
def ostron_button_click(event):
    selected_stock = ticker_select.value[0]  # Assuming single selection for demo
    start_date = start_date_picker.value
    end_date = end_date_picker.value

    analysis_pane = display_stock_analysis(selected_stock, analysis_type_select.value, start_date, end_date)
    additional_info_pane = display_additional_information(selected_stock, additional_info.value)
    
    # Call the LSTM prediction function
    lstm_predict_stock_price(selected_stock, start_date, end_date)
    
    main_layout.clear()
    main_layout.append(analysis_pane)
    main_layout.append(additional_info_pane)

# Assign the callback function to the OsTron button
summary_button.on_click(ostron_button_click)

# Functions for OscHero technical analysis

def display_candlestick_with_pivots(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)

    fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                         open=stock_df['Open'],
                                         high=stock_df['High'],
                                         low=stock_df['Low'],
                                         close=stock_df['Close'],
                                         name='Candlestick')])

    # Calculate Pivot Points
    stock_df['Pivot'] = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    stock_df['R1'] = 2 * stock_df['Pivot'] - stock_df['Low']
    stock_df['S1'] = 2 * stock_df['Pivot'] - stock_df['High']
    stock_df['R2'] = stock_df['Pivot'] + (stock_df['High'] - stock_df['Low'])
    stock_df['S2'] = stock_df['Pivot'] - (stock_df['High'] - stock_df['Low'])

    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Pivot'], mode='lines', name='Pivot'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['R1'], mode='lines', name='R1'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['S1'], mode='lines', name='S1'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['R2'], mode='lines', name='R2'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['S2'], mode='lines', name='S2'))

    fig.update_layout(title=f'{selected_stock} Candlestick Chart with Pivot Points', xaxis_title='Date', yaxis_title='Price')
    main_layout.append(fig)

def display_chaikin_oscillator(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)

    stock_df['ADL'] = ta.accdist(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'])
    stock_df['Chaikin'] = ta.chaikin_osc(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'])

    fig = go.Figure(data=[go.Scatter(x=stock_df.index, y=stock_df['Chaikin'], mode='lines', name='Chaikin Oscillator')])
    fig.update_layout(title=f'{selected_stock} Chaikin Oscillator', xaxis_title='Date', yaxis_title='Value')
    main_layout.append(fig)

def display_stochastic_oscillator(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    stock_df['%K'], stock_df['%D'] = ta.stoch(stock_df['High'], stock_df['Low'], stock_df['Close'])
    
    fig = go.Figure(data=[go.Scatter(x=stock_df.index, y=stock_df['%K'], mode='lines', name='%K'),
                          go.Scatter(x=stock_df.index, y=stock_df['%D'], mode='lines', name='%D')])
    fig.update_layout(title=f'{selected_stock} Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
    main_layout.append(fig)

def display_macd(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    stock_df['MACD'], stock_df['Signal'], stock_df['Hist'] = ta.macd(stock_df['Close'])
    
    fig = go.Figure(data=[go.Scatter(x=stock_df.index, y=stock_df['MACD'], mode='lines', name='MACD'),
                          go.Scatter(x=stock_df.index, y=stock_df['Signal'], mode='lines', name='Signal')])
    fig.add_trace(go.Bar(x=stock_df.index, y=stock_df['Hist'], name='Hist', marker_color='orange'))
    fig.update_layout(title=f'{selected_stock} MACD', xaxis_title='Date', yaxis_title='Value')
    main_layout.append(fig)

def display_rsi(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    stock_df['RSI'] = ta.rsi(stock_df['Close'])
    
    fig = go.Figure(data=[go.Scatter(x=stock_df.index, y=stock_df['RSI'], mode='lines', name='RSI')])
    fig.update_layout(title=f'{selected_stock} RSI', xaxis_title='Date', yaxis_title='Value')
    main_layout.append(fig)

# Add the layout to the dashboard
dashboard = pn.template.MaterialTemplate(
    title='Stock Analysis Dashboard',
    sidebar=[ticker_select, start_date_picker, end_date_picker, analysis_type_select, additional_info, analyze_button, summary_button, oschero_button, oschero_graph_select],
    main=[main_layout]
)

# Serve the dashboard
dashboard.servable()