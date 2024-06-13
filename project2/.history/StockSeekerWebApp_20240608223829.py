import streamlit as st
# import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import timedelta
from datetime import datetime
import datetime
import plotly.express as px
import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter# Add this import statement
import subprocess
# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Function to call local CSS sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Provide the path to the style.css file
style_css_path = r"C:\Users\Admin\Documents\MLAI\CSCN8030\Proje_Sprint2_V1\docs\assets\style.css"
local_css(style_css_path)

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Stock tickers combo box
st.sidebar.subheader("STOCK SEEKER WEB APP")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime.datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.datetime.now())

# Load DataFrame from CSV file
yf = pd.read_csv('stock_data.csv')

# Analysis type selection
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

# Display additional information based on user selection
st.sidebar.subheader("Display Additional Information")
selected_options = {
    "Stock Actions": st.sidebar.checkbox("Stock Actions"),
    "Quarterly Financials": st.sidebar.checkbox("Quarterly Financials"),
    "Institutional Shareholders": st.sidebar.checkbox("Institutional Shareholders"),
    "Quarterly Balance Sheet": st.sidebar.checkbox("Quarterly Balance Sheet"),
    "Quarterly Cashflow": st.sidebar.checkbox("Quarterly Cashflow"),
    "Analysts Recommendation": st.sidebar.checkbox("Analysts Recommendation"),
    "Predicted Prices": st.sidebar.checkbox("Predicted Prices")  # Add Predicted Prices option
}

# Submit button
button_clicked = st.sidebar.button("Analyze")

# Summary button
summary_clicked = st.sidebar.button("OsTron")



def run_notebook(notebook_path):
    command = f'jupyter nbconvert --to notebook --execute {notebook_path} --output {notebook_path.replace(".ipynb", "_executed.ipynb")}'
    subprocess.run(command, shell=True, check=True)

# Example usage
notebook_path = 'StockSeeker.ipynb'
run_notebook(notebook_path)

# Function to handle analysis
def handle_analysis(selected_stock, analysis_type, start_date, end_date):
    if analysis_type != "Predicted Prices":
        display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
        display_additional_information(selected_stock, selected_options)
    else:
        display_predicted_prices(selected_stock, start_date, end_date)

# Function to display stock analysis
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

# Function to display additional information
def display_additional_information(selected_stock, selected_options):
    for option, checked in selected_options.items():
        if checked:
            st.subheader(f"{selected_stock} - {option}")
            if option == "Stock Actions":
                display_action = yf.Ticker(selected_stock).actions
                if not display_action.empty:
                    st.write(display_action)
                else:
                    st.write("No data available")
            elif option == "Quarterly Financials":
                display_financials = yf.Ticker(selected_stock).quarterly_financials
                if not display_financials.empty:
                    st.write(display_financials)
                else:
                    st.write("No data available")
            elif option == "Institutional Shareholders":
                display_shareholders = yf.Ticker(selected_stock).institutional_holders
                if not display_shareholders.empty:
                    st.write(display_shareholders)
                else:
                    st.write("No data available")
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balance_sheet
                if not display_balancesheet.empty:
                    st.write(display_balancesheet)
                else:
                    st.write("No data available")
            elif option == "Quarterly Cashflow":
                display_cashflow = yf.Ticker(selected_stock).quarterly_cashflow
                if not display_cashflow.empty:
                    st.write(display_cashflow)
                else:
                    st.write("No data available")
            elif option == "Analysts Recommendation":
                display_analyst_rec = yf.Ticker(selected_stock).recommendations
                if not display_analyst_rec.empty:
                    st.write(display_analyst_rec)
                else:
                    st.write("No data available")
            elif option == "Predicted Prices":
                display_predicted_prices(selected_stock, start_date, end_date)

# Function to display predicted prices
def display_predicted_prices(selected_stock, start_date, end_date):
    st.subheader(f"{selected_stock} - Predicted Prices")
    
    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)
    
    # Prepare the data
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
    fig.update_layout(title=f'{selected_stock} Predicted Prices',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)

# Function to detect pivot points
def isPivot(candle, window, df):
    """
    Function that detects if a candle is a pivot/fractal point
    Args:
        candle: Candle index (datetime object)
        window: Number of days before and after the candle to test if pivot
        df: DataFrame containing the stock data
    Returns:
        1 if pivot high, 2 if pivot low, 3 if both, and 0 default
    """
    # Assuming candle is a datetime object
    candle_timestamp = pd.Timestamp(candle)
    if candle_timestamp - datetime.timedelta(days=window) < df.index[0] or candle_timestamp + datetime.timedelta(days=window) >= df.index[-1]:
        return 0

    pivotHigh = 1
    pivotLow = 2
    start_index = candle_timestamp - datetime.timedelta(days=window)
    end_index = candle_timestamp + datetime.timedelta(days=window)
    for i in range((end_index - start_index).days + 1):
        current_date = start_index + datetime.timedelta(days=i)
    
        if 'low' in df.columns and df.loc[candle_timestamp, 'low'] > df.loc[current_date, 'low']:
            pivotLow = 0
        if 'high' in df.columns and df.loc[candle_timestamp, 'high'] < df.loc[current_date, 'high']:
            pivotHigh = 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

# Function to calculate Chaikin Oscillator
def calculate_chaikin_oscillator(data):
    """
    Calculate Chaikin Oscillator using pandas_ta.
    """
    data['ADL'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
    data['Chaikin_Oscillator'] = ta.ema(data['ADL'], length=3) - ta.ema(data['ADL'], length=10)
    return data

# Define the calculate_stochastic_oscillator function
def calculate_stochastic_oscillator(df, period=14):
    """
    Calculate Stochastic Oscillator (%K and %D).
    """
    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max() 
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

def chart_stochastic_oscillator_and_price(ticker, df):
    """
    Plots the stock's closing price with its 50-day and 200-day moving averages,
    and the Stochastic Oscillator (%K and %D) below the price chart.
    """
    plt.figure(figsize=[16, 8])
    plt.style.use('default')
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(16, 8))
    fig.suptitle(ticker, fontsize=16)

    # Plotting the closing price and moving averages on the first subplot
    ax[0].plot(df['Close'], color='black', linewidth=1, label='Close')
    ax[0].plot(df['ma50'], color='blue', linewidth=1, linestyle='--', label='50-day MA')
    ax[0].plot(df['ma200'], color='red', linewidth=1, linestyle='--', label='200-day MA')
    ax[0].set_ylabel('Price [\$]')
    ax[0].grid(True)
    ax[0].legend(loc='upper left')
    ax[0].axes.get_xaxis().set_visible(False)  # Hide X axis labels for the price plot

    # Plotting the Stochastic Oscillator on the second subplot
    ax[1].plot(df.index, df['%K'], color='orange', linewidth=1, label='%K')
    ax[1].plot(df.index, df['%D'], color='grey', linewidth=1, label='%D')
    ax[1].grid(True)
    ax[1].set_ylabel('Stochastic Oscillator')
    ax[1].set_ylim(0, 100)
    ax[1].axhline(y=80, color='b', linestyle='-')  # Overbought threshold
    ax[1].axhline(y=20, color='r', linestyle='-')  # Oversold threshold
    ax[1].legend(loc='upper left')

    # Formatting the date labels on the X-axis
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Adjust space between the plots

    st.pyplot(fig)  # Display the plot in Streamlit
    return data

def display_technical_summary(selected_stock, start_date, end_date):
    st.subheader(f"{selected_stock} - Technical Summary")
    
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    # Calculate Chaikin Oscillator
    stock_df = calculate_chaikin_oscillator(stock_df)
    stock_df = calculate_stochastic_oscillator(stock_df)

    # Detect pivot points
    window = 5
    stock_df['isPivot'] = stock_df.apply(lambda x: isPivot(x.name, window, stock_df), axis=1)
    stock_df['pointpos'] = stock_df.apply(lambda row: row['Low'] - 1e-3 if row['isPivot'] == 2 else (row['High'] + 1e-3 if row['isPivot'] == 1 else np.nan), axis=1)

    # Plot candlestick with pivots
    fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                         open=stock_df['Open'],
                                         high=stock_df['High'],
                                         low=stock_df['Low'],
                                         close=stock_df['Close'],
                                         name='Candlestick')])
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['pointpos'], mode='markers',
                             marker=dict(size=5, color="MediumPurple"),
                             name="Pivot"))
    
    fig.update_layout(title=f'{selected_stock} Candlestick Chart with Pivots',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Plot Chaikin Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Chaikin_Oscillator'], mode='lines', name='Chaikin Oscillator'))
    fig.update_layout(title=f'{selected_stock} Chaikin Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Chaikin Oscillator Value')
    st.plotly_chart(fig)
    # Plot Stochastic Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['%D'], mode='lines', name='%D'))
    fig.update_layout(title=f'{selected_stock} Stochastic Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Stochastic Oscillator Value')
    st.plotly_chart(fig)
# Define the display_advanced_analysis function
def display_advanced_analysis(selected_stock, start_date, end_date):
    st.subheader(f"Advanced Analysis for {selected_stock}")

    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)

    # Add Moving Average Convergence Divergence (MACD)
    df['12 Day EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26 Day EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12 Day EMA'] - df['26 Day EMA']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # MACD Buy/Sell Signals
    df['MACD_Buy_Signal'] = np.where(df['MACD'] > df['Signal Line'], df['MACD'], np.nan)
    df['MACD_Sell_Signal'] = np.where(df['MACD'] < df['Signal Line'], df['MACD'], np.nan)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal Line'], label='Signal Line', color='red')
    ax.scatter(df.index, df['MACD_Buy_Signal'], marker='^', color='g', label='MACD Buy Signal')
    ax.scatter(df.index, df['MACD_Sell_Signal'], marker='v', color='r', label='MACD Sell Signal')
    ax.set_title(f'MACD for {selected_stock}')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # Format       
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    st.pyplot(fig)

    # Add Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # RSI Buy/Sell Signals
    df['RSI_Buy_Signal'] = np.where(df['RSI'] < 30, df['RSI'], np.nan)
    df['RSI_Sell_Signal'] = np.where(df['RSI'] > 70, df['RSI'], np.nan)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax.axhline(70, linestyle='--', alpha=0.5, color='green')
    ax.scatter(df.index, df['RSI_Buy_Signal'], marker='^', color='g', label='RSI Buy Signal')
    ax.scatter(df.index, df['RSI_Sell_Signal'], marker='v', color='r', label='RSI Sell Signal')
    ax.set_title(f'RSI for {selected_stock}')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # Format the date
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    st.pyplot(fig)
def stochastic_calculator(selected_stock, start_date, end_date):
    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Stochastic Oscillator (%K and %D)
    high14 = df['High'].rolling(window=14).max()
    low14 = df['Low'].rolling(window=14).min()
    df['%K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['%D'] = df['%K'].rolling(window=3).mean()

    fig, axs = plt.subplots(2, figsize=(12, 8), sharex=True)

    # Plotting the closing prices and moving averages
    axs[0].plot(df.index, df['Close'], label='Closing Price', color='blue')
    axs[0].plot(df.index, df['MA50'], label='50-day MA', color='red')
    axs[0].plot(df.index, df['MA200'], label='200-day MA', color='green')
    axs[0].set_ylabel('Price')
    axs[0].legend(loc='upper left')
    axs[0].set_title(f'{selected_stock} - Closing Prices and Moving Averages')

    # Plotting the Stochastic Oscillator
    axs[1].plot(df.index, df['%K'], label='%K', color='blue')
    axs[1].plot(df.index, df['%D'], label='%D', color='red')
    axs[1].axhline(y=20, color='gray', linestyle='--')
    axs[1].axhline(y=80, color='gray', linestyle='--')
    axs[1].set_ylabel('Oscillator')
    axs[1].set_title('Stochastic Oscillator')
    axs[1].legend(loc='upper left')

    date_format = DateFormatter("%Y-%m-%d")
    axs[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Execute analysis when button is clicked
if button_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            handle_analysis(selected_stock, analysis_type, start_date, end_date)
    else:
        st.sidebar.warning("Please select at least one stock ticker.")

# Execute technical summary when summary button is clicked
if summary_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            display_technical_summary(selected_stock, start_date, end_date)
            display_advanced_analysis(selected_stock, start_date, end_date)   
            stochastic_calculator(selected_stock, start_date, end_date)
# Define the stochastic_calculator function
    else:
        st.sidebar.warning("Please select at least one stock ticker.")

# Main logic for handling button clicks
# if button_clicked:
#     for selected_stock in selected_stocks:
#         handle_analysis(selected_stock, analysis_type, start_date, end_date)

# if summary_clicked:
#     for selected_stock in selected_stocks:
#         st.subheader(f"Oscillatron Summary for {selected_stock}")
#         stochastic_calculator(selected_stock, start_date, end_date)
# Define the stochastic_calculator function


