import panel as pn
import yfinance as yf
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
from matplotlib.dates import DateFormatter

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

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
    pn.pane.DataFrame(stock_df)  # Display DataFrame using Panel

# Function to display additional information
def display_additional_information(selected_stock, selected_options):
    for option, checked in selected_options.items():
        if checked:
            if option == "Stock Actions":
                display_action = yf.Ticker(selected_stock).actions
                if not display_action.empty:
                    pn.pane.DataFrame(display_action)  # Display DataFrame using Panel
            elif option == "Quarterly Financials":
                display_financials = yf.Ticker(selected_stock).quarterly_financials
                if not display_financials.empty:
                    pn.pane.DataFrame(display_financials)  # Display DataFrame using Panel
            elif option == "Institutional Shareholders":
                display_shareholders = yf.Ticker(selected_stock).institutional_holders
                if not display_shareholders.empty:
                    pn.pane.DataFrame(display_shareholders)  # Display DataFrame using Panel
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balance_sheet
                if not display_balancesheet.empty:
                    pn.pane.DataFrame(display_balancesheet)  # Display DataFrame using Panel
            elif option == "Quarterly Cashflow":
                display_cashflow = yf.Ticker(selected_stock).quarterly_cashflow
                if not display_cashflow.empty:
                    pn.pane.DataFrame(display_cashflow)  # Display DataFrame using Panel
            elif option == "Analysts Recommendation":
                display_analyst_rec = yf.Ticker(selected_stock).recommendations
                if not display_analyst_rec.empty:
                    pn.pane.DataFrame(display_analyst_rec)  # Display DataFrame using Panel
            elif option == "Predicted Prices":
                display_predicted_prices(selected_stock, start_date, end_date)

# Function to display predicted prices
def display_predicted_prices(selected_stock, start_date, end_date):
    pn.pane.Markdown(f"# {selected_stock} - Predicted Prices")
    
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
    pn.pane.Plotly(fig)  # Display Plotly figure using Panel

# Define the app layout
pn.extension()
pn.template.FastListTemplate(
    site="Stock Market Analysis and Prediction Web App",
    title="STOCK SEEKER WEB APP",
    sidebar=[
        pn.widgets.MultiSelect(name="Select stock tickers...", options=popular_tickers)
    ],
    main=[
        pn.widgets.Select(name="Select Analysis Type", options=["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])
    ]
).servable()