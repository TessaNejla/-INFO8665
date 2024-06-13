#============================================================================================================================================
# Stock Market Analysis and Prediction Web App
# Converted to use Panel for visualization
# NAME: TESSA NEJLA AYVAZOGLU 
# DATE: 06/10/2024
# ===========================================================================================================================================
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
import plotly.express as px
import matplotlib.pyplot as plt
import panel as pn
import hvplot.pandas  # for interactive plots

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Enable Panel extensions
pn.extension('plotly', 'matplotlib')

# Define the main template
template = pn.template.MaterialTemplate(title="Stock Market Analysis and Prediction Web App")

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Sidebar Widgets
ticker_selector = pn.widgets.MultiSelect(name='Select Stock Tickers', options=popular_tickers, size=len(popular_tickers))
# Correct the start and end parameters
date_range_picker = pn.widgets.DateRangePicker(
    name='Select Date Range',
    start=datetime(2020, 1, 1).date(),  # Use .date() to convert to date type
    end=datetime.now().date()  # Use .date() to convert to date type
)
analysis_type_selector = pn.widgets.Select(name='Select Analysis Type', options=["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])
additional_info = {
    "Stock Actions": pn.widgets.Checkbox(name="Stock Actions"),
    "Quarterly Financials": pn.widgets.Checkbox(name="Quarterly Financials"),
    "Institutional Shareholders": pn.widgets.Checkbox(name="Institutional Shareholders"),
    "Quarterly Balance Sheet": pn.widgets.Checkbox(name="Quarterly Balance Sheet"),
    "Quarterly Cashflow": pn.widgets.Checkbox(name="Quarterly Cashflow"),
    "Analysts Recommendation": pn.widgets.Checkbox(name="Analysts Recommendation"),
    "Predicted Prices": pn.widgets.Checkbox(name="Predicted Prices")  # Add Predicted Prices option
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
                display_financials = yf.Ticker(selected_stock).quarterly_financials
                if not display_financials.empty:
                    panes.append(pn.pane.DataFrame(display_financials, name=f"{selected_stock} - {option}"))
                else:
                    panes.append(pn.pane.Markdown(f"No data available for {option}"))
            elif option == "Institutional Shareholders":
                display_shareholders = yf.Ticker(selected_stock).institutional_holders
                if not display_shareholders.empty:
                    panes.append(pn.pane.DataFrame(display_shareholders, name=f"{selected_stock} - {option}"))
                else:
                    panes.append(pn.pane.Markdown(f"No data available for {option}"))
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balance_sheet
                if not display_balancesheet.empty:
                    panes.append(pn.pane.DataFrame(display_balancesheet, name=f"{selected_stock} - {option}"))
                else:
                    panes.append(pn.pane.Markdown(f"No data available for {option}"))
            elif option == "Quarterly Cashflow":
                display_cashflow = yf.Ticker(selected_stock).quarterly_cashflow
                if not display_cashflow.empty:
                    panes.append(pn.pane.DataFrame(display_cashflow, name=f"{selected_stock} - {option}"))
                else:
                    panes.append(pn.pane.Markdown(f"No data available for {option}"))
            elif option == "Analysts Recommendation":
                display_analyst_rec = yf.Ticker(selected_stock).recommendations
                if not display_analyst_rec.empty:
                    panes.append(pn.pane.DataFrame(display_analyst_rec, name=f"{selected_stock} - {option}"))
                else:
                    panes.append(pn.pane.Markdown(f"No data available for {option}"))
            elif option == "Predicted Prices":
                panes.append(display_predicted_prices(selected_stock, start_date, end_date))
    return pn.Tabs(*[(pane.name, pane) for pane in panes])

# Function to display predicted prices
def display_predicted_prices(selected_stock, start_date, end_date):
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

    # Get the predicted values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title(f'{selected_stock} Predicted Prices', fontsize=18)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'], label='Train Data')
    ax.plot(valid[['Close', 'Predictions']], label=['Actual Price', 'Predicted Price'])
    ax.legend(['Train Data', 'Actual Price', 'Predicted Price'], loc='lower right')
    
    return fig
           fig

    