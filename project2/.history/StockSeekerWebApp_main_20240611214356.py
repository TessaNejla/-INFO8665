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

# Define the callback functions
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    # Download historical data
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)

    # Create a new Panel pane to hold the output
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
            title = "Analysts Recommendation"
        elif option == "Predicted Prices":
            # Call display_predicted_prices function
            info_panes.append(display_predicted_prices(selected_stock, start_date_picker.value, end_date_picker.value))
            continue  # Skip appending to info_panes for predicted prices
        
        if not data.empty:
            info_panes.append(pn.Column(f"## {selected_stock} - {title}", pn.widgets.DataFrame(data)))
        else:
            info_panes.append(pn.pane.Markdown(f"## {selected_stock} - {title}\n\nNo data available."))
    
    return pn.Column(*info_panes)
# Function to Display Predicted Prices using LSTM
# Define the callback function to display predicted prices
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
    return pn.pane.Plotly(fig)
    
#Main App Logic
# Panels to display analysis and additional information
stock_analysis_pane = pn.Column()
additional_info_pane = pn.Column()

# Define callbacks for button clicks
# Define callback for analyze_button
def analyze_button_click(event):
    stock_analysis_pane.clear()
    selected_analysis_type = analysis_type_select.value  # Get the selected analysis type
    for selected_stock in ticker_select.value:
        stock_analysis_pane.append(display_stock_analysis(selected_stock, selected_analysis_type, start_date_picker.value, end_date_picker.value))
        #additional_info_pane.clear()
def summary_button_click(event):
    additional_info_pane.clear()
    for selected_stock in ticker_select.value:
        additional_info_pane.append(display_additional_information(selected_stock, additional_info.value))

# Assign callbacks to buttons
analyze_button.on_click(analyze_button_click)
summary_button.on_click(summary_button_click)

# Set up the main layout including stock_analysis_pane
main_layout = pn.Column(
    pn.pane.Markdown("# Stock Analysis Dashboard"),
    pn.Row(
        pn.Column(ticker_select, start_date_picker, end_date_picker, analysis_type_select, additional_info, analyze_button, summary_button),
        stock_analysis_pane,  # Add stock_analysis_pane to the layout
        additional_info_pane
    )
)

# Serve the app
main_layout.servable()