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
                             name="Pivot Points"))
    fig.update_layout(title=f"{selected_stock} Candlestick Chart with Pivot Points",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Display Chaikin Oscillator
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_df.index, stock_df['Chaikin_Oscillator'], label="Chaikin Oscillator", color="blue")
    ax.axhline(0, linestyle='--', alpha=0.5, color='gray')
    ax.fill_between(stock_df.index, stock_df['Chaikin_Oscillator'], 0, where=stock_df['Chaikin_Oscillator'] >= 0, facecolor='green', interpolate=True, alpha=0.5)
    ax.fill_between(stock_df.index, stock_df['Chaikin_Oscillator'], 0, where=stock_df['Chaikin_Oscillator'] < 0, facecolor='red', interpolate=True, alpha=0.5)
    ax.set_title(f"{selected_stock} Chaikin Oscillator")
    ax.set_xlabel("Date")
    ax.set_ylabel("Chaikin Oscillator")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Display Stochastic Oscillator and Price
    chart_stochastic_oscillator_and_price(selected_stock, stock_df)

# Main function to execute when 'Analyze' button is clicked
def analyze_button_click(event):
    selected_stock = ticker_selector.value
    analysis_type = analysis_type_selector.value
    start_date, end_date = date_range_picker.value_as_date
    additional_info_selected = {key: widget for key, widget in additional_info.items()}
    
    analysis_result = display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
    additional_info_result = display_additional_information(selected_stock, additional_info_selected)
    
    # Display the stock analysis result
    template.main[0][0] = pn.Column(
        pn.pane.Markdown(f"### {selected_stock} - {analysis_type} Analysis"),
        analysis_result,
        additional_info_result
    )

# Event handler for 'Analyze' button click
analyze_button.on_click(analyze_button_click)

# Display the app
template.servable()