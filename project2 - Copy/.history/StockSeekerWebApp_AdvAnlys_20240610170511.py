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
    return fig

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
    
        if 'Low' in df.columns and df.loc[candle_timestamp, 'Low'] > df.loc[current_date, 'Low']:
            pivotLow = 0
        if 'High' in df.columns and df.loc[candle_timestamp, 'High'] < df.loc[current_date, 'High']:
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

# Function to calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, period=14):
    """
    Calculate Stochastic Oscillator (%K and %D).
    """
    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max() 
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

# Function to display stochastic oscillator and price chart
def chart_stochastic_oscillator_and_price(ticker, df):
    """
    Plots the stock's closing price with its 50-day and 200-day moving averages,
    and the Stochastic Oscillator (%K and %D) below the price chart.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=(f"{ticker} Price and Moving Averages", f"{ticker} Stochastic Oscillator"))

    # Plotting the closing price and moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-day MA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200-day MA', line=dict(color='purple')), row=1, col=1)

    # Plotting the Stochastic Oscillator
    fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D', line=dict(color='red')), row=2, col=1)
    fig.update_layout(title=f"{ticker} Price and Stochastic Oscillator", xaxis_title="Date")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)