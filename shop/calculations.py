# calculations.py
import json
import pandas as pd

def va(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period value
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the Volume Average
    df['va'] = df['volume'].rolling(window=period).mean()

    # Check if the 'id' and 'va' columns exist in the DataFrame
    if 'id' not in df.columns or 'va' not in df.columns:
        print("Error: The 'id' and/or 'va' column does not exist in the DataFrame.")
        return

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['va'])}}
    return results

def movingAverage(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period value
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the moving average
    df['moving_average'] = df['close'].rolling(window=period).mean()

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['moving_average'])}}

    return results

def atr(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period value
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the true range
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = abs(df['high'] - df['close'].shift())
    df['low_prev_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

    # Calculate the average true range
    df['atr'] = df['true_range'].rolling(window=period).mean()

    # Check if the 'id' and 'atr' columns exist in the DataFrame
    if 'id' not in df.columns or 'atr' not in df.columns:
        print("Error: The 'id' and/or 'atr' column does not exist in the DataFrame.")
        return

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['atr'])}}

    return results

def vma(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the Volume Moving Average
    df['vma'] = (df['volume'] * df['close']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

    # Check if the 'id' and 'vma' columns exist in the DataFrame
    if 'id' not in df.columns or 'vma' not in df.columns:
        print("Error: The 'id' and/or 'vma' column does not exist in the DataFrame.")
        return

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['vma'])}}

    return results

def satr(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period, minATR, and maxATR values
    period = params_dict['period']
    minATR = params_dict['minATR']
    maxATR = params_dict['maxATR']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the true range
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = abs(df['high'] - df['close'].shift())
    df['low_prev_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

    # Calculate the average true range
    df['atr'] = df['true_range'].rolling(window=period).mean()

    # Calculate the SATR
    df['satr'] = df['true_range'].rolling(window=period).mean()

    # Replace the true range of small and large bars with the previous SATR value
    mask = (df['true_range'] < minATR * df['atr']) | (df['true_range'] > maxATR * df['atr'])
    df.loc[mask, 'true_range'] = df['satr'].shift()

    # Recalculate the SATR
    df['satr'] = df['true_range'].rolling(window=period).mean()

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['satr'])}}

    return results

def rsi(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period value
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the difference in price from the previous period
    delta = df['close'].diff()

    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:]

    # Make two separate series: one for gains and one for losses
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA (Exponential Weighted Moving Average)
    roll_up = up.ewm(span=period).mean()
    roll_down = down.abs().ewm(span=period).mean()

    # Calculate the RSI based on EWMA
    RS = roll_up / roll_down
    RSI = (100.0 - (100.0 / (1.0 + RS)))/100

    # Add RSI to the dataframe
    df['rsi'] = RSI

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['rsi'])}}

    return results

def va_slope(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period value
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the Volume Average
    df['va'] = df['volume'].rolling(window=period).mean()

    # Calculate the slope of the Volume Average
    df['va_slope'] = df['va'].diff()/df['va']

    # Check if the 'id' and 'va_slope' columns exist in the DataFrame
    if 'id' not in df.columns or 'va_slope' not in df.columns:
        print("Error: The 'id' and/or 'va_slope' column does not exist in the DataFrame.")
        return

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['va_slope'])}}

    return results

def ma_slope(stock_data, params, study_indicator_id):
    # Parse the params string into a dictionary
    params_dict = json.loads(params)

    # Extract the period value
    period = params_dict['period']

    # Convert the stock_data queryset to a DataFrame
    df = pd.DataFrame(list(stock_data.values()))

    # Calculate the Moving Average
    df['moving_average'] = df['close'].rolling(window=period).mean()

    # Calculate the slope of the Moving Average
    df['ma_slope'] = df['moving_average'].diff()/df['moving_average']

    # Check if the 'id' and 'ma_slope' columns exist in the DataFrame
    if 'id' not in df.columns or 'ma_slope' not in df.columns:
        print("Error: The 'id' and/or 'ma_slope' column does not exist in the DataFrame.")
        return

    # Convert the DataFrame to a dictionary and format the values
    results = {study_indicator_id: {id: json.dumps({"value": value}) for id, value in zip(df['id'], df['ma_slope'])}}

    return results