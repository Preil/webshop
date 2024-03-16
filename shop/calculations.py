# calculations.py
import json
import pandas as pd

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
    print("Results MA function:")
    print(results)
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
    print("Results ATR function:")
    print(results)
    return results
# Add more functions as needed...