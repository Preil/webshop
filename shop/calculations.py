# calculations.py

def movingAverage(data, period):
    # Ensure the data is sorted in ascending order by date
    data = sorted(data, key=lambda x: x.timestamp)

    # Extract the closing prices
    prices = [item.close for item in data]

    # Calculate the initial average
    moving_average = sum(prices[:period]) / period
    moving_averages = [moving_average]

    # Update the moving average for each price
    for i in range(period, len(prices)):
        moving_average = moving_average - prices[i - period] / period + prices[i] / period
        moving_averages.append(moving_average)

    return moving_averages

def atr(data):
    # Perform calculations for indicator2
    return result

# Add more functions as needed...