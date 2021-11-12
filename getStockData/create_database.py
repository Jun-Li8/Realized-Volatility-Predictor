import requests
import sys
import os.path
from os import path
import csv
import signal
import datetime
import pandas as pd
from calculate_trend import first_derivative, second_derivative

# Set up keyboard interrupt to stop updating the csv file
# No longer needed but just kept for future reference
"""
def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Stopping news watch...".format(signal))
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)
"""

def pair_ticker(ticker, name):
    with open("Data/pair_ticker.csv", mode='r') as read_file:
        reader = csv.reader(read_file)
        f = list(reader)

    for row in f:
        if row.count(ticker) > 0:
            print("Ticker (%s) and company name (%s) already exists" % (ticker, row[1]))
            return

    file = open("Data/pair_ticker.csv", mode='a', newline='')
    writer = csv.writer(file, delimiter=',')
    name = name.split(" - ")
    writer.writerow([ticker, name[0].replace(" Common Stock", "").replace(", ", "").replace("Inc","")])

def create_database(apikey, ticker, r=False):
    """
    This function creates a database that contains the past year's stock data for companies in tickerlist.

    Note that in order for this function to execute, you must have a TD Ameritrade
    account and provide an apikey to access the information

    apikey: apikey
    tickerlist: list of tickers in the format of ['TICKER1', 'TICKER2', etc]
    replace: if replace is True, then even if the database with the ticker already exists,
        the program will still execute and replace it
    """

    # if len(stock_avg) != len(tickerlist) or len(vol_avg) != len(tickerlist):
    #     stock_avg = [0]*len(tickerlist)
    #     vol_avg = [0]*len(tickerlist)

    # for j, ticker in enumerate(tickerlist):
    print("Creating %s database..." % ticker)
    """
    Checking if the database already exists and isn't empty.
    If the database exists and isn't empty and replace == False, then we
        move on to the next ticker
    If the database exists and replace == True,
        then we replace it by executing the program.
    Otherwise we create the database.
    """

    if path.exists('Data/%s_stock_normalized.csv' % ticker) and r == False:
        df = pd.read_csv('Data/%s_stock_normalized.csv' % ticker)
        if not df.empty:
            print("Database containing %s already exists, moving to next ticker." % ticker)
            stock_moving_average = []
            file = open("Data/%s_stock_normaized.csv" % ticker, mode='r')
            reader = csv.reader(file, delimiter=',')
            for i, data in enumerate(reader):
                if i == 0:
                    continue
                stock_moving_average.append(data[1])

            return sum(stock_moving_average)/len(stock_moving_average)

    # Get price history
    link = 'https://api.tdameritrade.com/v1/marketdata/%s/quotes' % ticker
    historylink = 'https://api.tdameritrade.com/v1/marketdata/%s/pricehistory' % ticker

    specs = {'apikey':apikey}
    history_specs = {'apikey':apikey, 'period':20, 'periodType':'year', 'frequency':1, 'frequencyType':'daily'}

    overall = requests.get(url = link, params = specs)
    history = requests.get(url = historylink, params = history_specs)

    overall_data = overall.json()
    history_data = history.json()

    yearhigh = overall_data[ticker]['52WkHigh']
    yearlow = overall_data[ticker]['52WkLow']

    # Pair ticker
    name = overall_data[ticker]['description']
    pair_ticker(ticker, name)

    stock_moving_average = 0
    volume_moving_average = 0
    max_volume = float('-inf')
    min_volume = float('inf')
    prev_result = 0
    prev_close = 0

    d = history_data['candles']

    data = []
    for i, row in enumerate(d):
        """
        Gets normalized past data trends

        yesterday_close_to_today_open: returns the percent increase/decrease from yesterday close to today open
        past_first_derivative: returns the average slope of the past 2 weeks
        past_avg_normalized_open_to_close: returns the average normalized day open to day close percent increase/decrease
        past_second_derivative: returns the average second derivative of the past 2 weeks

        In the case of data within the first 2 weeks, return past data trends of however many days have passed since epoche

        """

        dayopen = row['open']
        dayclose = row['close']
        volume = row['volume']
        t = row['datetime']/1000
        time = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d')

        """
        Updates
        ------------------------------------------------------------------------
        """
        # Update stock_moving_average
        if i == 0:
            stock_moving_average = stock_moving_average + dayopen
        else:
            stock_moving_average = (stock_moving_average * i + dayopen) / (i + 1)

        if i == 0:
            volume_moving_average = volume_moving_average + volume
        else:
            volume_moving_average = (volume_moving_average * i + volume) / (i + 1)
        """
        ------------------------------------------------------------------------
        """

        # Percent change of day close vs day open
        normalized_day_change = (dayclose - dayopen) / dayopen

        # Normalized volume compared to max and min volume
        normalized_volume_to_moving_average = volume*0.5/volume_moving_average

        # Normalized day open compared to 52wk high and low
        normalized_open_to_year = (dayopen - yearlow) / (yearhigh - yearlow)

        # Normalized day open to moving average
        normalized_open_to_moving_average = dayopen*0.5/stock_moving_average

        if i in range(0,10):
            if i == 0:
                past_first_derivative = 0
                past_avg_normalized_open_to_close = 0
                past_second_derivative = 0
                yesterday_close_to_today_open = 0
            else:
                yesterday_close_to_today_open, past_first_derivative, past_avg_normalized_open_to_close = first_derivative(d[0:i+1])
                if i == 1:
                    past_second_derivative = 0
                else:
                    past_second_derivative = second_derivative(d[0:i])
        else:
            yesterday_close_to_today_open, past_first_derivative, past_avg_normalized_open_to_close = first_derivative(d[i-9:i])
            past_second_derivative = second_derivative(d[i-9:i])

        # Seeing if today should have been buy or not
        if dayclose > dayopen:
            buy = 1
        else:
            buy = 0

        data.append([time, volume, dayopen, dayclose, normalized_day_change, \
        normalized_volume_to_moving_average, normalized_open_to_moving_average, \
        normalized_open_to_year, yesterday_close_to_today_open, past_first_derivative, \
        past_avg_normalized_open_to_close, past_second_derivative,-1])
        if i != 0:
            data[i-1][-1] = buy

    file = open('Data/%s_stock_normalized.csv' % ticker, mode='w', newline='')
    writer = csv.writer(file, delimiter=',')

    writer.writerow(['Time', 'Volume', 'Day Open', 'Day Close', 'Normalized Day Change', \
    'Normalized Volume to Moving Average', 'Normalized Open to Moving Average',\
    'Normalized Open (52wk High/Low)', 'Normalized Yesterday Close to Today Open', \
    'Past First Derivative', 'Past Average Normalized Open to Close', 'Past Second Derivative', 'Buy'])

    for row in data:
        writer.writerow(row)

    print("Database for %s has been created." % ticker)


if __name__ == "__main__":
    create_database(sys.argv[1], sys.argv[2], r=True)
