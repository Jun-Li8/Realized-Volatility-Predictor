import sys
import numpy as np
import time
import datetime
from datetime import datetime
import requests
import csv
from calculate_trend import first_derivative, second_derivative

def add_day(apikey, ticker):
    """
    This function adds onto an already existing database for a specific ticker.
    Adds today's data.

    Note that in order for this function to execute, you must have a TD Ameritrade
    account and provide an apikey to access the information

    apikey: apikey
    tickerlist: list of tickers in the format of ['TICKER1', 'TICKER2', etc]
    """
    #for i, ticker in enumerate(tickerlist):
    print("Adding today to the %s database" % ticker)

    with open("Data/%s_stock_normalized.csv" % ticker, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    stocks = []
    volumes = []
    for i, point in enumerate(data):
        if i == 0:
            continue
        stocks.append(float(point[2]))
        volumes.append(float(point[1]))

    stock_moving_average = sum(stocks)/len(stocks)
    volume_moving_average = sum(volumes)/len(volumes)

    if data[-1][0] == datetime.strftime(datetime.now(), "%Y-%m-%d"):
        print("Database was just updated with today's %s data, moving on to next ticker" % ticker)
        return 0
    if datetime.today().weekday() == 5 or datetime.today().weekday() == 6:
        print("Today is the weekend, and there will be no updated stock")
        return 0

    link = 'https://api.tdameritrade.com/v1/marketdata/%s/quotes' % ticker
    history_link = 'https://api.tdameritrade.com/v1/marketdata/%s/pricehistory' % ticker
    specs = {'apikey':apikey}
    month_specs = {'apikey':apikey, 'period':1, 'periodType':'month', 'frequency':1, 'frequencyType':'daily'}
    today = requests.get(url = link, params = specs)
    month = requests.get(url = history_link, params = month_specs)

    today_data = today.json()
    month_data = month.json()

    stock = today_data[ticker]
    two_weeks_data = month_data['candles'][-10:]

    yearhigh = stock['52WkHigh']
    yearlow = stock['52WkLow']
    dayopen = stock['openPrice']
    dayclose = stock['closePrice']
    volume = stock['totalVolume']

    yesterday_close_to_today_open, past_first_derivative, past_avg_normalized_open_to_close = first_derivative(two_weeks_data)
    past_second_derivative = second_derivative(two_weeks_data)

    # Percent change of day close vs day open
    normalized_day_change = (dayclose - dayopen) / dayopen

    # Normalized volume compared to max and min volume
    normalized_volume_to_moving_average = volume*0.5/volume_moving_average

    # Normalized day open compared to 52wk high and low
    normalized_open_to_year = (dayopen - yearlow) / (yearhigh - yearlow)

    # Normalized day open to moving average
    normalized_open_to_moving_average = dayopen*0.5/stock_moving_average

    time = datetime.strftime(datetime.now(), "%Y-%m-%d")

    if dayclose > dayopen:
        buy = 1
    else:
        buy = 0

    data[-1][-1] = buy

    data.append([time, volume, dayopen, dayclose, normalized_day_change, \
    normalized_volume_to_moving_average, normalized_open_to_moving_average, \
    normalized_open_to_year, yesterday_close_to_today_open, past_first_derivative, \
    past_avg_normalized_open_to_close, past_second_derivative,-1])

    file = open('Data/%s_stock_normalized.csv' % ticker, mode='w', newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerows(data)
    # writer.writerow([datetime.strftime(datetime.now(), "%Y-%m-%d"), \
    # prev_result, past_day_trend, past_trend, past_curve, from_close_to_open, \
    # open_to_moving_average, open_to_year_average, volume_to_moving_average, buy])
    print("Today added to Data/%s_stock_normalized database" % ticker)

if __name__ == "__main__":
    add_day(sys.argv[1], sys.argv[2])
