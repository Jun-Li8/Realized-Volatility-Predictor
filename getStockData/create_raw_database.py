import csv
import requests
import sys

def create_raw_database(apikey, ticker):
    file = open("Data/%s_raw_data.csv" % ticker, mode='w', newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Datetime", "Open", "Close", "High", "Low", "Volume"])

    historylink = 'https://api.tdameritrade.com/v1/marketdata/%s/pricehistory' % ticker
    history_specs = {'apikey':apikey, 'period':1, 'periodType':'year', 'frequency':1, 'frequencyType':'daily'}
    history = requests.get(url = historylink, params = history_specs)
    history_data = history.json()
    data = history_data['candles']

    s = []
    for each in data:
        s.append([each['datetime'], each['open'], each['close'], each['high'], each['low'], each['volume']])

    writer.writerows(s)

if __name__ == "__main__":
    create_raw_database(sys.argv[1], sys.argv[2])
