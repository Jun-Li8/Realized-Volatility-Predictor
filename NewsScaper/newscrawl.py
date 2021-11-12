import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import csv
import sys
import requests
import signal
import textdistance
import datetime
from datetime import datetime
import linecache

# def keyboardInterruptHandler(signal, frame):
#     print("KeyboardInterrupt (ID: {}) has been caught. Stopping news watch...".format(signal))
#     exit(0)
#
# signal.signal(signal.SIGINT, keyboardInterruptHandler)

def get_news():

    ret = []

    url = "https://finance.yahoo.com/news/"
    print("Scraping: " + url)

    client = urlopen(url)
    html = client.read()
    client.close()

    page = soup(html, "html.parser")

    news = page.find_all("p", {"class":"Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0)"})#[0].get_text())
    for n in news:
        line = n.get_text().replace('\n', '')
        ret.append(line)
    print("Finished scraping for " + url + '\n')

#---------------------------------------------------------------------------------------------------
# stock market news
    url = "https://finance.yahoo.com/topic/stock-market-news"
    print("Scraping: " + url)

    client = urlopen(url)
    html = client.read()
    client.close()

    page = soup(html, "html.parser")

    news = page.find_all("li", {"class":"js-stream-content"})
    #news = page.find_all("p", {"class": "Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0) C(#959595)"})
    for n in news:
        summary = n.find("h3")
        try:
            line = summary.get_text()
            ret.append(line)
        except:
            print("Line could not be found")
    print("Finished scraping for " + url + '\n')

#---------------------------------------------------------------------------------------------------
# forbes
    forbes = "https://www.forbes.com/news/"
    print("Scraping " + forbes)

    r = requests.get(forbes)
    con = r.content
    forbes_page = soup(con, "html.parser")
    l = forbes_page.find_all("article")
    for each in l:
        line = each.find("h2").get_text()
        line = line + ' ' +  each.find("div", {"class":"stream-item__description"}).get_text()
        if line == '' or line == '\n':
            continue
        ret.append(line)
    print("Finished scraping " + forbes)

    return ret

def add_news(news):
    with open("Data/news.txt", mode='r',errors='ignore') as read_file:
        data = list(read_file)
    with open("Data/news.txt", mode='a', newline='', errors='ignore') as file:
        for line in news:
            if check_if_news_exists(line, data) == False and remove_yahoo_finance(line) == False:
                print(line)
                file.write('|' + line + '\n')

    remove_empty_lines("Data/news.txt")

def get_news_for(source):
    print("Getting news for %s" % source)
    str = source.split(' ')
    s = str[0]
    for i, each in enumerate(str):
        if i == 0:
            continue
        s = s + '+' + each
    url = "https://www.google.com/search?q=%s&tbm=nws" % s
    r = requests.get(url)
    con = r.content
    page = soup(con, "html.parser")

    file = open("Data/news.txt")
    data = list(file)

    ret = []

    print("Opening up %s" % source)

    a = page.find_all("div", {"class":"BNeawe vvjwJb AP7Wnd"})
    for each in a:
        line = each.get_text()
        if check_if_news_exists(line, data) == False:
            line.replace('\n','')
            ret.append(line)

    return ret

def check_if_news_exists(first_sentence, data):
    for row in data:
        if textdistance.jaro.similarity(first_sentence,row.split('|')[1]) >= 0.75:
            return True
    return False

def check_news_eval(line):
    bad = ["decrease", "lower", "sank", "loss", "drop", "fell", "underperform", "bankruptcy"]
    good = ["increase", "higher", "gain", "rise", "outperform"]
    if any(b in line for b in bad):
        return 1
    elif any(g in line for g in good):
        return 2
    else:
        return 3

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def remove_empty_lines(file):
    with open(file, mode='r') as f:
        lines = f.readlines()

    with open(file, mode='w') as f:
        for line in lines:
            if line == '\n' or len(line) == 0:
                pass
            else:
                f.write(line)

def remove_yahoo_finance(line):
    words = ['Final Round', "Yahoo Finance's", "Akiko Fujita"]
    if any(w in line for w in words):
        return True
    else:
        return False

if __name__ == "__main__":
    print(datetime.now())
    try:
        r = get_news_for(sys.argv[1])
        print(r)
        print("Finished getting news for %s" % sys.argv[1])

    except:
        print("Check data file for empty lines")
        PrintException()
        r = get_news()
        print("Adding news to Data/news.txt")
        add_news(r)
        with open("Data/pair_ticker.csv", mode='r') as file:
            reader = csv.reader(file)
            data = list(reader)
        for company in data:
            r = get_news_for(company[1])
            add_news(r)
