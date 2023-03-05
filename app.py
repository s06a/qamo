import datetime
import numpy as np
import os.path
from math import e
import json
from flask import Response
from flask import Flask, request, render_template, jsonify
import requests
import pandas as pd
from simple_scheduler.recurring import recurring_scheduler
from sqlalchemy import create_engine
import sqlite3
import re

db = create_engine('sqlite:////static/data/db.db', echo=False)

def is_time_between(begin_time, end_time):
    # If check time is not given, default to current UTC time
    check_time = datetime.datetime.now().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def messari():
    def cal_messari(r):
      try:
        return {'coin':r['symbol'] ,'ath':float(r['athUsd']), 'cycleLow':float(r['cycleLowUsd'])}
      except:
        return {'coin':float('nan') ,'ath':float('nan'), 'cycleLow':float('nan')}
    r = requests.get('https://data.messari.io/api/v1/markets/prices-legacy').text
    r = json.loads(r)['data']
    df = [cal_messari(r) for r in r]
    df = pd.DataFrame(df)
    df = df.dropna()
    return df

class coinglass():

    def symbols():
        r = requests.get('https://fapi.coinglass.com/api/support/symbol').text
        return json.loads(r)['data']

    def longshort_ratio(symbol, period=2):
        r = requests.get('https://fapi.coinglass.com/api/futures/longShortRate?symbol=' + symbol + '&timeType=' + str(period)).text
        r= json.loads(r)['data'][0]['list']
        r = [{'exchange':r['exchangeName'], 'shortRate':float(r['shortRate']), 'longRate':float(r['longRate']), 'shortVol':r['shortVolUsd'], 'longVol':r['longVolUsd']} for r in r]
        r = pd.DataFrame(r)
        return sum(r['longRate'])/(sum(r['shortRate'])+sum(r['longRate'])+1) * 100

base_url = 'https://api.coingecko.com/api/v3'

class coingecko():

    def market():
        list = []
        for page in range(10):
            r = requests.get(base_url + '/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=' + str(page) + '&sparkline=false').text
            r = json.loads(r)
            list.append([{'coin':r['symbol'].upper()+'-USDT', 'market_cap':int(float(r['market_cap'])), 'image':r['image']} for r in r])
            r = [item for sublist in list for item in sublist]
        df = pd.DataFrame(r)
        db_write('db', df, 'coingecko_market', ifExists='replace')
        return None

    def trends():
        r = requests.get(base_url + '/search/trending').text
        r = json.loads(r)
        r = [{'name':r['item']['name'], 'coin':r['item']['symbol']} for r in r['coins']]
        return pd.DataFrame(r)

baseUrl = 'https://api.kucoin.com'

class kuCoin():

    def symbolsInfo():
        res = requests.get(baseUrl + '/api/v1/symbols').text
        res = json.loads(res)['data']
        return [stock for stock in res if stock['enableTrading'] == True]

    def symbolsList(currency):
        res = [res['symbol'] for res in kuCoin.symbolsInfo()]
        if currency == 0:
            return res
        elif currency == 'USDT':
            return [stock for stock in res if stock[-4:] == 'USDT']

    def orderList(stock, depth): # stock = 'BTC-USDT', depth = 20, 100
        res = requests.get(baseUrl + '/api/v1/market/orderbook/level2_' + str(depth) + '?symbol=' + stock).text
        return json.loads(res)['data']

    def bids(stock, depth):
        return orderList(stock, depth)['bids']

    def asks(stock, depth):
        return orderList(stock, depth)['asks']

    def bpaRatio(stock, depth): # stock = 'BTC-USDT', depth = 20, 100
        res = requests.get(baseUrl + '/api/v1/market/orderbook/level2_' + str(depth) + '?symbol=' + stock).text
        res = json.loads(res)['data']
        asks = sum([float(r[0])*float(r[1]) for r in res['asks']])
        bids = sum([float(r[0])*float(r[1]) for r in res['bids']])
        ratio = (bids/(bids+asks))*100
        return ratio

    ''' gives 1500 data lines, if you want more, you have to use start and end time
    period = 1min, 3min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 8hour, 12hour, 1day, 1week '''

    class klines(): # time, open, close, high, low, volume, amount

        def __init__(self, stock, period):
            res = requests.get(baseUrl + '/api/v1/market/candles?type=' + period + '&symbol=' + stock).text
            res = json.loads(res)['data']
            self.time = pd.DataFrame([datetime.datetime.fromtimestamp(int(float(r[0]))/1).strftime("%Y-%m-%d %H:%M:%S") for r in res][::-1])
            self.open = [float(r[1]) for r in res][::-1]
            self.close = [float(r[2]) for r in res][::-1]
            self.high = [float(r[3]) for r in res][::-1]
            self.low = [float(r[4]) for r in res][::-1]
            self.volume = [float(r[5]) for r in res][::-1]
            self.amount = [float(r[6]) for r in res][::-1]

    #logarithmic return
    def logReturn(stock, period):
        stock = kuCoin.klines(stock, period)
        df = pd.DataFrame({'price':stock.open})
        return np.log(df.price) - np.log(df.price.shift(1))

    #volatility
    def volatility(stock, period, start):
      return np.std(kuCoin.logReturn(stock, period)[-start:])

    def market():
      res = requests.get(baseUrl + '/api/v1/market/allTickers').text
      res = json.loads(res)['data']
      return res

def cal_kucoin(df, i):
    try:
        return {'name':df['ticker'][i]['symbol'], 'price':(float(df['ticker'][i]['buy']) + float(df['ticker'][i]['sell']))*0.5, 'volatility':(float(df['ticker'][i]['high'])/(float(df['ticker'][i]['low'])+1)-1)*100}
    except:
        return {'name':df['ticker'][i]['symbol'], 'price':0, 'volatility':0}

def market(currency):
    df = kuCoin.market()
    data = [cal_kucoin(df, i) for i in range(len(df['ticker'])) if currency in df['ticker'][i]['symbol']]
    return pd.DataFrame(data)

def db_write(database, data, table, ifExists='replace'):
    db = sqlite3.connect('static/data/'+database+'.db')
    data.to_sql(name=table, con=db, index=False, if_exists=ifExists)
    db.close
    return print('writing ' + table)

def db_read(database, table, howMany=0, sort=False, sort_on=''):
    db = sqlite3.connect('static/data/'+database+'.db')
    if sort == True and howMany>0:
        data = pd.read_sql('select * from ' + table, db)
        data = data.sort_values(sort_on, ascending=True)[-howMany:]
    else:
        data = pd.read_sql('select * from ' + table, db)
    db.close
    return data

def orderSum(r, type):
    try:
        r = r['orderBooks'][0]['orderBook'][type]
        return sum([float(r['price'])*float(r['quantity']) for r in r])
    except:
        return 0

"""
def exchangeFullOrder(depth):
    r = requests.get('https://dev-api.shrimpy.io/v1/orderbooks?exchange=kucoin&quoteSymbol=all,USDT&limit='+str(depth)).text
    r = json.loads(r)
    r = [{'coin':r['baseSymbol']+'-USDT', 'ask':orderSum(r, 'asks'), 'bid':orderSum(r, 'bids')} for r in r if len(r['baseSymbol'])>0]
    return pd.DataFrame(r)
"""

pd.options.mode.chained_assignment = None
app = Flask(__name__)

updateInterval = 60

def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)

def fulltable():
    if True: #is_time_between(datetime.time(8,35), datetime.time(23,35)):
        raw = requests.get(r'http://www.tsetmc.com/tsev2/data/MarketWatchInit.aspx?h=0&r=0', timeout=None, verify=False).text
        raw = raw.replace('\u200c','').split(',@')[1].split(';')
        raw = [raw.split(',') for raw in [ x for x in raw if "/" not in x ]]

        client = [c.split(',') for c in (requests.get(r'http://www.tsetmc.com/tsev2/data/ClientTypeAll.aspx', timeout=None, verify=False).text).split(';')]

        df = {}

        for i in range(len(raw)):
            df8 = []
            cl = []
            if len(raw[i]) == 25:
                if not any(char.isdigit() for char in (raw[i][2])):
                    for j in range(5,22):
                        if raw[i][j] == '':
                            raw[i][j] = 0
                    for k in range(len(raw)):
                        if raw[i][0] in raw[k][0] and hasNumbers(raw[k][2]):
                            df8.append(raw[k])
                    df8 = sorted(df8)
                    #fill the rest with zero
                    if len(df8) < 5:
                        for k in range(5-len(df8)):
                            df8.append([0,0,0,0,0,0,0,0])
                    for k in range(len(client)):
                        if raw[i][0] in client[k][0]:
                            cl.append(client[k])
                    if len(cl) == 0:
                        cl.append([0,0,0,0,0,0,0,0,0])
                    df[i] = {'id':raw[i][0], 'name':raw[i][2], 'open':raw[i][5], 'adj':raw[i][6], 'close':raw[i][7], 'ticks':raw[i][8], 'volume':raw[i][9], 'value':raw[i][10], 'low':raw[i][11], 'high':raw[i][12], 'yesterday':raw[i][13], 'eps':raw[i][14], 'bvol':raw[i][15], 'unk1':raw[i][16], 'bazar':raw[i][17], 'group':raw[i][18], 'tmax':raw[i][19], 'tmin':raw[i][20], 'shares':raw[i][21], 'type':raw[i][22],'tf1':df8[0][2],'hf1':df8[0][7],'gf1':df8[0][5],'gk1':df8[0][4],'hk1':df8[0][6],'tk1':df8[0][3],'tf2':df8[1][2],'hf2':df8[1][7],'gf2':df8[1][5],'gk2':df8[1][4],'hk2':df8[1][6],'tk2':df8[1][3],'tf3':df8[2][2],'hf3':df8[2][7],'gf3':df8[2][5],'gk3':df8[2][4],'hk3':df8[2][6],'tk3':df8[2][3],'tf4':df8[3][2],'hf4':df8[3][7],'gf4':df8[3][5],'gk4':df8[3][4],'hk4':df8[3][6],'tk4':df8[3][3],'tf5':df8[4][2],'hf5':df8[4][7],'gf5':df8[4][5],'gk5':df8[4][4],'hk5':df8[4][6],'tk5':df8[4][3],'tkha':cl[0][1],'tkho':cl[0][2],'hkha':cl[0][3],'hkho':cl[0][4],'tfha':cl[0][5],'tfho':cl[0][6],'hfha':cl[0][7],'hfho':cl[0][8]}

        df = pd.DataFrame.from_dict(df, orient='index')
        df = df[~df.apply(lambda series: series.astype(str).str.contains('@')).any(axis=1)]
        df['bvol'] = [max(float(df[i:i+1]['bvol']), 15000000000/float(df[i:i+1]['close'])) for i in range(len(df))]
        #df['eps'] = [max(float(df[i:i+1]['eps']), 0) for i in range(len(df))]
        df.iloc[:, 2:] = (df.iloc[:, 2:].astype('float')).astype('int')
        db_write('db', df, 'fulltable')
    return None

def index():
    r = requests.get('http://tsetmc.ir/Loader.aspx?ParTree=15131J&i=32097828799138957')
    df_list = pd.read_html(r.text) # this parses all the tables in webpages to a list
    df = df_list[4]
    df.columns = ['time', 'value', 'percent', 'low', 'high']
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] > datetime.datetime.combine(df['time'][1].date(), datetime.time(8,55,1))) & (df['time'] < datetime.datetime.combine(df['time'][1].date(), datetime.time(12,30,1)))]
    df['time'] = df['time'].astype('int')
    db_write('db', df, 'tsetmc_index')
    return None

def get_tgju():
    r = requests.get('https://call2.tgju.org/ajax.json?' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    r = r.json()
    list = ['sekee', 'sekeb', 'nim', 'rob', 'gerami', 'price_dollar_rl', 'crypto-bitcoin', 'sekeb_blubber']
    df = [{'symbol':i,'percent':r['current'][i]['dp'], 'sign':r['current'][i]['dt'], 'price':float(r['current'][i]['p'].replace(',', ''))} for i in list]
    df = pd.DataFrame(df)
    db_write('db', df, 'tgju')
    return print('wrote tgju')

def analyzer(row):
    if row['low']>row['yesterday'] and row['hkha']/(row['tkha']+1)>row['hfha']/(row['tfha']+1) and (row['hfho']-row['hkho']<2*row['bvol']):
        return 1
    elif (row['high']<row['yesterday'] and row['hkha']/(row['tkha']+1)<row['hfha']/(row['tfha']+1)) or (row['hfho']-row['hkho']>2*row['bvol']):
        return -1
    else:
        return 0

def buy(row, df):
    if (row['hkho']+row['hkha'])/row['bvol']>sum((df['hkho']+df['hkha'])/df['bvol'])/len(df['bvol']) and row['hkho']<row['hfho'] and row['hkha']/(row['tkha']+1)>row['hfha']/(row['tfha']+1):
        return True
    else:
        False

def calall():

    if True:#is_time_between(datetime.time(8,35), datetime.time(23,35)):

        df = db_read('db', 'fulltable')

        df = df[(df['type']<=309) & (df['group'] != 68)]

        haToHo = sum((df['hkho']-df['hfho'])*df['close'])

        pl_positive = len(df[(df['close']-df['yesterday'])>0])/(len(df['yesterday'])+1)*100
        pc_positive = len(df[(df['adj']-df['yesterday'])>0])/(len(df['yesterday'])+1)*100

        tgju = db_read('db', 'tgju')

        dollar = tgju[tgju['symbol'] == 'price_dollar_rl']['price'].values[0]/10
        dollar_pl = tgju[tgju['symbol'] == 'price_dollar_rl']['percent'].values[0] if tgju[tgju['symbol'] == 'price_dollar_rl']['sign'].values[0] == 'high' else -1*float(tgju[tgju['symbol'] == 'price_dollar_rl']['percent'].values[0])

        bitcoin = tgju[tgju['symbol'] == 'crypto-bitcoin']['price'].values[0]
        bitcoin_pl = tgju[tgju['symbol'] == 'crypto-bitcoin']['percent'].values[0] if tgju[tgju['symbol'] == 'crypto-bitcoin']['sign'].values[0] == 'high' else -1*float(tgju[tgju['symbol'] == 'crypto-bitcoin']['percent'].values[0])

        shakhes = db_read('db', 'tsetmc_index')
        shClose = shakhes['value'][len(shakhes['value'])-1]
        shPc = shakhes['percent'][len(shakhes['value'])-1]

        if '(' in str(shPc):
            shPc = shPc.replace("(", "")
            shPc = shPc.replace(")", "")
            shPc = float("{:.2f}".format(float(shPc)))*-1
        else:
            shPc = float("{:.2f}".format(float(shPc)))

        shakhesTime = shakhes['time'].tolist()
        shakhesValue = shakhes['value'].tolist()
        json.dumps(shakhesTime)
        json.dumps(shakhesValue)

        dfk = pd.DataFrame({'shakhes':shClose, 'shPc':shPc, 'dollar': dollar, 'dollar_pl':dollar_pl}, index=[0])

        dfk['ak'] = sum([(row['hk1']+row['hk2']+row['hk3']+row['hk4']+row['hk5'])/(row['tk1']+row['tk2']+row['tk3']+row['tk4']+row['tk5']+1)*row['close']/10000000 for i,row in df.iterrows()])/len(df['bvol'])
        dfk['af'] = sum([(row['hf1']+row['hf2']+row['hf3']+row['hf4']+row['hf5'])/(row['tf1']+row['tf2']+row['tf3']+row['tf4']+row['tf5']+1)*row['close']/10000000 for i,row in df.iterrows()])/len(df['bvol'])
        dfk['akw'] = sum([(row['hk1']+row['hk2']+row['hk3']+row['hk4']+row['hk5'])/(row['tk1']+row['tk2']+row['tk3']+row['tk4']+row['tk5']+1)*row['close']/10000000*row['shares'] for i,row in df.iterrows()])/sum(df['shares'])
        dfk['afw'] = sum([(row['hf1']+row['hf2']+row['hf3']+row['hf4']+row['hf5'])/(row['tf1']+row['tf2']+row['tf3']+row['tf4']+row['tf5']+1)*row['close']/10000000*row['shares'] for i,row in df.iterrows()])/sum(df['shares'])

        dfk['bid'] = sum([(row['hk1']+row['hk2']+row['hk3']+row['hk4']+row['hk5'])*row['close']/10000000000 for i,row in df.iterrows()])
        dfk['ask'] = sum([(row['hf1']+row['hf2']+row['hf3']+row['hf4']+row['hf5'])*row['close']/10000000000 for i,row in df.iterrows()])

        dfk['akha'] = sum([row['hkha']/(row['tkha']+1)*row['close']/10000000 for i,row in df.iterrows()])/len(df['bvol'])
        dfk['afha'] = sum([row['hfha']/(row['tfha']+1)*row['close']/10000000 for i,row in df.iterrows()])/len(df['bvol'])
        dfk['akhaw'] = sum([row['hkha']/(row['tkha']+1)*row['close']/10000000*row['shares'] for i,row in df.iterrows()])/sum(df['shares'])
        dfk['afhaw'] = sum([row['hfha']/(row['tfha']+1)*row['close']/10000000*row['shares'] for i,row in df.iterrows()])/sum(df['shares'])

        tsetmc_market_cap = sum([row['shares']*row['adj'] for i,row in df.iterrows()])/dollar/10/1000000000

        df['kkk'] = [analyzer(row) for i,row in df.iterrows()]
        dfk['hatoho'] = sum([(row['hkho']-row['hfho'])*row['close']/10000000000 for i,row in df.iterrows()])
        db_write('db', dfk, 'calculated_dfk')
        positive = len(df[df['kkk'] == 1])/len(df['bvol'])
        neutral = len(df[df['kkk'] == 0])/len(df['bvol'])
        negative = len(df[df['kkk'] == -1])/len(df['bvol'])

        dfm = pd.DataFrame({'pl_positive':pl_positive, 'pc_positive':pc_positive, 'haToHo':haToHo, 'bitcoin':bitcoin, 'bitcoin_pl':bitcoin_pl, 'dollar':dollar, 'dollar_pl':dollar_pl, 'positive':positive, 'negative':negative, 'neutral':neutral, 'tsetmc_market_cap':tsetmc_market_cap}, index=[0])

        df['adjp'] = (df['adj']/df['yesterday']-1)*100
        df['closep'] = (df['close']/df['yesterday']-1)*100
        df['span'] = [1 if row['tmax']>row['close'] and row['adj']>row['yesterday'] else 0 for i,row in df.iterrows()]
        df['hatoho'] = (df['hkho']-df['hfho'])*df['close']/10000000000
        df['hkhopbvol'] = (df['hkho']-df['hfho'])/df['bvol']
        df['hkhapbvol'] = df['hkha']/df['bvol']
        df['saf'] = ((df['hk1']*df['gk1']+df['hk2']*df['gk2']+df['hk3']*df['gk3']+df['hk4']*df['gk4']+df['hk5']*df['gk5'])-(df['hf1']*df['gf1']+df['hf2']*df['gf2']+df['hf3']*df['gf3']+df['hf4']*df['gf4']+df['hf5']*df['gf5']))/10000000000
        df['safpower'] = [(((df['hk1']/(df['tk1']+1)+df['hk2']/(df['tk2']+1)+df['hk3']/(df['tk3']+1)+df['hk4']/(df['tk4']+1)+df['hk5']/(df['tk5']+1)))-((df['hf1']/(df['tf1']+1)+df['hf2']/(df['tf2']+1)+df['hf3']/(df['tf3']+1)+df['hf4']/(df['tf4']+1)+df['hf5']/(df['tf5']+1))))*df['close']/10000000 for i,df in df.iterrows()]
        df['kp'] = [(df['hkha']/(1+df['tkha'])-df['hfha']/(1+df['tfha']))*df['close']/10000000 for i, df in df.iterrows()]
        df['kha'] = df['hkha']/(df['tkha']+1)/10000000*df['close']
        df['fha'] = df['hfha']/(df['tfha']+1)/10000000*df['close']
        df['kho_per_shares'] = (df['hkho']-df['hfho'])/(df['shares']+1)*100
        df['hk_per_shares'] = (df['hkho']+df['hkha'])/(df['shares']+1)*100
        df['daily_value'] = (df['hkho']+df['hkha'])*df['close']/10000000000
        df['totall_value'] = df['shares']*df['adj']//10000000000
        df = df.sort_values(by=['totall_value'],ignore_index=True, ascending=True)
        df['volpbvol'] = (df['hkha']+df['hkho'])/df['bvol']
        df['risk'] = (df['tmin']/df['yesterday']*df['adj'] / df['close'] - 1) * 100
        df['profit'] = (df['tmax']/df['yesterday']*df['adj'] / df['close'] - 1) * 100
        df['x'] = df['bvol']/df['volume']

        dfn = df[([buy(row, df) for i,row in df.iterrows()]) & (df['hf1'] != 0) & (df['gf1'] <= df['tmax'])]

        df = df[['open','close', 'high', 'low', 'tmax', 'tmin','adjp','closep', 'span', 'hatoho', 'hkhopbvol', 'hkhapbvol', 'saf', 'safpower', 'kp', 'kha',\
            'fha', 'kho_per_shares', 'hk_per_shares', 'daily_value', 'totall_value', 'volpbvol', 'id', 'name', 'group', 'risk', 'profit', 'x']]

        db_write('db', df, 'calculated_fulltable')

        db_write('db', dfn, 'calculated_dfn')
        db_write('db', dfm, 'calculated_dfm')

    return None

def calOrder():

    df = kuCoin.market()['ticker']
    def short_cal(r):
        try:
            return {'coin':r['symbol'], 'volume':float(r['vol']), 'volVal':float(r['volValue']), 'change':float(r['changeRate']), 'volatility':(float(r['high'])/(float(r['low'])+1)-1)*100, 'last':float(r['last'])}
        except:
            return {'coin':r['symbol'], 'volume':float('NaN'), 'volVal':float('NaN'), 'change':float('NaN'), 'volatility':float('NaN'), 'last':float('NaN')}
    df = pd.DataFrame([short_cal(r) for r in df if '-USDT' in r['symbol']])

    #df10 = exchangeFullOrder(10)
    #df10['bpa'] = df10['bid']/(df10['ask']+df10['bid'])*100
    #df = pd.merge(df, df10, on='coin', how='outer')

    df = df.dropna()

    symbols = list(set([r.split('3')[0] for r in [r for r in kuCoin.symbolsList('USDT') if ('3' in r) and ('API' not in r)]]))

    temp = []
    for coin in symbols:
      try:
        temp.append({'coin':coin+'-USDT', 'ratio':coinglass.longshort_ratio(coin)})
      except:
        pass

    temp = pd.DataFrame(temp)
    dft = pd.merge(df, temp, on='coin', how='left')

    df_coingecko = db_read('db', 'coingecko_market')
    dfn = pd.merge(dft, df_coingecko, on='coin', how='left')
    df = dfn.drop_duplicates(subset='coin', keep="last")
    df['image'] = ['https://assets.coingecko.com/coins/images/19691/small/GMC_logo_200x200.png?1635749466' if pd.isna(r) else r for r in df['image']]
    df = df.fillna(0)
    df['rank'] = df['market_cap'].rank(ascending=False)
    df['dominance'] = df['market_cap']/sum(df['market_cap'])*100

    df = pd.merge(df, messari(), on='coin', how='left')
    df['athc'] = (df['ath']/df['last']-1)*100
    df['clc'] = (df['last']/df['cycleLow']-1)

    db_write('db', df, 'exchangeFullOrder', ifExists='replace')

    #bid10 = sum(df10['bid'])
    #ask10 = sum(df10['ask'])

    #crypto_info_table = pd.DataFrame({'bid10':bid10, 'ask10':ask10, 'bpa10':bid10/(bid10+ask10)*100}, index=[0])

    #db_write('db', crypto_info_table, 'crypto_info_table', ifExists='replace')
    return None

def all_assets_market_cap():

    def to_numeric(r):
        if 'T' in r:
            return float(re.sub('[^0-9^.]','', r)) * 10**12
        elif 'B' in r:
            return float(re.sub('[^0-9^.]','', r)) * 10**9
        else:
            return float(re.sub('[^0-9^.]','', r))

    r = requests.get('https://8marketcap.com/').text
    all_assets_market_cap = pd.read_html(r)[0]
    all_assets_market_cap = all_assets_market_cap.drop(all_assets_market_cap.columns[[0, -1]], axis=1)
    all_assets_market_cap.columns = ['rank', 'name', 'symbol', 'market_cap', 'price', '24h%', '7d%']
    all_assets_market_cap['market_cap'] = [to_numeric(r) for r in all_assets_market_cap['market_cap']]
    bitcoin_price = to_numeric(all_assets_market_cap[all_assets_market_cap['name'] == 'Bitcoin']['price'].values[0])
    bitcoin_market_cap = all_assets_market_cap[all_assets_market_cap['name'] == 'Bitcoin']['market_cap'].values[0]
    all_assets_market_cap['bitcoin_price'] = [r/bitcoin_market_cap*bitcoin_price for r in all_assets_market_cap['market_cap']]
    eth_price = to_numeric(all_assets_market_cap[all_assets_market_cap['name'] == 'Ethereum']['price'].values[0])
    eth_market_cap = all_assets_market_cap[all_assets_market_cap['name'] == 'Ethereum']['market_cap'].values[0]
    all_assets_market_cap['eth_price'] = [r/eth_market_cap*eth_price for r in all_assets_market_cap['market_cap']]
    all_assets_market_cap['dominance'] = [r/sum(all_assets_market_cap['market_cap'])*100 for r in all_assets_market_cap['market_cap']]
    db_write('db', all_assets_market_cap, 'all_assets_market_cap', ifExists='replace')
    return None

recurring_scheduler.add_job(target=index, period_in_seconds=updateInterval, job_name="index")
recurring_scheduler.add_job(target=fulltable, period_in_seconds=updateInterval, job_name="fulltable")
recurring_scheduler.add_job(target=get_tgju, period_in_seconds=60, job_name="get_tgju")
recurring_scheduler.add_job(target=calall, period_in_seconds=updateInterval, job_name="calall")
recurring_scheduler.add_job(target=calOrder, period_in_seconds=60, job_name="calOrder")
recurring_scheduler.add_job(target=all_assets_market_cap, period_in_seconds=60, job_name="all_assets_market_cap")
recurring_scheduler.add_job(target=coingecko.market, period_in_seconds=60, job_name="coingecko.market")

recurring_scheduler.job_summary()
recurring_scheduler.run()
recurring_scheduler.job_summary()

@app.route('/')
def main():
    return render_template('index.html', updateInterval=updateInterval)

@app.route('/tsetmc_full_table/')
def tsetmc_full_table():
    return render_template('tsetmc_full_table.html', updateInterval=updateInterval)

@app.route('/crypto_full_table/')
def crypto_full_table():
    return render_template('crypto_full_table.html', updateInterval=updateInterval)

@app.route('/test')
def test():
    return render_template('test.html', updateInterval=updateInterval)

@app.route('/test2')
def test2():
    return render_template('test2.html', updateInterval=updateInterval)

@app.route('/visualize/<int:number>/')
def visualize(number):

    df = db_read('db', 'calculated_fulltable', howMany=number, sort=True, sort_on='totall_value')

    return render_template('visualize.html', df=df)

@app.route('/kucoin')
def kucoin():
    a = 4
    df = db_read('db', 'kuCoinData')

    temp = pd.DataFrame()
    for column in df.columns:
        temp = temp.append(df.sort_values(column, ascending = False)[:40])
    df = temp.drop_duplicates()

    return render_template('kucoin.html', kuCoinData=df, a=a)

"""
@app.route('/exchangeFullOrder/<int:number>/')
def exfullorder(number):
    df = db_read('db', 'exchangeFullOrder', howMany=number, sort=True, sort_on='market_cap')
    total = sum(df['volVal'])
    return render_template('exchangeFullOrder.html', df=df, total=total)


@app.route('/crypto_info_table')
def crypto_info_table():
    crypto_info_table = db_read('db', 'crypto_info_table')

    return render_template('crypto_info_table.html', crypto_info_table=crypto_info_table)
"""

@app.route('/all_assets_market_cap/<int:number>/')
def _all_assets_market_cap(number):
    all_assets_market_cap = db_read('db', 'all_assets_market_cap', howMany=number, sort=True, sort_on='market_cap')

    return render_template('all_assets_market_cap.html', df=all_assets_market_cap)

@app.route('/tsetmc_top_menu')
def tsetmc_top_menu():

    shakhes = db_read('db', 'tsetmc_index')

    dfm = db_read('db', 'calculated_dfm')
    dfk = db_read('db', 'calculated_dfk')
    return render_template('tsetmc_top_menu.html', **locals())

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
