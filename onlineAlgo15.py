'''
# Tenzin Trader - Trading Algorithm
'''
import cbpro
import sys
import json
import time
import os
import pickle
import pandas as pd
import numpy as np
import datetime as dt
# to do: the following libraries are to update the persisted ML model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def random_delay(max_delay):
    rng = np.random.default_rng()
    sec = rng.random()*max_delay
    time.sleep(sec)
    return

def stop_trading(symbol):
    # ends the trading session
    # long positions are held, open orders are closed
    print('ending trading session, max # ticks received')
    # cancel orders
    print('***CANCELING UNFILLED ORDERS***')
    auth_client.cancel_all(product_id=symbol)
    return

def save_datafeed(df, symbol):
    df_save = df.reset_index(drop=True, inplace=False)
    filename = str(symbol)+'_price_series.csv'
    filepath = os.path.join('crypto_data', filename)
    df_save.to_csv(filepath)
    return    

def load_crypto():
    products_df = pd.DataFrame(public_client.get_products())
    symbols = products_df['id']
    
    for symbol in symbols:
        filename = str(symbol)+'_price_series.csv'
        filepath = os.path.join('crypto_data', filename)
        if os.path.exists(filepath):
            continue
        else:
            print('Getting historic rates for: {0}'.format(symbol))
            price_series = public_client.get_product_historic_rates(symbol, granularity=86400)
            if len(price_series) == 0:
                continue
            else:
                price_series_df = pd.DataFrame.from_dict(price_series)
                price_series_df.rename({0:'time', 4:'close'}, axis='columns', inplace=True)
                price_series_df['symbol'] = symbol
                price_series_df.index = pd.to_datetime(price_series_df['time'], unit='s', utc=True)
                price_series_df = price_series_df[['symbol', 'close']]
                price_series_df['time'] = price_series_df.index
                price_series_df.rename(columns={'close':'price'}, inplace=True)
                save_datafeed(price_series_df, symbol)
        # random delay to avoid api throttling
        random_delay(2)
    return symbols

def sample_tick_data(tick, symbol, bar, df_temp):
    try:
        new_entry = [symbol, tick['time'],tick['price']]
        new_entry_series = pd.Series(new_entry, index=['symbol', 'time', 'price'])
        df_temp = df_temp.append(new_entry_series, ignore_index=True)
        df_temp.index = pd.to_datetime(df_temp['time'], infer_datetime_format=True)
        save_datafeed(df_temp, symbol)
        df = df_temp.resample(bar, label='right').last().ffill()
    except (TypeError, ValueError, KeyError, IOError, AttributeError):
        print('***While sampling tick data an error has occurred***')
        print('trading symbol {}'.format(symbol))
        new_entry = [symbol, tick['time'],tick['price']]
        new_entry_series = pd.Series(new_entry, index=[symbol, 'time', 'price'])
        df_temp = df_temp.append(new_entry_series, ignore_index=True)
        df_temp.index = pd.to_datetime(df_temp['time'], infer_datetime_format=True)
        save_datafeed(df_temp, symbol)
        df = df_temp.resample(bar, label='right').last().ffill()
        print('Error did not repeat. Continue trading.')    
    return df

def trade_crypto(amount, minbar, position, df, symbol, algorithm):
    # increment min_bar by length of df
    #print('{} trading will start when BARS > MINIMUM BARS'.format(symbol))
    #print('NUMBER OF BARS: {} |'.format(len(df))+\
    #        'MINIMUM BARS: {}'.format(minbar))
    if len(df) > minbar:
        print('{} trading in progress...'.format(symbol))
        minbar = len(df)

        # data processing and feature preparation
        df['price'] = df['price'].astype('float64')
        df['Returns'] = np.log(df['price']/df['price'].shift(1))
        df['Direction'] = np.where(df['Returns'] > 0, 1, -1)
        # picks relevant points
        features = df['Direction'].iloc[-(lags + 1): -1]
        # necessary reshaping
        features = features.values.reshape(1, -1)
        # generates the signal (+1 or -1)
        signal = algorithm.predict(features)[0]
        # stores trade signal
        df['Position'] = position
        df['Signal'] = signal

        # trading logic
        if position in [0, -1] and signal == 1:
            auth_client.place_market_order(product_id = symbol,
                               side = 'buy', \
                               funds = amount - position*amount)
            position = 1
            print('LONG')

        elif position in [0, 1] and signal == -1:
            auth_client.place_market_order(product_id = symbol,\
            side = 'sell', funds = 2*amount + position*amount)
            position = -1
            print('SHORT')

        else: # no trade
            auth_client.place_market_order(product_id = symbol,\
            side = 'sell', funds = 0 + position*0)
            position = 0
            print('no trade placed')
    
    return minbar, position

# callback function - algo trading minimal working example
# https://en.wikipedia.org/wiki/Minimal_working_example

def trading_mwe(symbols, amount, position, bar, min_bar, trading):
    # loads the persisted trading algorithm objects
    #btc_algo = pd.read_pickle('BTCalgo.pkl')
    algo = pd.read_pickle('algo.pkl')
    
    # initializes position and min_bar variables for each coin
    positions = []
    minbars = []
    i = 0 # counter for lists: positions, minbars
    
    for symbol in symbols:
        positions.append(position)
        minbars.append(min_bar)
    
    if trading == 'Y':
        
        print('trading symbols:')
        
        for symbol in symbols:
            print('{0}'.format(symbol), end=', ')
        
        while True:
            for symbol in symbols:
                filename = str(symbol)+'_price_series.csv'
                filepath = os.path.join('crypto_data', filename)
                if os.path.exists(filepath):
                    df_saved = pd.read_csv(filepath, index_col=0)
                else:
                    df_saved = pd.DataFrame()
                try:
                    tick = auth_client.get_product_ticker(product_id=symbol)    
                except (IOError):
                    return
                # resampling of the tick data
                if 'price' in tick:
                    df = sample_tick_data(tick, symbol, bar, df_saved)
                    # trade crypto
                    trade_coin = trade_crypto(amount, minbars[i], \
                                              positions[i], df, symbol, algo)
                    minbars[i] = trade_coin[0]
                    positions[i] = trade_coin[1]
                else:
                    continue

                i = i + 1
                
                if i == len(symbols):
                    i = 0 # reset counter to loop through symbols list

                print(".",end='')

                '''
                if len(df) > 270:
                    for symbol in symbols:
                        stop_trading(symbol)
                        trading = 'n'
                '''
                # random delay to avoid api throttling
                random_delay(60)
    
    return trading


if __name__ == '__main__':

    # Authentication credentials
    api_key = os.environ.get('CBPRO_KEY')
    api_secret = os.environ.get('CBPRO_SECRET')
    passphrase = os.environ.get('CBPRO_PASSPHRASE')
    
    '''
    # sandbox authenticated client
    #auth_client = cbpro.AuthenticatedClient(api_key, api_secret, passphrase, \
                                            #api_url='https://api-public.sandbox.pro.coinbase.com')
    # live account authenticated client
    # uses a different set of API access credentials (api_key, api_secret, passphrase)
    #auth_client = start_auth_client(api_key, api_secret, passphrase, \
    #                                          api_url='https://api.pro.coinbase.com')
    # public client
    #public_client = start_public_client()

    # parameters for the trading algorithm
    # the trading algorithm runs silently for 500 ticks
    5 min: 300s, 10 min: 600s, 15 min: 900s, 30 min: 1800s, 45 min: 2700s
    1 hr: 3600s, 2hr: 7200s, 3hr: 10800s, 6hr: 21600s, 8hr: 28800s, 12hr: 43200s, 24hr: 86400s, 4 days: 345600s, 28 days: 2419200s

    symbols = ['BTC-USD', 'BTC-EUR', 'BTC-GBP', 'BTC-USDT','ETH-USD', 'ETH-EUR', 'ETH-GBP', 'ETH-BTC', 'ETH-USDT', 'MATIC-USD', 'MATIC-BTC', 'MATIC-EUR', 'MATIC-GBP', 'MIR-USD', 'MIR-BTC', 'MIR-EUR', 'MIR-GBP', 'CGLD-USD', 'CGLD-EUR', 'CGLD-GBP', 'CGLD-BTC', 'FORTH-USD', 'FORTH-EUR', 'FORTH-GBP', 'FORTH-BTC', 'USDT-USD', 'USDT-EUR', 'USDT-GBP']
    '''
    
    bar = '345600s'      # 15s is for testing; reset to trading frequency
    amount = 25.17      # amount to be traded in $USD - $50 minimum
    position = 0        # beginning, neutral, position
    lags = 20           # number of lags for features data

    # minumum number of resampled bars required for the first predicted value (& first trade)
    min_bar = lags + 1
    
    # load and save historical price series data
    public_client = cbpro.PublicClient()
    symbols = load_crypto()
    
    # start authorized client
    auth_client = cbpro.AuthenticatedClient(api_key, api_secret, passphrase, \
                                          api_url='https://api.pro.coinbase.com')
    
    # the main asynchronous loop using the callback function
    # Coinbase Pro web socket connection is rate-limited to 4 seconds per request per IP.

    try:
        while True:
            trading = 'Y'
            print('Saving datafeed. Trading starts after {0} bars are saved.'.format(min_bar))
            print(time.strftime('%Y-%m-%d %H:%M', time.gmtime()))
            trading = trading_mwe(symbols, amount, position, bar, min_bar, trading)
    except (TypeError, ValueError, KeyError, IOError, AttributeError):
        # start public and authorized clients
        public_client = cbpro.PublicClient()
        auth_client = cbpro.AuthenticatedClient\
        (api_key, api_secret,\
         passphrase, api_url='https://api.pro.coinbase.com')
        time.sleep(60) # allow clients to connect
        while True:
            trading = 'Y'
            print('Saving datafeed. Trading starts after {0} bars are saved.'.format(min_bar))
            print(time.strftime('%Y-%m-%d %H:%M', time.gmtime()))
            trading = trading_mwe(symbols, amount, position, bar, min_bar, trading)
