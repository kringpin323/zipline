#-*- coding: utf-8 -*-
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
from zipline.finance.trading import TradingEnvironment
from zipline.api import order, record, symbol, history
from zipline.finance import trading
from zipline.utils.factory import create_simulation_parameters
import pandas as pd
import numpy as np

n = 0
# Define algorithm
def initialize(context):
    context.asset = symbol('AAPL')
    print "initialization"
    pass
 
def handle_data(context, data):
    global n
    print "handle", n
    print data.history(context.asset, 'price', 1, '1d')#close price
    # print history(1, '1d', 'price').mean()
    n += 1

    order(symbol('AAPL'), 10)
    # print data.current(symbol('AAPL'), 'close')
    # print data.current(symbol('AAPL'), 'price')
    # exit(1)
    record(AAPL=data.current(symbol('AAPL'), 'price'))
 
def analyze(context=None, results=None):
    import matplotlib.pyplot as plt
    # Plot the portfolio and asset data.
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    results.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL price (USD)')
 
    # Show the plot.
    plt.gcf().set_size_inches(18, 8)
    plt.show()
 
# 本地化工作开始
def load_t(trading_day, trading_days, bm_symbol):
    # dates = pd.date_range('2001-01-01 00:00:00', periods=365, tz="Asia/Shanghai")
    bm = pd.Series(data=np.random.random_sample(len(trading_days)), index=trading_days)
    tr = pd.DataFrame(data=np.random.random_sample((len(trading_days), 7)), index=trading_days,
                      columns=['1month', '3month', '6month', '1year', '2year', '3year', '10year'])
    return bm, tr

trading.environment = TradingEnvironment(load=load_t, bm_symbol='^HSI', exchange_tz='Asia/Shanghai')
 
# 回测参数设置
sim_params = create_simulation_parameters(year=2016,
    start=pd.to_datetime("2016-06-01 00:00:00").tz_localize("Asia/Shanghai"),
    end=pd.to_datetime("2016-12-28 00:00:00").tz_localize("Asia/Shanghai"),
    data_frequency="daily", emission_rate="daily")  # 原始版本是上面这样的，代码里面是交易日历，然而，如何产生交易日历呢？

 
# setting the algo parameters
algor_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
                             sim_params=sim_params, env=trading.environment, analyze=analyze
                             )
# algor_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, analyze=analyze)
# data format definition
parse = lambda x: datetime.date(datetime.strptime(x, '%Y-%m-%d'))

# data generator
data_s = pd.read_csv('EOD-AAPL.csv', parse_dates=['Date'], index_col=0, date_parser=parse)
# print data_s.keys
# print data_s.index
print data_s.columns
data_s.columns = data_s.columns.str.lower()
print data_s.columns
# print data_s.head(5)
data_s = data_s.sort_index(axis=0,ascending=True)
# data_s = data_s.sort_values(by='Date', ascending=True)
print data_s.head(5)
# exit(1)

data_c = pd.Panel({'AAPL': data_s})
# data_c.minor_axis = ['open', 'high', 'low', 'close', 'volume']

# for each in data_c.minor_axis:
    # print each

# print data_c.major_axis

# exit(1)

perf_manual = algor_obj.run(data_c)
# Print
perf_manual.to_csv('myoutput.csv')
