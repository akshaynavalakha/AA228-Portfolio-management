#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:34:59 2017

@author: javenxu
"""

# data source: http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html
import pandas as pd
import numpy as np
import math
import scipy as sp

returns = pd.read_csv('asset_ret.csv')

discount_r = returns['stock'].mean()*0.6 + returns['bond'].mean()*0.4

def calc_util(c, eta1, eta2):
    if c > 1:        
        if eta1 == 1:
            return(np.log(c))
        else:
            return((math.pow(c, 1-eta1)-1)/(1-eta1))
    else:
        if eta2 == 1:
            return(np.log(c))
        else:
            return((math.pow(c, 1-eta2)-1)/(1-eta2))

def calc_cumu_ret(ret, v0, c):
    ret_list= ret.tolist()
    for r in ret_list:
        v0 = sp.fv(r, 1, -c, -v0)
    return(v0)

def calc_state_utility(h, ret, w_s, v_init, c=1, v_th=100, eta=1.5):
    dret = ret.copy()
    dret['port_ret'] = dret['stock'] * w_s + dret['bond'] * (1-w_s)
    dret['port_val'] = dret['port_ret'].rolling(window=h).apply(lambda x: calc_cumu_ret(x, v_init, c=c))    
    dret['u'] = (dret['port_val'] / v_th).apply(lambda x: calc_util(x, eta, eta*2)) 
    return(dret['u'].mean())

u_table = pd.DataFrame(index=range(1, 121), columns=range(1, 31))
a_table = pd.DataFrame(index=range(1, 121), columns=range(1, 31))

train_ret = returns.iloc[0:40, 0:3]

for h in range(1, 11):
    for v in range(1, 121):
        policy = pd.Series()
        for s in [t*0.05 for t in range(0, 21)]:
            policy[s] = calc_state_utility(h, train_ret, s, v_init=v, c=1, v_th=100, eta=2)
        policy.max()
        u_table.loc[v,h] = policy.max()
        a_table.loc[v,h] = policy.argmax()

u_table1 = u_table.copy() # eta = 1.5
a_table1 = a_table.copy()
u_table2 = u_table.copy() # eta = 2
a_table2 = a_table.copy()

## visualization
import matplotlib.pyplot as plt
for i in  [1, 2, 5, 7, 10]:
    plt.plot(a_table1[i], label='h='+str(i))
plt.ylabel('stock allocation')
plt.xlabel('initial portfolio value')
plt.legend(loc='best', prop={'size': 8})
plt.title('stock allocation as a function of horizon and init_port_value, eta=1.5')

# test code
# testing periods
test_ret = returns.tail(40)
portval_table = pd.DataFrame(index=test_ret['year'].tolist(), columns=range(1, 21))
benchval_table = pd.DataFrame(index=test_ret['year'].tolist(), columns=range(1, 21))
full_ret = returns.set_index('year')
def test_port(start_year, h, r=discount_r, ws_bench=0.6,
              c=1.8, v_th=100, eta=1.5, lookback=50, full_ret=full_ret):
    v_init = sp.pv(r , h, c, -v_th)
    v = v_init
    v_bench = v_init
    time_remain = h
    i = 0
    while time_remain >= 1:
        train_ret = full_ret.loc[(start_year+i-lookback):(start_year+i-1)]
        policy = pd.Series()
        for s in [t * 0.05 for t in range(0, 21)]:
            policy[s] = calc_state_utility(time_remain, train_ret, s, v_init=v,
                  c=c, v_th=v_th, eta=eta)
        ws = policy.argmax()
        port_ret = ws * full_ret.loc[start_year+i, 'stock'] + (1 - ws) * full_ret.loc[start_year+i, 'bond']
        bench_ret = ws_bench * full_ret.loc[start_year+i, 'stock'] + (1 - ws_bench) * full_ret.loc[start_year+i, 'bond'] 
        # update portfolio and benchmark value
        v = v * (1 + port_ret) + c
        v_bench = v_bench * (1 + bench_ret) + c
        i = i + 1
        time_remain = time_remain -1
    return(v, v_bench)



training_period_mean_ret = returns.head(49).mean()
discount_r = training_period_mean_ret['stock']*0.6 + training_period_mean_ret['bond']*0.4
test_period_start = int(test_ret.iloc[0].year) # starting in year 1977
for start_year in test_ret['year'].tolist():
    start_year = int(start_year)
    for h in range(1, min(21, (2016 - start_year + 2))):
        print(start_year, h)
        (v, v_bench) = test_port(start_year, h, r=discount_r, c=1.5, v_th=100, eta=1.5, lookback=40)
        portval_table.loc[start_year, h] = v
        benchval_table.loc[start_year, h] = v_bench


compare = pd.concat((pd.DataFrame((portval_table > benchval_table).sum()),
             pd.DataFrame((portval_table < benchval_table).sum())), axis=1)

compare.columns = ['proposed_approach_win', 'benchmark_win']

