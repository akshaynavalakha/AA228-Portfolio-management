import pandas as pd
import numpy as np
from random import *
import matplotlib.pyplot as plt

###########################################################################
## Helper Training Function
##########################################################################
def train_q(td, v_start):

    c = np.unique(td.stock_index.values)
    Q_update = pd.DataFrame(index=c, columns=action_set, data=0.0)
    learning_rate = 0.5
    discount_factor = 0.9
    sample_size = len(td)
    loops = 8000
    for i in range(loops):
        v = v_start   # random value taken to train the data
        for j in range(1,sample_size):
            rs_past = td.stock_index.values[j-1]
            rs_current = td.stock_index.values[j]
            rb_current = td.bond.values[j]
            b = randint(1, 100)
            # epsilon greedy selection of current action
            if b > 75:
                a1 = sample(action_set, 1)
                a = a1[0]
            else:
                a = Q_update.loc[rs_past, :].idxmax()

            returns = a*rs_current + (1-a)*rb_current
            rt = v*returns

            Q_update.loc[rs_past,a] = Q_update.loc[rs_past,a] + learning_rate*(rt + discount_factor*max(Q_update.loc[rs_current,:]) - Q_update.loc[rs_past,a])
    return Q_update


##############################################################################
##               Main Loop
##############################################################################

## Read the file
input_table = pd.read_csv("asset_ret.csv")
action_set = [0.0, 1.0] # is the weight of stocks

#The first 50 years are used for training
################################################
## training
##############################################
train_data = input_table.iloc[0:51, 0:3]
train_data['stock_index'] = train_data['stock'].round(2)  # Discretize the statespace
v_train = 20
v_threshold_train = 1000
annual_contribution_train = 20

q_table = train_q(train_data, v_train) #Q values after training
q_table.to_csv("policy1.csv")
###################################
## Test 1
## Comapring the portfolio value with the benchmark having a 60-40 distribution
## For 10 years beginning immediately after training
##################################

v = np.arange(1,121,1)
portfolio = pd.Series(index=v)
portfolio_benchmark = pd.Series(index=v)
improvement = pd.Series(index=v)
years = 10
v_threshold = 100
annual_contribution = 1.8

#Test 1
#q = pd.read_csv("C:\\Users\\anavalakha\\Downloads\\out.csv")
#q_table =q.set_index('r_index')
for v_start in v:
    value = v_start
    value_benchmark = v_start
    test_data = input_table.iloc[60:61+years, 0:3]
    test_data['stock_index'] = round(test_data['stock'],2)
    for i in range(1,len(test_data)):
            rs = test_data.stock_index.values[i-1]
            train_data['min_stock'] = abs(train_data['stock_index'] - rs)
            ### Nearest Neighbor
            if rs in train_data['stock_index'].values:
                rstock_index = rs
            else:
                rstock_index = train_data.stock_index[train_data['min_stock'].idxmin()]

            action = float(q_table.loc[rstock_index, :].idxmax())
            profit = action * test_data.stock.values[i] + (1 - action) * test_data.bond.values[i]
            value = value * (1 + profit) + annual_contribution
            profit_benchmark = 0.6*test_data.stock.values[i] + (1 - 0.6) * test_data.bond.values[i]
            value_benchmark = value_benchmark * (1 + profit_benchmark) + annual_contribution
    portfolio[v_start] = value
    portfolio_benchmark[v_start] = value_benchmark
    improvement[v_start] = value/value_benchmark -1

    print("the portfolio value is " ,value )
    print("the benchmark value is " ,value_benchmark)

print("The average improvement is ", improvement.mean())

plt.plot(v, portfolio.values, 'r', label='Q learning ')
plt.plot(v, portfolio_benchmark.values, 'b', label = "benchmark")
plt.ylabel('portfolio value')
plt.xlabel('initial value')
plt.legend(loc='best')
plt.title('portfolio value for 10 years investing period')
plt.show()


############################
###Test 2
## Extracting the policy for the next 30 year time frame
###########################
#Location fo year 1986 is 58 and location of year 2016 is 88
# importing in test_set 2
test_set_2 = input_table.iloc[58:89,0:3]
test_set_2['stock_index'] = round(test_set_2['stock'], 2)
horizon = np.arange(0,31)
starting_time = np.arange(0,31)
years = np.arange(1986,2017)
policy_table = pd.Series(index=years, data='x')
for start_time in starting_time:
        rs = test_set_2.stock_index.values[start_time]
        train_data['min_stock'] = abs(train_data['stock_index'] - rs)
        if rs in train_data['stock_index'].values:
            rstock_index = rs
        else:
            rstock_index = train_data.stock_index[train_data['min_stock'].idxmin()]

        action = float(q_table.loc[rstock_index, :].idxmax())
        policy_table[start_time + 1986] = action
print(policy_table)
policy_table.to_csv("policy.csv")

############################################################
### Test 3
# For 4 10 year period calculate portfolio with Q learning and benchmark
## plot the improvement in performance on a graph
###############################################################

data_set_1 = input_table.iloc[51:61, 0:3] # year 1979 - 1988
data_set_2 = input_table.iloc[61:71,0:3] # year 1989 - 1998
data_set_3 = input_table.iloc[71:81, 0:3] # year 1999-2008
data_set_4 = input_table.iloc[81:89, 0:3] # year 2009:2016

portfolio_1 = pd.Series(index=v)
portfolio_benchmark_1 = pd.Series(index=v)
percentage_change_1 = pd.Series(index=v)

portfolio_2 = pd.Series(index=v)
portfolio_benchmark_2 = pd.Series(index=v)
percentage_change_2 = pd.Series(index=v)

portfolio_3 = pd.Series(index=v)
portfolio_benchmark_3 = pd.Series(index=v)
percentage_change_3 = pd.Series(index=v)

portfolio_4 = pd.Series(index=v)
portfolio_benchmark_4 = pd.Series(index=v)
percentage_change_4 = pd.Series(index=v)

#############
# Period 1
##############
for v_start in v:
    value = v_start
    value_benchmark = v_start
    test_data = data_set_1
    test_data['stock_index'] = round(test_data['stock'],2)
    for i in range(1,len(test_data)):
            rs = test_data.stock_index.values[i-1]
            train_data['min_stock'] = abs(train_data['stock_index'] - rs)
            if rs in train_data['stock_index'].values:
                rstock_index = rs
            else:
                rstock_index = train_data.stock_index[train_data['min_stock'].idxmin()]

            action = float(q_table.loc[rstock_index, :].idxmax())
           # print("the action is ",action)
            profit = action * test_data.stock.values[i] + (1 - action) * test_data.bond.values[i]
            value = value * (1 + profit) + annual_contribution
            profit_benchmark = 0.6*test_data.stock.values[i] + (1 - 0.6) * test_data.bond.values[i]
            value_benchmark = value_benchmark * (1 + profit_benchmark) + annual_contribution
    portfolio_1[v_start] = value
    portfolio_benchmark_1[v_start] = value_benchmark
    percentage_change_1[v_start] = value/value_benchmark -1
#############
# Period 2
##############

for v_start in v:
    value = v_start
    value_benchmark = v_start
    test_data = data_set_2
    test_data['stock_index'] = round(test_data['stock'],2)
    for i in range(1,len(test_data)):
            rs = test_data.stock_index.values[i-1]
            train_data['min_stock'] = abs(train_data['stock_index'] - rs)
            if rs in train_data['stock_index'].values:
                rstock_index = rs
            else:
                rstock_index = train_data.stock_index[train_data['min_stock'].idxmin()]

            action = float(q_table.loc[rstock_index, :].idxmax())
           # print("the action is ",action)
            profit = action * test_data.stock.values[i] + (1 - action) * test_data.bond.values[i]
            value = value * (1 + profit) + annual_contribution
            profit_benchmark = 0.6*test_data.stock.values[i] + (1 - 0.6) * test_data.bond.values[i]
            value_benchmark = value_benchmark * (1 + profit_benchmark) + annual_contribution
    portfolio_2[v_start] = value
    portfolio_benchmark_2[v_start] = value_benchmark
    percentage_change_2[v_start] = value / value_benchmark - 1

#############
# Period 3
##############

for v_start in v:
    value = v_start
    value_benchmark = v_start
    test_data = data_set_3
    test_data['stock_index'] = round(test_data['stock'],2)
    for i in range(1,len(test_data)):
            rs = test_data.stock_index.values[i-1]
            train_data['min_stock'] = abs(train_data['stock_index'] - rs)
            if rs in train_data['stock_index'].values:
                rstock_index = rs
            else:
                rstock_index = train_data.stock_index[train_data['min_stock'].idxmin()]

            action = float(q_table.loc[rstock_index, :].idxmax())
           # print("the action is ",action)
            profit = action * test_data.stock.values[i] + (1 - action) * test_data.bond.values[i]
            value = value * (1 + profit) + annual_contribution
            profit_benchmark = 0.6*test_data.stock.values[i] + (1 - 0.6) * test_data.bond.values[i]
            value_benchmark = value_benchmark * (1 + profit_benchmark) + annual_contribution
    portfolio_3[v_start] = value
    portfolio_benchmark_3[v_start] = value_benchmark
    percentage_change_3[v_start] = value / value_benchmark - 1

#############
# Period 4
##############

for v_start in v:
    value = v_start
    value_benchmark = v_start
    test_data = data_set_4
    test_data['stock_index'] = round(test_data['stock'],2)
    for i in range(1,len(test_data)):
            rs = test_data.stock_index.values[i-1]
            train_data['min_stock'] = abs(train_data['stock_index'] - rs)
            if rs in train_data['stock_index'].values:
                rstock_index = rs
            else:
                rstock_index = train_data.stock_index[train_data['min_stock'].idxmin()]

            action = float(q_table.loc[rstock_index, :].idxmax())
           # print("the action is ",action)
            profit = action * test_data.stock.values[i] + (1 - action) * test_data.bond.values[i]
            value = value * (1 + profit) + annual_contribution
            profit_benchmark = 0.6*test_data.stock.values[i] + (1 - 0.6) * test_data.bond.values[i]
            value_benchmark = value_benchmark * (1 + profit_benchmark) + annual_contribution
    portfolio_4[v_start] = value
    portfolio_benchmark_4[v_start] = value_benchmark
    percentage_change_4[v_start] = value / value_benchmark - 1


plt.plot(v, percentage_change_1,  label='Period 1 1979-1988 ')
plt.plot(v, percentage_change_2,  label='Period 2 1989-1998 ')
plt.plot(v, percentage_change_3,  label='Period 3 1999-2008 ')
plt.plot(v, percentage_change_4,  label='Period 4 2009-2016 ')
plt.ylabel('Improvement over benchmark')
plt.xlabel('initial value')
plt.legend(loc='best')
plt.title('Comparision of porfolio value for different benchmarks')
plt.show()



