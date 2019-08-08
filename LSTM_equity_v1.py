# NVDA plot and then LSTM predicition based on Jason Brownlee's machine learning
# mastery (herein denoted MLM) multivariate LSTM
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# This is the cleaned up version for github. The development version LSTM_equity_dev1
# is saved elsewhere (not online).
import sys
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
# borrowed from MLM
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
# put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# set run parameters
# the number of rows in your dataset (call it nrows)determined automatically by read_csv
# timesteps rows are eliminated by series_to_supervised, leaving nrows2=nrows-timesteps
# n_train is some fraction, \approx 2/3 say, of these, e.g. nrows=250, timesteps=3, n_train=150.

# features is the number of columns in your original dataset. This code outputs a prediction for
# one sequence, but it can use additional columns of data in that prediction.
# E.g. col1 = closing price of IBM (daily, say),
# col2=IBM daily volume, col3=DJIA, col4=daily temperature in Rio de Janeiro, col5=MMM, col6=GOOG close
# => features=6

# timesteps is the number of input timesteps = number of LSTM cells.
# So what does this mean? You say: "my data has 600 timesteps?!?" why is timesteps not equal to 600?
# This code takes your 600 time steps and creates sub-sequences of length timesteps (e.g. timesteps=3).
# it can only create 597 such sequences because it doesn't e.g. have data at t-3 for t=1. So those
# 597 sequences (or that fraction that go into n_train) are what are used to train your three-step prediction.
# got it?

# finally predict is the number of timesteps ahead to predict. Start with predict=1.

# note, this code is hardwired to use batch_size=1. That makes it generally slow.
# You can fool with that if you want to but no guarantees.

n_train=150
timesteps=5
features=6
predict=1

# load data
dataset = read_csv('NVDA.csv', header=0, index_col=0)

# manually specify column names [generalize this]
# dataset.columns is set to a list with features (=6 here) elements
dataset.columns = ['open','high','low','close','adj close','volume']

# summarize first 5 rows
print(dataset.head(5))

values = dataset.values # this is a numpy array even if numpy not imported
values = values.astype('float32')
print('shape',values.shape)

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, timesteps, predict)

pd.set_option('display.max_columns', 1000)
print('reframed.head',reframed.head())
print('reframed.tail',reframed.tail())

# remove columns that are features which are not to be predicted in the
# final timestep.
# E.g. timestep=3, features=6, x(0),x(6),x(12) t-3,t-2,t-1 resp
# of input data, trying to predict x(18). x(1-5), x(7-11), x(13-17) are
# relevant features. But x(19-23) are irrelevant so drop [19,20,21,22,23]
# reframed.drop(reframed.columns[[19,20,21,22,23]], axis=1, inplace=True)

col_list=list(range(timesteps*features+1,(timesteps+1)*features))
reframed.drop(reframed.columns[col_list], axis=1, inplace=True)
print('reframed.head \n',reframed.head())
print('reframed.tail \n',reframed.tail())

# split into train and test sets
values = reframed.values
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs, for python "-1" is last index.
# ":-1" means all indices except the last one
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
print('train_X.shape',train_X.shape,'train_y.shape',train_y.shape)
print('test_X.shape',test_X.shape,'test_y.shape',test_y.shape)

# reshape input to be 3D [samples, timesteps, features] for LSTM
# cf https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# search "Samples/TimeSteps/Features"
train_X = train_X.reshape((train_X.shape[0], timesteps, features))
test_X = test_X.reshape((test_X.shape[0], timesteps, features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network - note shuffle:=True changes nothing since rows in dataset are independent
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
# pyplot.show()

yhat=model.predict(test_X,batch_size=1)
sum1=0.0
for i in range(0,len(test_y)):
#    print(yhat[i,0],test_y[i])
    sum1=sum1+np.abs(yhat[i,0]-test_y[i])

print('sum1',sum1)
pyplot.figure(2)
pyplot.plot(yhat[:,0],'bo',label='yhat')
pyplot.plot(test_y[:],'r+',label='test_y')
pyplot.legend()
pyplot.show()