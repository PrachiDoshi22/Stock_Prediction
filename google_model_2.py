
"""
Created on Mon Jun 18 11:27:54 2018

@author: vyomunadkat
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:08:27 2018

@author: vyomunadkat
"""


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

training_set = pd.read_csv('./Google_Stock_Price_Train.csv')

training_set = training_set.iloc[:,1:2].values

#feature scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

training_set_scaled = scaler.fit_transform(training_set)

#creating data with 20 timestamps
# Creating a data structure with 20 timesteps and t+1 output
X_train = []
y_train = []
for i in range(20, 1258):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


"""
#getting the data to train

train_set = training_set[0:1257]
test_set = training_set[1:1258]


#reshape

train_set = np.reshape(train_set, (1257,1,1))"""

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#import keras libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initialising the model

regressor = Sequential()

#adding layers

#1st layer
regressor.add(LSTM(units=4, return_sequences = True, input_shape = (None,1)))
#in input shape, none signifies any number of timestamp and 1 is the no. of feature i.e. opening price in this case

#2nd layer
regressor.add(LSTM(units=4, return_sequences = True))

#3rd layer
regressor.add(LSTM(units=4, return_sequences = True))

#4th layer
regressor.add(LSTM(units=4))



regressor.add(Dense(units = 1))

#compiling
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')

#fitting the training set

regressor.fit(X_train, y_train, batch_size=32, epochs=200)

#getting the actual price to be predicted

actual_price = pd.read_csv('./Google_Stock_Price_Test.csv')

actual_price = actual_price.iloc[:,1:2].values

real_stock_price = np.concatenate((training_set[0:1258], actual_price), axis = 0)


# Getting the predicted stock price of 2017
scaled_real_stock_price = scaler.fit_transform(real_stock_price)
inputs = []
for i in range(1258, 1278):
    inputs.append(scaled_real_stock_price[i-20:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[1258:], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()







"""
#predicting the price using the models

inputs = actual_price
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted_price = regressor.predict(inputs)
predicted_price = scaler.inverse_transform(predicted_price)

#visualising the data
plt.plot(actual_price, color='red', label='actual price')
plt.plot(predicted_price, color='blue', label='predicted price')
plt.title('The google stock prediction model')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

#evaluate and calculate the rmse
from sklearn.metrics import mean_squared_error
import math
rmse = math.sqrt(mean_squared_error(actual_price,predicted_price))"""

