#!/usr/bin/env python3

from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.metrics import mean_absolute_error as mae
from sklearn.utils import shuffle
import random
import argparse


#-------------------------------------------

def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Individual plant temperature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('-c',
                        '--train',
                        help='csv with all of the weather and trait data',
                        metavar='weather_data',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-t',
                        '--test',
                        help='csv with all of the test data',
                        metavar='test_data',
                        type=str,
                        default=None,
                        required=True)

    # parser.add_argument('-o',
    #                 '--outdir',
    #                 help='Output directory where resulting csv will be saved',
    #                 metavar='str',
    #                 type=str,
    #                 default='Model_Output')

    return parser.parse_args()
#-------------------------------------------

# convert series to supervised learning
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

#-------------------------------------------

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

#-------------------------------------------

def main():
    args = get_args()

    ## Data organization and variable selection ##
    big_csv = pd.read_csv(args.train)
    test_csv = pd.read_csv(args.test)

    del big_csv['Unnamed: 0']
    big_csv['State'] = big_csv['State'].str.replace('"', '')
    test_csv['State'] = test_csv['State'].str.replace('"', '')
    big_csv['date'] = big_csv.groupby('Performance Record').cumcount() + 1
    test_csv['date'] = test_csv.groupby('Performance Record').cumcount() + 1

    del big_csv['AvgSur']
    del big_csv['AP']
    del big_csv['MDNI']
    #del big_csv['Cluster']
    del big_csv['ADNI']
    #del big_csv['State']

    del test_csv['AvgSur']
    del test_csv['AP']
    del test_csv['MDNI']
    #del big_csv['Cluster']
    del test_csv['ADNI']
    #del big_csv['State']

    groups = [big_csv for _, big_csv in big_csv.groupby('Performance Record')]
    random.shuffle(groups)
    groups2 = [test_csv for _, test_csv in test_csv.groupby('Performance Record')]
    random.shuffle(groups2)

    big_csv = pd.concat(groups).reset_index(drop=True)
    test_csv = pd.concat(groups2).reset_index(drop=True)

    big_csv = big_csv.set_index('Performance Record')
    test_csv = test_csv.set_index('Performance Record')

    #dataset = read_csv('LSTM_Model/avg_performance_record.csv', header=0, index_col=0)
    dataset = big_csv
    input_df = dataset[dataset.columns[:-1]]
    output_df = pd.DataFrame(dataset['Yield'])
    
    # Define values that are input and output
    values_input = input_df.values
    values_output = output_df.values
    values_test = test_csv.values

    # Encoding the Sate Column
    encoder = LabelEncoder()
    values_input[:,5] = encoder.fit_transform(values_input[:,5])

    encoder = LabelEncoder()
    values_test[:,6] = encoder.fit_transform(values_test[:,6])

    # Convert values to floats
    values_input = values_input.astype('float32')
    values_output = values_output.astype('float32')
    values_test = values_test.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_input = scaler.fit_transform(values_input)
    scaled_output = scaler.fit_transform(values_output)
    scaled_test = scaler.fit_transform(values_test)

    # frame as supervised learning
    reframed_input = series_to_supervised(scaled_input, 1, 1)
    reframed_output = series_to_supervised(scaled_output, 1, 1)
    reframed_test = series_to_supervised(scaled_test, 1, 1)

    # drop columns we don't want to predict (keeping Yield in this case)
    reframed_input.drop(reframed_input.columns[[8, 9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    #reframed_input.drop(reframed_input.columns[[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]], axis=1, inplace=True)
    print(reframed_input.head())

    ## Define Input and Output values
    values_input = reframed_input.values
    values_output = reframed_output.values
    values_test = reframed_output.values

    # Set train and validation split
    # n_train_hours = int(len(dataset)*0.80)

    # Apply train and validation split for input and output
    # train_input = values_input[:n_train_hours, :]
    # train_output = values_output[:n_train_hours, :]
    # validation_input = values_input[n_train_hours:int(len(values_input)), :]
    # validation_output = values_output[n_train_hours:int(len(values_input)), :]

    # assign split into input and outputs that make a little more sense
    train_X, train_y = values_input, values_output
    validation_X = values_test

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    validation_X = validation_X.reshape((validation_X.shape[0], 1, validation_X.shape[1]))
    print(train_X.shape, train_y.shape, validation_X.shape)

    # design network
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    #history = model.fit(train_X, train_y, epochs=35, batch_size=300, validation_data=(validation_X), verbose=2, shuffle=False)
    history = model.fit(train_X, train_y, epochs=35, batch_size=300, verbose=2, shuffle=False)
    #plot history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='validation')
    # pyplot.legend()
    # pyplot.show()

    # make a prediction
    yhat = model.predict(validation_X)
    validation_X = validation_X.reshape((validation_X.shape[0], validation_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    numpy.save(predictions, inv_yhat)

    #invert scaling for actual
    #validation_y = validation_y.reshape((validation_y.shape[0], validation_y.shape[2]))
    # inv_y = concatenate((validation_y, validation_X[:, 1:]), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:,0]
    

    # calculate RMSE
    # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # MAE = mae(inv_y, inv_yhat)
    # r2 = rsquared(inv_y, inv_yhat)
    # print('validation RMSE: %.3f' % rmse, 'r-squared: %.6f' % r2, 'MAE: %.5f' % MAE)

    

#-------------------------------------------
if __name__ == '__main__':
    main()