import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

from util import *

train_x = get_time_features()
test_x = get_time_features("test")
train_y = get_labeled_value()

train_x = train_x.reshape(-1, TIME_STEPS, INPUT_SIZE)
test_x = test_x.reshape(-1, TIME_STEPS, INPUT_SIZE)

model = Sequential()
model.add(LSTM(TIME_STEPS, input_shape=(INPUT_SIZE, LOOK_BACK)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)
