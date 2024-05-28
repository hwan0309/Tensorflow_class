import inline
import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(4,input_dim=,activation = 'relu'))

model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation='linear'))
model.compile(loss = 'mse', optmizer='adam')

model.summary()


m = 2
b = 3
x = np.linspace(0,50, 100)
np.random.seed(101)
noise = np.random.normal(loc=0, scale=4, size=len(x))

y = 2*x + b + noise
plt.plot(x,y,"*")
plt.show()
model.fit(x,y,epochs=10)

