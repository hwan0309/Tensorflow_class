import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

x = np.array([1,3,5,6,10], dtype = np.float32)
y = np.array([0.5,1,2,1.8,3], dtype=np.float32)

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1], activation="linear")
])

init_w, init_b = model.weights
print(init_w, init_b)

# plt.scatter(x,y)
# plt.plot(x, init_w[0][0] * x + init_b[0], color = 'red')
# plt.xlabel('x')
# plt.ylabel('init Linear Regression')
# plt.show()

sgd = keras.optimizers.SGD(learning_rate = 0.001)
model.compile(optimizer=sgd, loss='mean_squared_error')

history = model.fit(x,y, batch_size=1, epochs=10)

w,b = model.layers[0].get_weights()[0][0][0], model.layers[0]. get_weights()[1][0]
print(f"\n\n\n학습된 w: {w}, b: {b}\n\n\n")

plt.scatter(x,y)
plt.plot(x,w*x + b, color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()
