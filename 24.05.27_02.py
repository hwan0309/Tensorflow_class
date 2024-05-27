import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

x = np.array([-50, -40,-35, -25, -22, 10, 25, 30, 45 ], dtype=np.float32)
y = np.array([0,0,0,0,0,0,1,1,1,1], dtype=np.float32)

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1], activation='sigmoid')
])
init_w, init_b = model.get_weights()
print(init_w[0])
print(init_b)
