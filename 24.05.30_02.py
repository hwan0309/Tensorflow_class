import tensorflow as tf
import numpy as np
from sympy.abc import lamda

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)


def nav():
    예측_y = train_x * a + b
    return tf.keras.losses.mse(train_y, 예측_y)


opt = tf.keras.optimizers.Adam(learning_rate=0.001)

for i in range(300):
    opt.minimize( lamda:nav(a, b), var_list=[a, b])
    print(a.numpy(), b.numpy())

