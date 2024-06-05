import requests
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

url = 'inception.h5'
r = requests.get(url, allow_redirects=True)

open('inception_v3.h5', 'wb').write(r.content)

inception_model = InceptionV3(input_shape=(150, 150,3), include_top=False, weights=None )
inception_model .load_weights('inception_v3.h5')

#inception_model.summary()

for i in inception_model.layers:
    i.trainable = False

마지막레이어 = inception_model.get_layer('mixed7')
print(마지막레이어)
print(마지막레이어.output)
