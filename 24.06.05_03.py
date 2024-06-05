import tensorflow as tf

layer1 = tf.keras.layers.Flatten()(마지막레이어.output)
layer2 = tf.keras.layers.Dense(1024,activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(drop1)

model = tf.keras.Model(inception_model.input,layer3)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(train_ds, validation)