import os
import tensorflow as tf
#print(len(os.listdir('image/data_set/')))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'image/data_set/',
    image_size=(64,64),
    batch_size=64,
    subset ='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'image/data_set/',
    image_size=(64,64),
    batch_size=64,
    subset ='validation',
    validation_split=0.2,
    seed=1234
)
def 전처리함수(i, 정답):
    i = tf.cast( i/255.0, tf.float32)
    return i, 정답

train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding="same",activation='relu', input_shape=(64, 64, 3) ),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.summary()

model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'] )
model.fit(train_ds,validation_data=val_ds, epochs=5)