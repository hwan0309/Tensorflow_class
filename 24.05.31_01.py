import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# print(trainX[0])
# print(trainX.shape )
# print(trainY.shape)

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) )
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding="same",activation='relu', input_shape=(28, 28, 1) ),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    #tf.keras.layers.Dense(128,input_shape=(28,28), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()


model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )
model.fit(trainX, trainY,validation_data=(testX, testY), epochs=5)

# score = model.evaluate(testX, testX)
# print(score)