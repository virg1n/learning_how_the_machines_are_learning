import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import keras
import matplotlib.pyplot as plt
# %matplotlib inline

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential(
    [
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ]
)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

print(np.argmax(model.predict(np.expand_dims(x_test[0], axis=0))))