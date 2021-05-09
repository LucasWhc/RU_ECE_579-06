import tensorflow as tf
from tensorflow.keras import models, layers
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential()

model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

time_start = time.time()

model.fit(x_train, y_train, epochs=10)

time_end = time.time()

print("The training takes", time_end - time_start, 's.')

model.evaluate(x_test, y_test, verbose=2)
