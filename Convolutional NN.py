import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10   #it is natural images of birds cars etc each image is 32x32x3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model = keras.Sequential()

model.add(keras.Input(shape=(32,32,3)))
model.add(layers.Conv2D(32, 3, padding="valid", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64,3, activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128,3, activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
print(model.summary())


model.compile(

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)

model.evaluate(x_test,y_test,batch_size=32, verbose=2)