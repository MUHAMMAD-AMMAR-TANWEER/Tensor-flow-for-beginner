import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#loading dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# print(y_train.shape)
#to check the dimensions of the trainind and test data

#converting them to float and flaten them from (60000,28,28) 28*28 = 784  and normalizing them as well
x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0

#Using sequential API it is good for one input to output
model = keras.Sequential(
[
keras.Input(shape=(28*28)),
layers.Dense(512, activation='relu'),
layers.Dense(256, activation='relu'),
layers.Dense(10, activation='softmax')
]

)
# now the above think can be done in a different way

model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))





model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),#if we doesnot define activation function in last layer then it needs to be true
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train, batch_size=32, epochs=5, verbose=2)

model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# it's time to use functional API new way of typing above code and most adaptive one

inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name="first-layer")(inputs)
x = layers.Dense(256, activation='relu', name='second-layer')(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train, batch_size=32, epochs=5, verbose=2)

model.evaluate(x_test, y_test, batch_size=32, verbose=2)

print(model.summary())