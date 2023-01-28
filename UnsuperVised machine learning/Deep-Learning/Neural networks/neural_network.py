# import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# print(tf.__version__)
data = keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_imgs ,train_labels),(test_imgs,test_labels) = data.load_data()

train_imgs=train_imgs/255.0
test_imgs=test_imgs/255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(128,activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_imgs,train_labels,epochs=5)

prdictions = model.predict(test_imgs)


for i in range(0,10):
    plt.grid(False)
    plt.imshow(test_imgs[i],cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predicted: "+class_names[np.argmax(prdictions[i])])
    plt.show()


# model.evaluate(test_imgs,test_labels)
# print(test_imgs[0])