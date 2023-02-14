import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#loading dataset and model
(x_train, y_train), (x_test, y_test)= datasets.cifar10.load_data()
model = load_model('model2_ok_v2')

#getting max and min values
min = x_train.min()
max = x_train.max()


#normalizing data from 0 to 1
x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
x_train, x_test = (x_train - min) / (max - min), (x_test - min) / (max - min)

#picking a image and expanding the dimensions
image_id = 67
img = x_test[image_id]


#deactivating softmax function in the last layer (if not linear)
model.layers[-1].activation = None




def show_saliency_map(image, model, image_id) :
    img_batch = (np.expand_dims(image, 0))
    # creating saliency object
    saliency = Saliency(model)
    # creating loss function
    loss = lambda output: tf.keras.backend.mean(output[:, tf.argmax(y_test[image_id])])

    # creating and normalizing saliency map
    saliency_map = saliency(loss, img_batch)
    saliency_map = normalize(saliency_map)

    # reshaping for vizualization
    sal_vis = saliency_map.reshape(saliency_map.shape[1], saliency_map.shape[2])

    # showing the map
    return sal_vis



for i in range(10):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.subplot(4, 5, i + 11)
    img = x_test[i]
    map = show_saliency_map(img, model, i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(map, cmap='jet')

plt.show()

model.summary()