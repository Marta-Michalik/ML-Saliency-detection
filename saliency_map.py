import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tensorflow.keras import datasets
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt



model = load_model('model1_ok')


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    train_images= train_images.reshape(train_images.shape[0], 1, img_width, img_height)
    test_images = test_images.reshape(test_images.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    train_images= train_images.reshape(train_images.shape[0], img_width, img_height, 1)
    test_images = test_images.reshape(test_images.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

# Parse numbers as floats
train_images= train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalize data
train_images= train_images/ 255
test_images = test_images / 255

#picking a image and expanding the dimensions



#deactivating softmax function in the last layer (if not linear)
model.layers[-1].activation = None




def show_saliency_map(image, model, image_id) :
 #   img_batch = (np.expand_dims(image, 0))
    img_batch = image.reshape(1, image.shape[0],image.shape[1], image.shape[2] )
    # creating saliency object
    saliency = Saliency(model)
    # creating loss function
    loss = lambda output: tf.keras.backend.mean(output[:, tf.argmax([test_labels[image_id]])])

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
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.subplot(4, 5, i + 11)
    img = test_images[i]
    map = show_saliency_map(img, model, i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(map, cmap='jet')

plt.show()

model.summary()