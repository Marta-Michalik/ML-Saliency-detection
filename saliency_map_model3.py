import numpy as np
import tensorflow as tf
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
batch_size = 128
target_size = (110, 110)
# create generator

model = load_model('model3')

datagen = ImageDataGenerator(rescale=1./255)
test_images = datagen.flow_from_directory('D:/ES/data/test/', class_mode='binary', batch_size= batch_size, target_size=target_size)

x = test_images.next()




def show_saliency_map(image, model, image_id):
    img_batch = (np.expand_dims(image, 0))
    # creating saliency object
    saliency = Saliency(model)
    # creating loss function
    loss = lambda output: tf.keras.backend.mean(output[:, tf.argmax([x[1][image_id]])])

    # creating and normalizing saliency map
    saliency_map = saliency(loss, img_batch)
    saliency_map = normalize(saliency_map)

    # reshaping for vizualization
    sal_vis = saliency_map.reshape(saliency_map.shape[1], saliency_map.shape[2])

    # showing the map
    return sal_vis


columns = 4
rows = 1
w=10
h=10


for i in range(10):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x[0][i+1], cmap=plt.cm.binary)

    plt.subplot(4, 5, i + 11)
    map = show_saliency_map(x[0][i+1], model, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(map, cmap='jet')

plt.show()