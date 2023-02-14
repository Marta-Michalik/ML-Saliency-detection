import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from keras.layers import Dropout
from tensorflow.keras.models import load_model


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
plt.figure(figsize=(10,10))

for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()

history = model.fit(train_images, train_labels, epochs=25, batch_size= 125,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


print(test_acc)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


model.save('model1_ok')


