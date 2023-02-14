import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
target_size = (110, 110)
# create generator
datagen = ImageDataGenerator(rescale=1./255)
# prepare an iterators for each dataset
train_images = datagen.flow_from_directory('D:/ES/data/train/', class_mode='binary', batch_size= batch_size, target_size=target_size)
test_images = datagen.flow_from_directory('D:/ES/data/test/', class_mode='binary', batch_size= batch_size, target_size=target_size)
validation_images = datagen.flow_from_directory('D:/ES/data/validation/', class_mode='binary', batch_size= batch_size, target_size=target_size)


print(type(test_images.class_indices))
steps_per_epochs = len(train_images)
validation_steps = len(test_images)

verbosity = 1
'''
x = train_images.next()

columns = 4
rows = 1
w=10
h=10
fig = plt.figure(figsize=(16, 16))

for i in range(1, columns*rows +1):
    image = x[0][i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(image)
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(110, 110, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='linear'))


model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit_generator(train_images,
                              steps_per_epoch=steps_per_epochs,
                              verbose=verbosity,
                              validation_data=validation_images,
                              validation_steps=validation_steps,
                              epochs= 30)


test_loss, test_acc = model.evaluate(test_images,  verbose=2)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

model.save('model3')

'''