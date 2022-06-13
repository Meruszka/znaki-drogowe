import time

import tensorflow as tf
from tensorflow.keras import layers, models


def model1(image_size, epochs, train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Rescaling(1./255))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128 ,(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(len(test_labels), activation = 'softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), use_multiprocessing=True)
    end = time.time()
    return history, end-start

def model2(image_size, epochs, train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Rescaling(1./255))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128 ,(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(len(test_labels), activation = 'softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), use_multiprocessing=True)
    end = time.time()
    return history, end-start

def model3(image_size, epochs, train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(len(test_labels), activation = 'softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), use_multiprocessing=True)
    end = time.time()
    return history, end-start


def model4(image_size, epochs, train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(len(test_labels), activation = 'softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), use_multiprocessing=True)
    end = time.time()
    return history, end-start

def modelExtra(image_size, epochs, train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Rescaling(1./255))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128 ,(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(len(test_labels), activation = 'softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), use_multiprocessing=True)
    end = time.time()
    return history, end-start