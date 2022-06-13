import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models

from models import *

class_ids = []
filenames = []

df = pd.read_csv('./archive/labels.csv')

for file in os.listdir("./archive/traffic_Data/TEST/"):
    if file.endswith(".png"):
        filenames.append("./archive/traffic_Data/TEST/" + file)
        classname = file.split("_")[0].lstrip('0')
        if classname == "":
            classname = "0"
        class_ids.append(classname)

classes = df["ClassId"]

for index, value in classes.items():
    for file in os.listdir(os.path.join("./archive/traffic_Data/DATA/", str(value))):
        if file.endswith(".png"):
            filenames.append(os.path.join(
                "./archive/traffic_Data/DATA/", str(value), file))
            classname = file.split("_")[0].lstrip('0')
            if classname == "":
                classname = "0"
            class_ids.append(classname)

df = pd.DataFrame(list(zip(class_ids, filenames)),
                  columns=['classid', 'filename'])

# Dzieli dane na testowe i treningowe
(train_set, test_set) = train_test_split(
    df.values, train_size=0.7, random_state=244)

train_labels = train_set[:, 0]
test_labels = test_set[:, 0]

train_images = []
test_images = []

image_size = 100
epochs = 20
threshold = 0.2

# Zamienia ścieżki do obrazu na macierz obrazu dla treningowych
# Dodatko zamienia obraz na macierz obrazu
for image_path in train_set[:, 1]:
    image = (Image.open(image_path))
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = image/255
    image = 1.0 * (image > threshold)
    train_images.append(image)


# Zamienia ścieżki do obrazu na macierz obrazu dla testowych
# Dodatko zamienia obraz na macierz binarna
for image_path in test_set[:, 1]:
    image = (Image.open(image_path))
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = image/255
    image = 1.0 * (image > threshold)
    test_images.append(image)

test_images = np.array(test_images)
train_images = np.array(train_images)

test_labels = np.reshape(test_labels, (test_labels.size, 1))
train_labels = np.reshape(train_labels, (train_labels.size, 1))

test_labels = test_labels.astype(np.int0)
train_labels = train_labels.astype(np.int0)


h2, t2 = model2(image_size=image_size, epochs=epochs, train_images=train_images,
                train_labels=train_labels, test_images=test_images, test_labels=test_labels)


plt.plot(h2.history['val_accuracy'], label='val_accuracy 2')
plt.plot(h2.history['accuracy'], label='accuracy 2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.7, 1])
plt.legend(loc='lower right')
plt.show()

with open('czas.csv', 'a') as file:
    file.write(str(epochs) + ',' + str(t2) + ',extra\n')

