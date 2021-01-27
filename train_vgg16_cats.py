from imutils import paths
import random
import cv2
from keras.preprocessing.image import img_to_array
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from VGG16 import VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt

print("[INFO] loading images...")
data = []
labels = []

imagePaths = paths.list_images('./images')
imagePaths = list(imagePaths)
imagePaths = sorted(imagePaths)
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (224, 224))
	image = img_to_array(image)
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "cats" else 0
	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = VGG16.build(width=224, height=224, depth=3, classes=2)
EPOCHS = 10
INIT_LR = 0.1
BS = 5
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save('./Vgg16_Cats_notCats.model')

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('./plot.png')