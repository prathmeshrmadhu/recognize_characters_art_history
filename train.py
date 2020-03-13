# Author: Arun Ponnusamy
# website: http://www.arunponnusamy.com

# import necessary packages
import matplotlib

matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from model.smallervggnet import SmallerVGGNet
# from model.resnet import ResNet152
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob

# initial parameters
seed = 42
random.seed(seed)
epochs = 2
lr = 1e-3
batch_size = 32
img_dims = (96, 96, 3)

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", type=str, default="gender_detection.model",
                help="path to output model")
ap.add_argument("-p", "--plot", type=str,
                default="plots/SVGGNet_plot_E" + str(epochs) + "_B" + str(batch_size) + "_I" + str(
                    img_dims[0]) + ".png",
                help="path to output accuracy/loss plot")
#ap.add_argument("-n", "--expname", type=str,
#                default=modelname + "E" + str(epochs) + "_B" + str(batch_size) + "_I" + str(img_dims[0]),
#                help="path to output accuracy/loss plot")
args = ap.parse_args()

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(args.dataset + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# create groud-truth label from the image path
for img in image_files:

    image = cv2.imread(img)

    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)
    # data.append(img)

    label = img.split(os.path.sep)[-2]
    if label == "Mary":
        label = 1
    else:
        label = 0

    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("The unique labels are : " + str(np.unique(labels)))

# split dataset for training and validation
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15,
#                                                  random_state=seed)

#labels = to_categorical(labels, num_classes=2)
# trainY = to_categorical(trainY, num_classes=2)
# testY = to_categorical(testY, num_classes=2)

# augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
itr = 0
for train, test in kfold.split(data, labels):
    # create model
    # build model
    labelsX = to_categorical(labels[train], num_classes=2)
    labelsY = to_categorical(labels[test], num_classes=2)
    model = SmallerVGGNet.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                                classes=2)

    # compile the model
    opt = Adam(lr=lr, decay=lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    print ('The length of train and batch_size are : {}, {}'.format(len(train), batch_size))
    
    # train the model
    model.fit(aug.flow(data[train], labelsX, batch_size=batch_size), epochs=epochs, verbose=1)

    # evaluate the model
    scores = model.evaluate(data[test], labelsY, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    # save the model to disk
    model.save('model_' + str(itr) + '.model')

    # plot training/validation loss/accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy" + " itr_" + str(itr))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")

    # save plot to disk
    plt.savefig(args.plot)
'''H = model.fit_generator(aug.flow(data[train], labels[train], batch_size=batch_size),
                    validation_data=(testX,testY),
                    steps_per_epoch=len(trainX) // batch_size,
                    epochs=epochs, verbose=1)


model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
'''

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

# model = ResNet152(input_shape=(img_dims[0], img_dims[1], img_dims[2]), classes=2)




