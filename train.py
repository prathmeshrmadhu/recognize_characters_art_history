# Adapted from: Arun Ponnusamy
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
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob
import pandas as pd
import keras.callbacks
from keras.models import load_model


# initial parameters
seed = 42
random.seed(seed)
epochs = 100
lr = 1e-4
batch_size = 32
img_dims = (224, 224, 3)

# handle command line arguments
'''
INCLUDE ALL THE MODEL RELATED POINTS in ARGS.
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="datasets/classification_mary_cropped/train/",
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", type=str, default="gender_detection.model",
                help="path to output model")
ap.add_argument("-n", "--expname", type=str,
                default= 'C_cropped_softmax_' + VGGFace.__name__ + "_pretrained_resnet_E" + str(epochs) + "_B" + str(batch_size) +  "_I" + str(img_dims[0]), help="name of the experiment")
ap.add_argument("-f", "--finetune", type=bool, default=False, help='Flag to decide the finetuning on art history data')
ap.add_argument("-mp", "--pretrained_model_path", type=str, help='Path to pretrained model path. Pass only if style:True')

args = ap.parse_args()

print("----------------\n Experiment Name: ", args.expname)
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

    '''
    In our case/paper, we have considered Gabriel as male and
    Mary as female. Since we are also using the model to fine-tune on
    styled data, we use the common labels : male and female. 
    '''
    label = img.split(os.path.sep)[-2]
    if label == "male": 
        label = 1
    else:
        label = 0

    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("The unique labels are : " + str(np.unique(labels)))

# augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# custom parameters
nb_class = 2

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
itr = 0
for train, test in kfold.split(data, labels):
    # convert labels
    labelsX = to_categorical(labels[train], num_classes=2)
    labelsY = to_categorical(labels[test], num_classes=2)

    # select vggface resnet50 or vgg16 model architectures or load a pretrained network
    if args.finetune:
        model = load_model(args.pretrained_model_path)
    else:  
        vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        model = Model(vgg_model.input, out)

    # compile the model
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 10 ** (-8)
    opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    print ('The length of train and batch_size are : {}, {}'.format(len(train), batch_size))

    # train the model
    cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=0, mode='auto',
                                       baseline=None,
                                       restore_best_weights=False)
    H = model.fit_generator(aug.flow(data[train], labelsX, batch_size=batch_size, shuffle=True),
                    validation_data=(data[test], labelsY),
                    steps_per_epoch=len(train) // batch_size,
                    epochs=epochs, verbose=1, callbacks=[cb])

    # evaluate the model
    scores = model.evaluate(data[test], labelsY, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    # save the model to disk
    model.save('model_snaps/' + args.expname + '_cvfold_' + str(itr) + '.model')

    # plot training/validation loss/accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = min(len(H.history["loss"]), epochs)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy" + " cvfold_" + str(itr))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")

    # save plot to disk
    plt.savefig('plots/' + args.expname + '_cvfold_' + str(itr) + '.png')

    # saving the history of the experiment to json
    hist_df = pd.DataFrame(H.history)
    hist_json_file = "history_logs/" + args.expname + '_cvfold_' + str(itr) + '.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    itr = itr + 1

print(cvscores)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


## DUMPYARD
# model = ResNet152(input_shape=(img_dims[0], img_dims[1], img_dims[2]), classes=2)
# hidden_dim = 512
# vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
# last_layer = vgg_model.get_layer('pool5').output
# x = Flatten(name='flatten')(last_layer)
# x = Dense(hidden_dim, activation='relu', name='fc6')(x)
# x = Dense(hidden_dim, activation='relu', name='fc7')(x)
# out = Dense(nb_class, activation='sigmoid', name='fc8')(x)
# model = Model(vgg_model.input, out)


