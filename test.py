# Adapted from: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os, glob
import cvlib as cv
import matplotlib.pyplot as plt

from pandas_ml import ConfusionMatrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import cm_analysis

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testdir", required=True, #type=str, default="classification_mary/test",
	help="path to test images' folder")
ap.add_argument("-p", "--preddir", required=True, #type=str, default="predictions_B_softmax",
	help="path to predictions' folder")
ap.add_argument("-m", "--model", required=True,
	help="pass True if want the styled model")
ap.add_argument("-hq", "--high_quality", type=bool, default=False)

args = ap.parse_args()
model_path = args.model 

if os.path.exists(model_path):
    print ("Model path is :" + str(model_path))
else:
    raise Exception("Model not found")

# load model
print (model_path)
model = load_model(model_path)

# read the test images' directory
image_files = [f for f in glob.glob(args.testdir + "/**/*", recursive=True) if not os.path.isdir(f)]

#print (image.shape)
if image_files is None:
    print("Could not test images' folder")
    exit()

'''
In our case/paper, we have considered Gabriel as male and
Mary as female. Since we are also using the model to fine-tune on
styled data, we use the common labels : male and female. 
'''
classes = ['Mary','Gabriel']
classes_a = ['Gabriel', 'Mary']

all_preds = []
all_groundtruths = []

# Generate predictions_old as well as face bboxes
for imagename in image_files:
    # detect faces in the image
    image = cv2.imread(imagename)
    face, confidence = cv.detect_face(image)
    
    if confidence != []:
         # get corner points of face rectangle
        (startX, startY) = face[0][0], face[0][1]
        (endX, endY) = face[0][2], face[0][3]

        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    else:
        startY = 10
        startX = 10

    if (np.shape(image)[0] != 0)  and (np.shape(image)[1] != 0):
        image = cv2.resize(image, (224,224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        # apply gender detection on the body
        conf = model.predict(image)[0]

        image = np.uint8(np.squeeze(image)*255.0)
        # get label with max accuracy
        idx = np.argmax(conf)
        
        # While training for Model A, we used classes_a as labels.
        if model_path.split('/')[-1] == 'A_VGGFace_resnet_E100_B32_I224_cvfold_2.model':
            classes = classes_a
        
        label = classes[idx]
        possible_true_label = imagename.split('/')

        if 'female' in possible_true_label or 'female' in possible_true_label[-1].split('_'):
            true_label = 'Mary'
        else:
            true_label = 'Gabriel'
                
        all_preds.append(label)
        all_groundtruths.append(true_label)
        
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 1)

        if not os.path.exists(args.preddir):
            print("Making the output directory")
            os.makedirs(args.preddir)

        out_image_name = '_'.join(imagename.split('/')[2:])
        cv2.imwrite(args.preddir + os.sep + out_image_name, image)

        if args.high_quality == True:
            high_quality_image_name = args.image.split('/')[-1].split('.jpg')[0] + '_gender_detection.eps'
            print("Saving {} image".format(high_quality_image_name))
            plt.imshow(image)
            plt.savefig(high_quality_image_name, format='eps', dpi=1000)
            
cm_folder = 'final_stuff/confusion_matrix/' 
if not os.path.exists(cm_folder):
    os.makedirs(cm_folder)
    
cm_path = cm_folder + model_path.split('/')[-1].split('.')[0] + '_confusion_matrix.eps'
cm = cm_analysis(all_groundtruths, all_preds, cm_path, ['Mary', 'Gabriel'])
print (cm)