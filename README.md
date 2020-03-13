# Recognizing Characters in Art History Using Deep Learning
This is the official github repository to the paper : https://dl.acm.org/citation.cfm?id=3357242

This repository is for the gender/sex detection/classification on faces/persons within art images (paintings, sculptures, art-works).
Important things to note :


Run pip install -r requirements.txt. If you are using a GPU, edit the requirements.txt file to install tensorflow-gpu instead of tensorflow


Install 'cvlib' : pip install --upgrade cvlib


Before running train and test, make sure you have downloaded and placed the following files as follows:
a. cfg/yolov3.cfg
b. model-weights/yolov3.weights


Run train as follows :
python train.py -d <path-to-dataset>
The dataset should be structured as :
/class-0/
/class-1/
