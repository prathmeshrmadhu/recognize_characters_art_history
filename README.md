# Recognizing Characters in Art History Using Deep Learning
This is the official github repository to the paper : https://dl.acm.org/citation.cfm?id=3357242

This repository is for the gender/sex detection/classification on faces/persons within art images (paintings, sculptures, art-works).
Important things to note :

## Installation and getting started:

- Run `pip install -r requirements.txt`. 
- If you are using a GPU, check/edit the `requirements.txt` file to install `tensorflow-gpu` instead of `tensorflow`

1) Install `cvlib` : `pip install --upgrade cvlib`
2) Before running train and test, make sure you have downloaded and placed the following files as follows:  
    a. `cfg/yolov3.cfg`  
    b. `model-weights/yolov3.weights`  
3) The data directories should be structured as :
```
data
├── train
│   ├── class0
│   ├── class1
├── test
│   ├── class0
│   ├── class1
```
4) There is one training script to generate all the models : `train.py`. 
    - To generate model A and B:
   > `python train.py -d  <path_to_dataset>`
    - To generate model C:
   > `python train.py -d  <path_to_dataset> -f True -mp <path_to_styled_model (model B)>`
5) After the training, check if the appropriate models are saved in the respective folders (self-explanatory from the code)
6) Testing the model on random folder of images. Run
> `python test.py --testdir <path_to_testdir> --preddir <path_to_save_predictions> --model <path_to_trained_model>`

### Acknowledgements
This repo is adapted from the github repo : https://github.com/arunponnusamy/gender-detection-keras. The authors would like to thank Arun Ponnusamy for his amazing work and sharing the code to build and continue working together without "rediscovering the wheel". 
