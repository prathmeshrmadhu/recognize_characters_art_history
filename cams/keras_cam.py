from keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize
import copy
import argparse
import os
import numpy as np
import pandas as pd
import cv2

K.clear_session()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to model")
ap.add_argument("-t", "--testdir", type=str, default='test/female')
ap.add_argument("-s", "--savedir", type=str, default='cams_A/female')
args = ap.parse_args()

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)

def cam_scratchtrained(args, img_path, output_img_path, output_eps_path):
    
    model = load_model(args.model)   
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x/255.0
    
    classes = ['female','male']
    preds = model.predict(x)[0]
    predictions = pd.DataFrame(columns=['category', 'probability'])
    predictions['probability'] = preds
    idx = np.argmax(preds)
    label = classes[idx]
    predictions['category'] = classes
    print (preds)

    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]

    layers = model.layers
    for layer in layers:
        if 'conv' in layer.name.lower():
            layer_to_use = layer.name.lower()
    last_conv_layer = model.get_layer(layer_to_use)
    
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    print (pooled_grads_value)
    
    for i in range(2048):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, -1)
    print (np.max(heatmap))
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + img
    output = output_img_path
    cv2.imwrite(output, superimposed_img)
    img=mpimg.imread(output)
    plt.subplot(2, 1, 2)
    plt.imshow(img)
    plt.axis('off')
    plt.title(predictions.loc[0,'category'].upper())
    plt.savefig(output_eps_path, type='eps', dpi=300)
    return None

test_images = os.listdir(args.testdir)
test_images_paths = [os.path.abspath(os.path.join(args.testdir, i)) for i in test_images]

for test_image_path in test_images_paths[:5]:
    print ('Generating CAM image for {}'.format(test_image_path.split('/')[-1]))
    output_cam_name = test_image_path.split('/')[-1].split('.')[0] + '_cam.jpg'
    output_eps_name = test_image_path.split('/')[-1].split('.')[0] + '_cam.eps'
    output_cam_path = os.path.abspath(os.path.join(args.testdir, output_cam_name))
    output_eps_path = os.path.abspath(os.path.join(args.testdir, output_eps_name))
    cam_scratchtrained(args, test_image_path, output_cam_path, output_eps_path)
    break



