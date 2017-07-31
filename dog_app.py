import sys
import argparse
import numpy as np
from glob import glob
import logging
import random
import dlib
import pandas as pd
from skimage import io
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import model_from_json

from extract_bottleneck_features import *

random.seed(8675309)

logging.basicConfig(
    format='%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-m", "--model", required=True, help="path to serialized model in json format")
ap.add_argument("-w", "--weights", required=True, help="path to trained model weights in h5 format")
args = vars(ap.parse_args())


# Step 0: Load data =========================================================
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


dog_names = [ item[ 20:-1 ] for item in sorted(glob("dogImages/train/*/")) ]
_, train_targets = load_dataset('dogImages/train')
_, valid_targets = load_dataset('dogImages/valid')
_, test_targets = load_dataset('dogImages/test')


# Step 1 : Object detection ==================================================
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def dlib_face_detector(img_path):
    detector = dlib.get_frontal_face_detector()
    img = io.imread(img_path)
    dets = detector(img, 1)
    return len(dets) > 0


ResNet50_model = ResNet50(weights='imagenet')


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# Step 2: Load pretrained Inception v3 =========================================

json_file = open(args['model'], 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(args['weights'])
logging.info("Loaded model from disk: {}".format(args['model']))



def inception_predict_breed(img_path, loaded_model):
    #tmp = img_path.split('/')[ -1 ].split('_')[ :-1 ]
    #y_true = "_".join(tmp)

    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))

    # obtain predicted vector
    y_pred = loaded_model.predict(bottleneck_feature)
    preds = {}
    for n, p in zip(dog_names, y_pred[ 0 ]):
        preds[ n ] = p
    top5 = list(reversed(sorted(preds.items(), key=lambda x: x[ 1 ])))[ :5 ]
    tmp = pd.DataFrame(top5, columns=[ 'breed', 'prob' ])

    # return dog breed that is predicted by the model
    return tmp


def main():


    # Step 1: Detect object ===============================================

    img_path = args['image']
    is_human = dlib_face_detector(img_path)
    is_dog = dog_detector(img_path)


    if not is_human and not is_dog:
        logging.info("Can't detect human or dogs\n Try again with different image?")

    elif is_dog and is_human:
        logging.info("")


    elif is_dog and not is_human:
        logging.info("Hello, dog!")
        tmp = inception_predict_breed(img_path, loaded_model)
        logging.info("You look like a {}...".format(tmp.loc[0, 'breed']))
        logging.info("Or maybe {}, {}, {}, or, {}?".format(tmp.loc[1, "breed"],
                                                        tmp.loc[2, 'breed'],
                                                        tmp.loc[3, 'breed'],
                                                        tmp.loc[4, 'breed']))

    elif is_human and not is_dog:
        logging.info("Hello, human!")
        tmp = inception_predict_breed(img_path, loaded_model)
        logging.info("You look like a {}...".format(tmp.loc[ 0, 'breed' ]))
        logging.info("Or maybe {}, {}, {}, or, {}?".format(tmp.loc[ 1, "breed" ],
                                                        tmp.loc[ 2, 'breed' ],
                                                        tmp.loc[ 3, 'breed' ],
                                                        tmp.loc[ 4, 'breed' ]))



if __name__ == '__main__':
    main()





















