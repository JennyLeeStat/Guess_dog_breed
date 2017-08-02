import os
import argparse
import numpy as np
from glob import glob
import warnings
import random
import cv2
import dlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
from skimage import io, transform
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import model_from_json

from extract_bottleneck_features import *


random.seed(8675309)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sns.set_style("whitegrid")
mycol = sns.husl_palette(10)[0]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

img_path = args['image']

# Step 0: Load data =========================================================
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


dog_names = [ item[ 20:-1 ] for item in sorted(glob("dogImages/train/*/")) ]


# To detect dogs, use pretrained resnet50 on imagenet
ResNet50_model = ResNet50(weights='imagenet')

# To classify dog breed, load trained model and weights
json_file = open('saved_models/inception_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('saved_models/weights.best.inception.hdf5')
print('The model and trained '
      'weights are loaded to disk.')


# Step 1 : Object detection ==================================================
def path_to_tensor(img_path):
    # image preprocess helper
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def dlib_face_detector(img_path):
    detector = dlib.get_frontal_face_detector()
    img = io.imread(img_path)
    dets = detector(img, 1)
    return len(dets) > 0


def crop_face(img_path, display=False):
    detector = dlib.get_frontal_face_detector()
    img = io.imread(img_path)
    dets, _, _ = detector.run(img, 1)
    if len(dets) == 0:
        print("No human face is detected...")
        return
    else:
        #print("{} human faces are detected.".format(len(dets)))
        cropped = [ ]
        plt.figure(figsize=(7, 7))
        for i, d in enumerate(dets):
            c = img[ int(d.top() * 0.9):int(d.bottom() * 1.1),
                int(d.left() * 0.9):int(d.right() * 1.1) ]
            c = transform.resize(c, (224, 224, 3))
            if display:
                _ = plt.subplot(6, 6, i + 1)
                plt.imshow(c)
                plt.xticks(())
                plt.yticks(());

            c = np.expand_dims(c, axis=0)
            cropped.append(c)
        return cropped


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# Step 2: pretrained Inception v3 =========================================
def inception_predict_breed(img, show_sample=False):
    try:
        if img.shape == (1, 224, 224, 3):
            img_tensor = img
            img_copy = img.copy()

    except:
        img_tensor = path_to_tensor(img)
        img_copy = img

    print('Let me guess...')
    bottleneck_feature = extract_InceptionV3(img_tensor)
    y_pred = loaded_model.predict(bottleneck_feature)

    preds = {}
    for n, p in zip(dog_names, y_pred[ 0 ]):
        preds[ n ] = p
    top5 = list(reversed(sorted(preds.items(), key=lambda x: x[ 1 ])))[ :5 ]
    tmp = pd.DataFrame(top5, columns=[ 'breed', 'prob' ])
    if tmp.loc[ 0, 'prob' ] > 0.995:
        print("I'm pretty sure you are {}.".format(tmp.loc[ 0, 'breed' ]))

    else:
        print("You look like a(n) {}...".format(tmp.loc[ 0, 'breed' ]))
        print("or maybe {}, {}, {}, {}?".format(tmp.loc[ 1, "breed" ],
                                                tmp.loc[ 2, 'breed' ],
                                                tmp.loc[ 3, 'breed' ],
                                                tmp.loc[ 4, 'breed' ]))
    inception_show_predicted(img_copy, tmp)
    if show_sample:
        img_to_show = get_img_to_show(tmp)
        print("Here are sample dog images that look like you.")
        images_square_grid(img_to_show, img)
    return tmp


def get_img_to_show(tmp):
    img_to_show = []
    for i in range(5):
        tmp_list = glob('dogImages/train/*{}/*.jpg'.format(tmp.loc[i, 'breed']))
        tmp_selected = np.random.choice(tmp_list, int(1000 * tmp.loc[i, 'prob'])).tolist()
        img_to_show.append(tmp_selected)
    img_to_show = sum(img_to_show, [])
    if len(img_to_show) > 24:
        img_to_show = img_to_show[:24]
    random.shuffle(img_to_show)
    return img_to_show

def img_reader_helper(img_path):
    img = io.imread(img_path)
    img = transform.resize(img, (224, 224))
    return img


def images_square_grid(images, input_image_path):
    random.shuffle(images)
    if len(images) > 24:
        images = images[ :24 ]

    img_in_square = [ ]
    for _, f in enumerate(images):
        img = img_reader_helper(f)
        img_in_square.append(img)

    input_image = img_reader_helper(input_image_path)
    img_in_square.insert(12, input_image)
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio

    for i, f in enumerate(img_in_square):
        _ = plt.subplot(5, 5, i + 1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(f)


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return


def inception_show_predicted(img, tmp):
    try:
        if img.shape == (224, 224, 3):
            img = img
        elif img.shape == (1, 224, 224, 3):
            img = np.squeeze(img, axis=0)
    except:
        img = io.imread(img)

    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[ 4, 1 ])
    ax0 = plt.subplot(gs[ 0 ])
    ax0.imshow(img)
    ax0.set_xticks(())
    ax0.set_yticks(())

    ax1 = plt.subplot(gs[ 1 ])
    sns.barplot(x="prob", y="breed", data=tmp, color=mycol, edgecolor=".2")
    ax1.set_xlim((0, 1))
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_yticks(())
    ax1.set_xticks(())

    ax1.set_title("Top 5 Predicted breed", fontweight='bold')
    for i, p in enumerate(ax1.patches):
        ax1.annotate(tmp[ 'breed' ][ i ],
                     (p.get_width() * 0.9, p.get_y() + .4), fontweight='bold')
        ax1.annotate(np.round(tmp[ 'prob' ][ i ], 4), (p.get_width() * 0.9, p.get_y() + .6))
    plt.tight_layout()
    plt.show();


def crop_face2(img_path, display=False):
    detector = dlib.get_frontal_face_detector()
    img = io.imread(img_path)
    dets, _, _ = detector.run(img, 1)
    if len(dets) == 0:
        print("No human face is detected...")
        return
    else:
        #print("{} human faces are detected.".format(len(dets)))
        _ = plt.figure(figsize=(11, 11))
        for i, d in enumerate(dets):
            c = img[ int(d.top() * 0.9):int(d.bottom() * 1.1),
                int(d.left() * 0.9):int(d.right() * 1.1) ]

            filename = 'images/tmp_' + str(i) + '.jpg'
            io.imsave(filename, c)

            if display:
                _ = plt.subplot(11, 11, i + 1)
                c = transform.resize(c, (224, 224, 3))
                plt.imshow(c)
                plt.xticks(())
                plt.yticks(())
        return len(dets)


def predict_cropped(n_cropped):
    for n in range(n_cropped):
        filename = 'images/tmp_' + str(n) + '.jpg'
        inception_predict_breed(filename, False)


def main():
    # Step 1: Detect object ===============================================
    print("=======================================")
    is_human = dlib_face_detector(img_path)
    is_dog = dog_detector(img_path)

    if not is_human and not is_dog:
        print("Can't detect neither human nor dogs.\n "
              "Try again with a different image?\n"
              "It'd be more accurate with a picture of face facing front.")

    elif is_dog and is_human:
        # Need to distinguish two cases:
        # 1) a human looks like a dog or
        # 2) both human and dogs are in the picture
        print("Hello, there! I found both dog and human")

        n_cropped = crop_face2(img_path, True)
        filenames = [ 'images/tmp_' + str(i) + '.jpg' for i in range(n_cropped) ]

        is_dog_really = 0
        for f in filenames:
            is_dog_really = dog_detector(f)
            if is_dog_really:
                print("Hmmm.. this is confusing. What a man-ly dog!")
                _ = inception_predict_breed(f, True)
                is_dog_really += 1

            else:
                _ = inception_predict_breed(f)

        if is_dog_really == 0:
            # Inspected all cropped faces and no dogs were found
            print('Now guessing your dog...')
            inception_predict_breed(img_path, True)



    elif is_dog and not is_human:
        # predict the dog breed
        print("Hello, dog!")
        result = inception_predict_breed(img_path, True)


    else:
        # guess the breed that most resembles the person
        n_cropped = crop_face2(img_path, True)

        if n_cropped > 1:
            # when there are more than one face are found
            print('Hello, people!')
            print("I found {} faces in the image".format(n_cropped))
            predict_cropped(n_cropped)
        else:
            # If only one person was found
            print('Hello, human!')
            print("I found {} faces in the image".format(n_cropped))
            inception_predict_breed(img_path, True)


if __name__ == '__main__':
    main()





















