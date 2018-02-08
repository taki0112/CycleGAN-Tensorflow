import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os
import numpy as np


def prepare_data(dataset_name, size):
    data_path = os.path.join("./dataset", dataset_name)

    trainA = []
    trainB = []
    for path, dir, files in os.walk(data_path):
        for file in files:
            image = os.path.join(path, file)
            if path.__contains__('trainA') :
                trainA.append(misc.imresize(misc.imread(image), [size, size]))
            if path.__contains__('trainB') :
                trainB.append(misc.imresize(misc.imread(image), [size, size]))


    trainA = preprocessing(np.asarray(trainA))
    trainB = preprocessing(np.asarray(trainB))

    np.random.shuffle(trainA)
    np.random.shuffle(trainB)

    return trainA, trainB

def load_test_data(image_path, size=256):
    img = misc.imread(image_path)
    img = misc.imresize(img, [size, size])
    img = img/127.5 - 1
    return img

def preprocessing(x):
    """
    # Create Normal distribution
    x = x.astype('float32')
    x[:, :, :, 0] = (x[:, :, :, 0] - np.mean(x[:, :, :, 0])) / np.std(x[:, :, :, 0])
    x[:, :, :, 1] = (x[:, :, :, 1] - np.mean(x[:, :, :, 1])) / np.std(x[:, :, :, 1])
    x[:, :, :, 2] = (x[:, :, :, 2] - np.mean(x[:, :, :, 2])) / np.std(x[:, :, :, 2])
    """
    x = x/127.5 - 1 # -1 ~ 1
    print(np.shape(x))
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir