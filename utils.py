import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os
import numpy as np


def prepare_data(dataset_name):
    data_path = os.path.join("./datasets", dataset_name)

    trainA = []
    trainB = []
    testA = []
    testB = []
    for path, dir, files in os.walk(data_path):
        for file in files:
            image = os.path.join(path, file)
            if path.__contains__('trainA') :
                trainA.append(misc.imresize(misc.imread(image), [256, 256]))
            if path.__contains__('trainB') :
                trainB.append(misc.imresize(misc.imread(image), [256, 256]))
            if path.__contains__('testA') :
                testA.append(misc.imresize(misc.imread(image), [256, 256]))
            if path.__contains__('testB') :
                testB.append(misc.imresize(misc.imread(image), [256, 256]))


    trainA = preprocessing(np.asarray(trainA))
    trainB = preprocessing(np.asarray(trainB))
    testA = preprocessing(np.asarray(testA))
    testB = preprocessing(np.asarray(testB))

    np.random.shuffle(trainA)
    np.random.shuffle(trainB)
    np.random.shuffle(testA)
    np.random.shuffle(testB)

    return trainA, trainB, testA, testB

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