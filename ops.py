import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, batch_norm
from tensorflow.contrib.framework import arg_scope
import random
initializer = tf.truncated_normal_initializer(stddev=0.02)

class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image

def conv_layer(x, filter_size, kernel, stride=1, padding="VALID",do_norm=True, norm='instance', is_training=True, do_relu=True, leak=0, layer_name="conv"):
    with tf.variable_scope(layer_name):
        if padding == 1 :
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]])
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride)
        else :
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride, padding=padding)

        if do_norm:
            if norm == 'instance' :
                x = instance_norm(x)
            else :
                x = Batch_Normalization(x, training=is_training)

        if do_relu:
            if leak == 0:
                x = relu(x)
            else:
                x = lrelu(x)

        return x


def deconv_layer(x, filter_size, kernel, stride=1, padding="VALID",do_norm=True, norm='instance', is_training=True, do_relu=True, leak=0, layer_name="deconv") :
    with tf.variable_scope(layer_name):
        if padding == 1 :
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]])
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride)
        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride, padding=padding)

        if do_norm:
            if norm == 'instance' :
                x = instance_norm(x)
            else :
                x = Batch_Normalization(x, training=is_training)

        if do_relu:
            if leak == 0 :
                x = relu(x)
            else:
                x = lrelu(x)

        return x

def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out

def Batch_Normalization(x, training, scope='batch_norm'):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def tanh(x):
    return tf.tanh(x)

def relu(x):
    return tf.nn.relu(x)

def swish(x): # may be it will be test
    return x * tf.sigmoid(x)


