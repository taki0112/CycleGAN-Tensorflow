import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def deconv(x, channels, kernel=3, stride=2, use_bias=True, scope='deconv_0') :
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding='SAME')

        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            real_loss = 0.5 * tf.reduce_mean(tf.squared_difference(real[i], 1.0))
            fake_loss = 0.5 * tf.reduce_mean(tf.square(fake[i]))

        if type == 'gan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

        loss.append(real_loss + fake_loss)

    return sum(loss)


def generator_loss(type, fake):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

        if type == 'gan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

        loss.append(fake_loss)


    return sum(loss)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss
