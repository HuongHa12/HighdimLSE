# coding=utf-8
# This code is written following the code in the paper "Can You Trust Your
# Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", NeurIPS'19
# Lint as: python2, python3
"""Build and train MNIST models for UQ experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import attr
import tensorflow as tf
from utils import uq_utils
keras = tf.keras

_NUM_CLASSES = 10
_MNIST_SHAPE = (28, 28, 1)
_NUM_IMAGE_EXAMPLES_TO_RECORD = 32
_BATCH_SIZE_FOR_PREDICT = 1024

ARCHITECTURES = ['mlp', 'lenet', 'cnn_complex']
METHODS = ['vanilla', 'dropout', 'svi', 'll_dropout', 'll_svi']


@attr.s
class ModelOptions(object):
    """Parameters for model construction and fitting."""
    num_layer = attr.ib()
    num_train_examples = attr.ib()
    batch_size = attr.ib()
    method = attr.ib()
    architecture = attr.ib()
    mlp_layer_sizes = attr.ib()
    num_examples_for_predict = attr.ib()
    predictions_per_example = attr.ib()


def _build_mlp(opts):
    """Builds a multi-layer perceptron Keras model."""
    layer_builders = uq_utils.get_layer_builders(opts.method,
                                                 opts.dropout_rate,
                                                 opts.num_train_examples)
    _, dense_layer, dense_last, dropout_fn, dropout_fn_last = layer_builders
    
    if (opts.dataset_name == 'MNIST') | (opts.dataset_name == 'FashionMNIST') | (opts.dataset_name == 'EMNIST'):
        input_shape = _MNIST_SHAPE

    inputs = keras.layers.Input(input_shape)
    net = keras.layers.Flatten(input_shape=input_shape)(inputs)
    for size in opts.mlp_layer_sizes:
        net = dropout_fn(net)
        net = dense_layer(size, activation='relu')(net)
    net = dropout_fn_last(net)
    if (opts.dataset_name == 'EMNIST'):
        _NUM_CLASSES = 47
    else:
        _NUM_CLASSES = 10
    logits = dense_last(_NUM_CLASSES)(net)

    return keras.Model(inputs=inputs, outputs=logits)


def _build_mlp_reg(opts):
    """Builds a multi-layer perceptron Keras model."""
    layer_builders = uq_utils.get_layer_builders(opts.method,
                                                 opts.dropout_rate,
                                                 opts.num_train_examples)
    _, dense_layer, dense_last, dropout_fn, dropout_fn_last = layer_builders
    
    if (opts.dataset_name == 'MNIST') | (opts.dataset_name == 'FashionMNIST') | (opts.dataset_name == 'EMNIST'):
        input_shape = _MNIST_SHAPE
    else:
        input_shape = [opts.input_size]

    inputs = keras.layers.Input(input_shape)
    net = keras.layers.Flatten(input_shape=input_shape)(inputs)
    for i in range(opts.num_layer):
        net = dropout_fn(net)
        net = dense_layer(opts.mlp_layer_sizes, activation='relu')(net)
    net = dropout_fn_last(net)
    if (opts.dataset_name == 'EMNIST'):
        _NUM_CLASSES = 47
    elif (opts.dataset_name == 'EMNIST'):
        _NUM_CLASSES = 10
    else:
        _NUM_CLASSES = 1
    logits = dense_last(_NUM_CLASSES)(net)

    return keras.Model(inputs=inputs, outputs=logits)


def _build_cnn_complex(opts):
    """Builds a more copmlicated CNN Keras model."""
    layer_builders = uq_utils.get_layer_builders(opts.method,
                                                 opts.dropout_rate,
                                                 opts.num_train_examples)
    conv2d, dense_layer, dense_last, dropout_fn, dropout_fn_last = layer_builders

    if (opts.dataset_name == 'MNIST') | (opts.dataset_name == 'FashionMNIST') | (opts.dataset_name == 'EMNIST'):
        input_shape = _MNIST_SHAPE
        
    inputs = keras.layers.Input(input_shape)
    net = inputs
    net = conv2d(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape)(net)
    net = conv2d(32, (3, 3), activation='relu')(net)
    net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = dropout_fn(net)
    
    net = conv2d(64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape)(net)
    net = conv2d(64, (3, 3), activation='relu')(net)
    net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = dropout_fn(net)
    
    net = keras.layers.Flatten()(net)
    net = dense_layer(512, activation='relu')(net)
    net = dropout_fn_last(net)
    if (opts.dataset_name == 'EMNIST'):
        _NUM_CLASSES = 47
    else:
        _NUM_CLASSES = 10
    logits = dense_last(_NUM_CLASSES)(net)
    
    return keras.Model(inputs=inputs, outputs=logits)


def _build_lenet(opts):
    """Builds a LeNet Keras model."""
    layer_builders = uq_utils.get_layer_builders(opts.method,
                                                 opts.dropout_rate,
                                                 opts.num_train_examples)
    conv2d, dense_layer, dense_last, dropout_fn, dropout_fn_last = layer_builders

    if (opts.dataset_name == 'MNIST') | (opts.dataset_name == 'FashionMNIST') | (opts.dataset_name == 'EMNIST'):
        input_shape = _MNIST_SHAPE
        
    inputs = keras.layers.Input(input_shape)
    net = inputs
    net = conv2d(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(net)
    net = conv2d(64, (3, 3), activation='relu')(net)
    net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = dropout_fn(net)
    net = keras.layers.Flatten()(net)
    net = dense_layer(128, activation='relu')(net)
    net = dropout_fn_last(net)
    if (opts.dataset_name == 'EMNIST'):
        _NUM_CLASSES = 47
    else:
        _NUM_CLASSES = 10
    logits = dense_last(_NUM_CLASSES)(net)
    return keras.Model(inputs=inputs, outputs=logits)


def build_model(opts):
    """Builds (uncompiled) Keras model from ModelOptions instance."""
    return {'mlp': _build_mlp,
            'lenet': _build_lenet,
            'cnn_complex': _build_cnn_complex}[opts.architecture](opts)


def build_model_reg(opts):
    """Builds (uncompiled) Keras model from ModelOptions instance."""
    return {'mlp': _build_mlp_reg}[opts.architecture](opts)


def build_model_cifar(opts):
    """Builds (uncompiled) Keras model from ModelOptions instance."""
    return {'mlp': _build_mlp,
            'lenet': _build_lenet,
            'cnn_complex': _build_cnn_complex}[opts.architecture](opts)

