# coding=utf-8
# This code is written following the code in the paper "Can You Trust Your
# Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", NeurIPS'19
# Lint as: python2, python3
"""Library for constructing MNIST and distorted MNIST datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import range

NUM_LAYER = 3
NUM_TRAIN_EXAMPLES = 50 * 1000
DUMMY_DATA_SIZE = 99
MNIST_IMAGE_SHAPE = (28, 28, 1)

DATA_OPTS_ROLL = [dict(split='test', roll_pixels=k) for k in range(2, 28, 2)]
DATA_OPTS_ROTATE = [dict(split='test', rotate_degs=k)
                    for k in range(15, 181, 15)]
DATA_OPTS_OOD = [dict(split='test', dataset_name='fashion_mnist'),
                 dict(split='test', dataset_name='not_mnist')]

DATA_OPTIONS_LIST = [
    dict(split='train'),
    dict(split='valid'),
    dict(split='test')] + DATA_OPTS_ROLL + DATA_OPTS_ROTATE + DATA_OPTS_OOD
