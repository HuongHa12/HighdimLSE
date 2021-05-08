# coding=utf-8
# This code is written following the code in the paper "Can You Trust Your
# Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", NeurIPS'19
# Lint as: python2, python3
"""Configures and runs distributional-skew UQ experiments on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils.func_helper import record_config
from utils import data_lib
from utils import models_lib


_PREDICTIONS_PER_EXAMPLE = 100
_BATCH_SIZE = 16


def get_experiment_config(method, architecture,
                          test_level, output_dir=None):
    """Returns model and data configs any dataset."""
    
    fake_data = test_level > 1
    fake_training = test_level > 0
    num_train_examples = (data_lib.DUMMY_DATA_SIZE if fake_data else
                          data_lib.NUM_TRAIN_EXAMPLES)
    num_layer = data_lib.NUM_LAYER
    model_opts = models_lib.ModelOptions(
      method=method,
      architecture=architecture,
      num_train_examples=num_train_examples,
      batch_size=_BATCH_SIZE,
      num_layer=num_layer,
      mlp_layer_sizes=[400, 400, 400],
      num_examples_for_predict=55 if fake_training else int(1e4),
      predictions_per_example=4 if fake_training else _PREDICTIONS_PER_EXAMPLE,
    )

    if method == 'vanilla':
        model_opts.predictions_per_example = 1
    return model_opts
    
    
    if output_dir:
        record_config(model_opts, output_dir+'/model_options.json')
    return model_opts
