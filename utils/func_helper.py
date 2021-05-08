# -*- coding: utf-8 -*-
"""
author: Huong Ha
@References: Majority of this code is taken from the Github:
https://github.com/google-research/google-research/tree/master/uq_benchmark_2019

"""

import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io

import functions
import functions_ML
from numpy import genfromtxt
from utils.general import system_samplesize, seed_generator
from mlp_sparse_model import MLPSparseModel
from emnist import extract_training_samples, extract_test_samples
from collections import Counter

from absl import logging
from six.moves import cPickle as pickle
import attr
import scipy.special
import os
import json

keras = tf.keras
gfile = tf.io.gfile


# Callback class during training
class Trainlogger(keras.callbacks.Callback):
    """ Callback class to display classifier performance during training """

    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self.metric_inst = {}

    def on_epoch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                # self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
                self.metric_inst[k] = logs[k]
                
        if (self.step % self.display == 0) | (self.step == 1):
            
            metrics_log_inst = ''
            for (k, v) in self.metric_inst.items():
                val = v
                if abs(val) > 1e-3:
                    metrics_log_inst += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log_inst += ' - %s: %.4e' % (k, val)
            
            # print('Epochs: {} ... Average ... {}'.format(self.step, metrics_log))
            print('Epochs: {} ... Instant ... {}'.format(self.step, metrics_log_inst))
            self.metric_cache.clear()
            self.metric_inst.clear()


@attr.s
class MnistDataOptions(object):
    split = attr.ib()
    dataset_name = attr.ib('mnist')
    roll_pixels = attr.ib(0)
    rotate_degs = attr.ib(0)


@attr.s
class ModelOptions(object):
    """Parameters for model construction and fitting."""
    num_layer = attr.ib()
    train_epochs = attr.ib()
    num_train_examples = attr.ib()
    batch_size = attr.ib()
    learning_rate = attr.ib()
    method = attr.ib()
    architecture = attr.ib()
    mlp_layer_sizes = attr.ib()
    dropout_rate = attr.ib()
    num_examples_for_predict = attr.ib()
    predictions_per_example = attr.ib()
    dataset_name = attr.ib()
    input_size = attr.ib()


def _crop_center(images, size):
    height, width = images.shape[1:3]
    i0 = height // 2 - size // 2
    j0 = width // 2 - size // 2
    return images[:, i0:i0 + size, j0:j0 + size]


def generate_hlevel_func(bm_function):
    
    # The corresponding h-threshold for the failure probability being [0, 1, 5, 10, 20, 100, 99, 95, 90, 80]
    if (bm_function == 'Alpine10'):
        myfunction = functions.alpine(10)
        h_thrs_list = [76.04699347219815, 50.98420778222492, 44.79947384905441, 41.54907469884088, 37.68340375926266,
                       3.0277925312926275, 14.337478653925471, 18.451052965079207, 20.88200722173364, 24.03840918221876]
        sign = -1
    elif (bm_function == 'Ackley10'):
        myfunction = functions.ackley(10)
        h_thrs_list = [22.168974596979595, 21.819036138457378, 21.680282110195844, 21.595118278189634, 21.479578840833277,
                       12.082318857638047, 19.762729053142312, 20.38778918546939, 20.626778039811253, 20.863106443916987]
        sign = -1
    elif (bm_function == 'Levy10'):
        myfunction = functions.Levy(10)
        h_thrs_list = [475.0465027868454, 266.8203483500954, 218.69511329895465, 194.34782865332252, 166.2374142826767,
                       3.6077953962697595, 30.603099182609224, 47.997214432079794, 59.610831294850485, 76.35266446267853]
        sign = -1
    else:
        raise AssertionError("Unexpected value of 'bm_function'!")

    max_func = h_thrs_list[0]
    
    return myfunction, max_func, h_thrs_list, sign


def generate_hlevel_MLfunc(bm_function):
    
    if ('DeepPerf' in bm_function):
        _, sys_name, idx_ss = bm_function.split('_')
        idx_ss = int(idx_ss)
        
        # Read and extract data
        print('Read whole dataset from csv file ...')
        dir_data = 'deepperf_experiment/' + sys_name + '_AllNumeric.csv'
        print('Dataset: ' + dir_data)
        whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
        (N, n) = whole_data.shape
        n = n-1
    
    elif ('Protein' in bm_function):
        
        # Read the data
        filename = 'Protein_data/Protein_data.csv'
        data_all_pd = pd.read_csv(filename)
        whole_data = data_all_pd.values
        X_data_seq = whole_data[:, 0]

        # Get all the possible characters in the sequence
        char_all = []
        for item in X_data_seq:
            char_all += item
        char_distinct = set(char_all)
        
        # Convert the sequence in X_data to bags of words
        # The new input data will consist of the number of distinct characters and 
        # the corresponding 
        n = len(char_distinct)
        myfunction = functions_ML.Protein_wl_Test(n)
    
    else:
        raise('Invalid bm_function!')
    
    # The corresponding h-threshold for the failure probability being [0, 1, 5, 10, 20, 100, 99, 95, 90, 80]
    if (bm_function == 'DeepPerf_hsmgp_0'): #14
        myfunction = functions_ML.DeepPerf_Assurance_Test(n)
        h_thrs_list = [252.398220703125, 163.79725624999975, 67.33930664062484, 42.44147792968752, 22.9844081542969,
                       0.0046920166015524956, 0.1825607275390661, 0.8584244873046658, 1.9698970458984888, 3.924331054687491]
        sign = 'NA'
    elif (bm_function == 'DeepPerf_hipacc_0'): #33
        myfunction = functions_ML.DeepPerf_Assurance_Test(n)
        h_thrs_list = [54.439930694580084, 28.328944942932132, 15.916741574096672, 10.438403762817384, 5.926678015136724,
                       0.00011256408691551201, 0.023947037963865617, 0.15022379455566492, 0.3031221145629882, 0.6267720947265616]
        sign = 'NA'
    elif (bm_function == 'Protein_wl'):
        myfunction = functions_ML.Protein_wl_Test(n) #20
        h_thrs_list = [622.0, 613.0, 602.0, 579.0, 562.0, 436.0, 462.95, 482.25, 500.0, 516.0]
        sign = 'NA'
    return myfunction, h_thrs_list, sign


def data_generate_DeepPerf(bm_function):
    _, sys_name, idx_ss = bm_function.split('_')
    idx_ss = int(idx_ss)
    
    # Read and extract data
    print('Read whole dataset from csv file ...')
    dir_data = 'deepperf_experiment/' + sys_name + '_AllNumeric.csv'
    print('Dataset: ' + dir_data)
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    (N, n) = whole_data.shape
    n = n-1
    myfunction = functions_ML.DeepPerf_Assurance_Test(n)
    
    # Set training data
    n_exp = 30
    m = 1
    sample_size_all = list(system_samplesize(sys_name))
    N_train = sample_size_all[idx_ss]
    seed_init = seed_generator(sys_name, N_train)
    seed = seed_init*n_exp + m
    np.random.seed(seed)
    permutation = np.random.permutation(N)
    training_index = permutation[0:N_train]
    training_data = whole_data[training_index, :]
    X_train = training_data[:, 0:n]
    Y_train = training_data[:, n][:, np.newaxis]
    
    # Scale X_train and Y_train
    max_X = np.amax(X_train, axis=0)
    if 0 in max_X:
        max_X[max_X == 0] = 1
    X_train = np.divide(X_train, max_X)
    max_Y = np.max(Y_train)/100
    if max_Y == 0:
        max_Y = 1
    Y_train = np.divide(Y_train, max_Y)
    
    # Generate testing data
    # Load the stored tuned hyperparameters
    filename = 'deepperf_experiment/result_' + sys_name + '_AutoML_veryrandom.npy'
    result_temp = np.load(filename, allow_pickle=True).tolist()
    result_sample = result_temp[idx_ss]
    n_layer_opt = result_sample['n_layer_all'][m-1]
    lambda_f = result_sample['lambda_all'][m-1]
    
    config = dict()
    config['num_neuron'] = 128
    config['num_input'] = n
    config['num_layer'] = n_layer_opt
    config['lambda'] = lambda_f
    config['verbose'] = 1
    
    dir_output = 'deepperf_experiment/model/' + sys_name + str(idx_ss) + '/'
    model = MLPSparseModel(config, dir_output)
    model.build_train()
    model.restore_session(dir_output + "model.weights/")
    
    # Compute the threshold levels
    func = myfunction.func
    bounds_all = np.asarray(myfunction.bounds)
    testing_index = np.setdiff1d(np.array(range(N)), training_index)
    testing_data = whole_data[testing_index, :]
    X_MC = testing_data[:, 0:n]
    Y_MC = func(X_MC, model, max_X, max_Y, whole_data)

    return X_MC, Y_MC, bounds_all, model, max_X, max_Y, whole_data


def data_generate_Protein(filename):
    
    data_all_pd = pd.read_csv(filename)
    whole_data = data_all_pd.values
    X_data_seq = whole_data[:, 0]
    Y_data = whole_data[:, 1]
    Y_data = Y_data[..., np.newaxis]
    
    # Get all the possible characters in the sequence
    char_all = []
    for item in X_data_seq:
        char_all += item
    char_distinct = set(char_all)
    
    # Convert the sequence in X_data to bags of words
    # The new input data will consist of the number of distinct characters and 
    # the corresponding 
    N = len(Y_data)
    n = len(char_distinct)
    X_data = np.zeros((N, n))
    for X_idx, item in enumerate(X_data_seq):
        counter = Counter(item)
        for idx, char in enumerate(char_distinct):
            X_data[X_idx, idx] = counter[char]

    return X_data, Y_data

def build_dataset_manual(dataset_name, opts):
    """ Build train, valid, test datasets based on model options """
    opts = MnistDataOptions(**opts)
    logging.info('Building dataset with options: %s', opts)

    # Load train and test data (MNIST)
    # train: 60k instances
    # test: 10k instances
    if (dataset_name == 'MNIST'):
        train, test = tf.keras.datasets.mnist.load_data()
    elif (dataset_name == 'EMNIST'):
        train = extract_training_samples('bymerge')
        test = extract_test_samples('bymerge')
    else:
        raise ValueError('Dataset is not supported!')

    # ALL we need is the test dataset so the train/valid doesnot matter here
    if (opts.split == 'train'):
        images, labels = train[0][0: 50000], train[1][0: 50000]
    elif (opts.split == 'valid'):
        images, labels = train[0][50000: 60000], train[1][50000: 60000]
    elif (opts.split == 'test'):
        images, labels = test
    else:
        raise ValueError('opts.split is not valid!')

    # Change images size
    if (dataset_name == 'MNIST') | (dataset_name == 'EMNIST'):
        images = np.expand_dims(images, -1)
    images = images/255

    if opts.rotate_degs:
        images = scipy.ndimage.rotate(images, opts.rotate_degs, axes=[-2, -3])
        images = _crop_center(images, 28)
    if opts.roll_pixels:
        images = np.roll(images, opts.roll_pixels, axis=-2)

    return images, labels


def model_predict(model, dataset_test_all, _PREDICTS_PER_EXAMPLE,
                  _BATCH_SIZE_FOR_PREDICT, _NUM_CLASSES):
    '''
    Compute model prediction
    Inputs:
        model: trained model for prediction
        dataset_test_all: dataset (images, labels)
        _BATCH_SIZE_FOR_PREDICT: batch size for each prediction
        _NUM_CLASSES: number of classes of the output
    Outputs:
        probs_all: predicted probability of each class of the dataset_test
        probs_argmax_class: predicted class for the dataset_test
    '''
    N_batch = int(np.ceil(dataset_test_all[0].shape[0] /
                          _BATCH_SIZE_FOR_PREDICT))

    logits_samples = np.zeros((dataset_test_all[0].shape[0],
                               _PREDICTS_PER_EXAMPLE,
                               _NUM_CLASSES))
    count = 0
    for i in range(N_batch):
        images = dataset_test_all[0][i*_BATCH_SIZE_FOR_PREDICT:
                                     (i+1)*_BATCH_SIZE_FOR_PREDICT]
        logits_samples_temp = np.stack(
                [model.predict(images) for _ in range(_PREDICTS_PER_EXAMPLE)],
                axis=1)  # shape: [batch_size, num_samples, num_classes]
        logits_samples[count*_BATCH_SIZE_FOR_PREDICT:(count+1)*_BATCH_SIZE_FOR_PREDICT, :, :] = logits_samples_temp.copy()
        count += 1

    if (_NUM_CLASSES > 1):
        probs_all = scipy.special.softmax(logits_samples, axis=-1)
        probs_argmax_class = np.argmax(probs_all, axis=2)
        if probs_argmax_class.shape[1] == 1:
            probs_argmax_class = probs_argmax_class.ravel()
    else:
        probs_all = logits_samples
        probs_argmax_class = logits_samples

    return probs_all, probs_argmax_class


class _SimpleJsonEncoder(json.JSONEncoder):

    def default(self, o):
        return o.__dict__


def json_dumps(x):
    return json.dumps(x, indent=2, cls=_SimpleJsonEncoder)


def record_config(config, path):
    out = json_dumps(config)
    logging.info('Recording config to %s\n %s', path, out)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # with open(path, 'w', encoding='utf-8') as f:
    #     json.dump(out, f, ensure_ascii=False, indent=4)
    with gfile.GFile(path, 'w') as fh:
        fh.write(out)


def makedirs(path):
    '''
    Make a directory (if it doesn't exist)
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(path):
    logging.info('Loading config from %s', path)
    with gfile.GFile(path) as fh:
        return json.loads(fh.read())


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
