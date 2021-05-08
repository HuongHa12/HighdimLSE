# -*- coding: utf-8 -*-
"""
author: Huong Ha

"""

import numpy as np
import math
import tensorflow as tf

from utils import models_lib

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


class TrainHalt(keras.callbacks.Callback):
    """ Callback class to stop when validation error increases """
    
    def __init__(self):
        self.step = 0
        self.val_mse = []
        
    def on_epoch_end(self, batch, logs={}):
        self.step += 1
        self.val_mse.append(logs["loss"])
        if (logs["loss"] <= 0.001):
            self.model.stop_training = True


def cross_validation_split(dataset, folds=3, seed=0):
    """ Split a dataset into k folds """

    # Split dataset to X, y
    train_testImages, train_testLabels = dataset
    fold_size = int(train_testImages.shape[0] / folds)

    np.random.seed(seed)
    permutation = np.random.permutation(train_testImages.shape[0])
    dataset_split = list()
    for i in range(folds):
        index = permutation[i*fold_size:(i+1)*fold_size]
        # dataset_split.append([train_testImages[index, :, :, :],
        #                       train_testLabels[index]])
        dataset_split.append([train_testImages[index],
        train_testLabels[index]])

    return dataset_split


def model_opts_update(model_opts, X):
    '''
    Inputs:
        model_opts: Original model_opts
        X: store the hyperparameters that need to be updated
    Output:
        model_opts: Updated model_opts
    '''
    lr = 10**X[0]
    dr = X[1]
    epoch = int(X[2])
    n_layer = int(X[3])

    model_opts.learning_rate = lr
    model_opts.train_epochs = int(epoch)
    model_opts.dropout_rate = dr
    model_opts.num_layer = n_layer
    
    return model_opts


def model_val(train_set, val_set, model_opts):
    
    '''
    Compute the model accuracy on the validation dataset
    '''

    # Clear previous session
    keras.backend.clear_session()

    # Extract the data
    X_train1, Y_train1 = train_set
    X_val1, Y_val1 = val_set
    
    # Construct the learning rate scheduler
    lr_initial = model_opts.learning_rate
    def scheduler(epoch):
        drop = 0.75
        epochs_drop = 100
        lr = lr_initial*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr

    # Build model
    tf.compat.v1.set_random_seed(0) 
    model = models_lib.build_model_reg(model_opts)
    model_type = 'reg'

    model.compile(
            keras.optimizers.Adam(model_opts.learning_rate, clipnorm=1),
            loss=keras.losses.MSE,
            metrics=['mse'])

    # Train the model
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(X_train1, Y_train1,
                        epochs=model_opts.train_epochs,
                        # NOTE: steps_per_epoch causes OOM for some reason.
                        validation_data=val_set,
                        batch_size=model_opts.batch_size,
                        callbacks=[Trainlogger(display=500), callback_lr],
                        verbose=0)

    # Compute the accuracy error on the evaluation dataset
    temp = history.history
    accuracy_val_f = -np.mean(temp['val_loss'][-1:])
    
    return accuracy_val_f, model


def hyperpara_tune_nas(train_Dataset, model_opts, batch_size,
                       N_train_init, N_train, seed, h_thrs):
    
    # Requires to tune architecture & learning rate first, then augment
    # the architecture and then tune again the dropout rate and learning rate
    # First, search the architecture
    # Set training and validation data
    np.random.seed(seed)
    X_train, Y_train = train_Dataset
    idx_pos = np.where(Y_train >= h_thrs)[0]
    idx_neg = np.where(Y_train < h_thrs)[0]
    if (len(idx_pos) == 0):
        X_train1 = X_train[0:int(2/3*X_train.shape[0]),:]
        Y_train1 = Y_train[0:int(2/3*X_train.shape[0])]
        X_val1 = X_train[int(2/3*X_train.shape[0]):,:]
        Y_val1 = Y_train[int(2/3*X_train.shape[0]):]
    else:
        idx_perm_pos = np.random.permutation(len(idx_pos))
        idx_perm_neg = np.random.permutation(len(idx_neg))
        N_pos_split = int(np.floor(0.7*len(idx_pos)))
        N_neg_split = int(np.floor(0.7*len(idx_neg)))
        X_train1 = np.concatenate((X_train[idx_pos[idx_perm_pos[0:N_pos_split]]],
                                   X_train[idx_neg[idx_perm_neg[0:N_neg_split]]]), axis=0)
        Y_train1 = np.concatenate((Y_train[idx_pos[idx_perm_pos[0:N_pos_split]]],
                                   Y_train[idx_neg[idx_perm_neg[0:N_neg_split]]]), axis=0)
        X_val1 = np.concatenate((X_train[idx_pos[idx_perm_pos[N_pos_split:]]],
                                 X_train[idx_neg[idx_perm_neg[N_neg_split:]]]), axis=0)
        Y_val1 = np.concatenate((Y_train[idx_pos[idx_perm_pos[N_pos_split:]]],
                                 Y_train[idx_neg[idx_perm_neg[N_neg_split:]]]), axis=0)
    train_set = (X_train1, Y_train1)
    val_set = (X_val1, Y_val1)

    # This is the bound for the learning rate and the drop-out rate
    epochs = 1000 # This is the maximum epoch without early stopping
    bounds = np.array([[-3, -1.5], [0.01, 0.2]])  # This is good for all the tested benchmarks
    temp = [np.random.uniform(x[0], x[1], size=25) for x in bounds]
    temp = np.asarray(temp)
    search_grid = temp.T

    # Set 3 possible architecture
    mlp_layer_sizes_ori = model_opts.mlp_layer_sizes
    num_layer_ori = model_opts.num_layer

    max_inas = 2
    mlp_layer_sizes = mlp_layer_sizes_ori
    num_layer = num_layer_ori
    for iter_inax in range(max_inas):

        print(iter_inax)
        archi_0 = (mlp_layer_sizes, num_layer)
        archi_1 = (int(np.ceil((mlp_layer_sizes*num_layer)/(num_layer+1)))+1,
                    num_layer+1)
        archi_2 = (mlp_layer_sizes+128, num_layer)
        # archi_1 = (mlp_layer_sizes, num_layer+1)
        archi_all = [archi_0, archi_1, archi_2]
                   
        # Find the validation error for each architecture
        Y_val = np.zeros((len(archi_all), search_grid.shape[0]))
        Y_val_l = []
        for idx_ar, archi in enumerate(archi_all):
            model_opts.mlp_layer_sizes = archi[0]
            model_opts.num_layer = archi[1]
    
            # For each architecture, run with different values of learning rate
            for idx_s, para_s in enumerate(search_grid):
                
                lr = 10**para_s[0]
                dr_rate = para_s[1]

                # Schedule learning rate
                lr_initial = lr
                def scheduler(epoch):
                    drop = 0.75
                    epochs_drop = 100
                    lr = lr_initial*math.pow(drop, math.floor((1+epoch)/epochs_drop))
                    return lr
                
                # Update the model_ops
                model_opts.learning_rate = lr
                model_opts.num_train_examples = X_train1.shape[0]
                model_opts.batch_size = min(int(np.ceil(X_train1.shape[0]/5)), 32)
                model_opts.dropout_rate = dr_rate
                model_opts.train_epochs = epochs
    
                # Compute the validation error
                val_error, model = model_val(train_set, val_set, model_opts)
                Y_val[idx_ar, idx_s] = val_error
                Y_val_l.append(val_error)

        # Find the architecture with the best model_val (max as we assign - for the val error)
        archi_opt, pr_opt_idx = np.unravel_index(Y_val.argmax(), Y_val.shape)
        mlp_layer_sizes, num_layer = archi_all[archi_opt]
        lr_opt = 10**search_grid[pr_opt_idx][0]
        dr_opt = search_grid[pr_opt_idx][1]
        if (archi_opt == 0):
            break

    return mlp_layer_sizes, num_layer, epochs, dr_opt, lr_opt

