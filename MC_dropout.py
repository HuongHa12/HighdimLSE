# -*- coding: utf-8 -*-
"""
@Author: Huong Ha
@References: Majority of this code is taken from the Github:
https://github.com/google-research/google-research/tree/master/uq_benchmark_2019

"""


import os
import tensorflow as tf
import numpy as np
import math

from utils import models_lib
from utils.func_helper import record_config, load_config
from utils.func_helper import Trainlogger, ModelOptions
from utils.hyperparatune_helper import hyperpara_tune_nas

keras = tf.keras
gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
if (len(gpus) != 0):
    tf.compat.v1.config.experimental.set_memory_growth(gpus[0], True) 


# Custom loss function
def weighted_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred/y_true - 1), axis=-1)


class TrainHalt(keras.callbacks.Callback):
    """ Callback class to stop when validation error increases """
    
    def __init__(self):
        self.step = 0
        self.val_mse = []
        
    def on_epoch_end(self, batch, logs={}):
        self.step += 1
        self.val_mse.append(logs["loss"])
        if (logs["loss"] <= 0.01):
            self.model.stop_training = True


def bnn_dr_nas(X_train, Y_train, model_opts, func_name, acq_model,
               N_train_init, batch_size, h_idx, n_exp, output_dir,
               seed, lse_type, h_thrs):
    
    '''
    Inputs:
        X_train : input data, n_data x n_features
        Y_train : output data, n_data x 1
        func_name: name of the synthetic function / dataset
        h_thrs: level h
        N_train: number of random data points for starting LSE (3d)
        n_exp : index of the experiment

    Output:
        model: The MC-dropout BNN model trained on (X_train, Y_train)
    '''
    
    # Scale the output/input to be between -1 and 1
    scale_y = 1/max(np.abs(Y_train))
    Y_train = Y_train*scale_y
    scale_x = 1/(np.max(np.abs(X_train), axis=0)+1)
    X_train = X_train*scale_x
    
    # Clear session
    keras.backend.clear_session() # Safeguard just in case

    # Specify some settings
    N_train = len(Y_train)
    architecture = model_opts.architecture

    # Tune hyperparameters
    if (N_train == N_train_init):
        filename_hyper = output_dir + model_opts.dataset_name + '_' + acq_model \
                        + '_' + architecture + '_nexp'\
                        + str(n_exp) + '_ninit' + str(N_train_init) \
                        + '_Ntrain' + str(N_train) + '.json'
    else:
        filename_hyper = output_dir + model_opts.dataset_name + lse_type + '_' + acq_model \
                        + '_' + architecture + str(h_idx) + '_bs' + str(batch_size) + '_nexp'\
                        + str(n_exp) + '_ninit' + str(N_train_init) \
                        + '_Ntrain' + str(N_train) + '.json'
    
    if (os.path.isfile(filename_hyper) is True):
        print('Loading tuned hyperparameters ...')
        model_opts = load_config(filename_hyper)
        model_opts = ModelOptions(**model_opts)
        print('Done loading tuned hyperparameters ...')
        
    else:
        print('Tune hyperparameters using iNAS ...')
        # Use Bayesian Optimization to tune hyperparameters
        # To tune hyperparameters, need to use method = vanilla            
        model_opts.method = 'vanilla'
        train_Dataset = (X_train, Y_train)
        layer_sizes_opt, num_layer_opt, epoch_opt, dr_opt, lr_opt = hyperpara_tune_nas(train_Dataset,
                                                                                       model_opts,
                                                                                       batch_size=batch_size,
                                                                                       N_train_init=N_train_init,
                                                                                       N_train=N_train,
                                                                                       seed=seed,
                                                                                       h_thrs=h_thrs)
        print('Finish tuning hyperparameters ...')
        
        # Re-specify model_opts to generate uncertainty
        model_opts.method = 'dropout'
        model_opts.learning_rate = lr_opt
        model_opts.train_epochs = int(epoch_opt)
        model_opts.dropout_rate = dr_opt
        model_opts.num_layer = int(num_layer_opt)
        model_opts.mlp_layer_sizes = int(layer_sizes_opt)
        model_opts.batch_size = min(int(np.ceil(X_train.shape[0]/5)), 32)
        print('Print model_opts:')
        print(model_opts)
        print(filename_hyper)
        record_config(model_opts, filename_hyper)
    
    # Extract data and generate a random splitted train/val (70-30)
    np.random.seed(seed)
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

    # Build, compile and fit the model
    # Function for LearningRateScheduler Callback
    lr_initial = model_opts.learning_rate
    def scheduler(epoch):
        drop = 0.75
        epochs_drop = 100
        lr = lr_initial*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr

    # Build and train model
    keras.backend.clear_session()
    tf.compat.v1.set_random_seed(0)  # Set seed for each iteration
    model = models_lib.build_model_reg(model_opts)
    model_type = 'reg'
        
    model.compile(
          keras.optimizers.Adam(model_opts.learning_rate, clipnorm=1),
          loss=keras.losses.MSE,
          metrics=['mse'])

    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(
          X_train, Y_train,
          epochs=model_opts.train_epochs,
          validation_data=val_set,
          batch_size=model_opts.batch_size,
          callbacks=[Trainlogger(display=500), callback_lr],
          verbose=0
      )

    return model, scale_y, scale_x, model_opts                    
