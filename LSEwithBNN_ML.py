# -*- coding: utf-8 -*-
"""
@Author: Huong Ha

"""


import numpy as np
import os

from utils import experiment
from utils.func_helper import model_predict, generate_hlevel_MLfunc, save_dict, data_generate_DeepPerf, makedirs, data_generate_Protein
from MC_dropout import bnn_dr_nas

import argparse

# Main function
if __name__ == '__main__':
    
    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-func_name",
                        help="name of function to be evaluated: Branin, Hartman6, etc.",
                        type=str)
    parser.add_argument("-h_index",
                        help="index of the threshold level to be evaluated: 0, 1, 2, etc.",
                        type=int)
    parser.add_argument("-n_exp",
                        help="The n_exp experiments: 0, 1, 2, etc.",
                        type=int)
    parser.add_argument("-acq_model",
                        help="name of acquisition function to be evaluated: LSE, Var, etc.",
                        type=str)
    parser.add_argument("-batch_size",
                        help="batch size of the active learning process: 1, 4, 8, etc. Note if gp is used as the surrogate model, batchsize is always 1.",
                        type=int)
    args = parser.parse_args()

    # Getting parameters from argparse
    bm_function = args.func_name
    h_idx = args.h_index
    n_exp = args.n_exp
    acq_model = args.acq_model
    batch_size = args.batch_size
    output_dir = 'results/'
    if (h_idx < 0) or (h_idx > 9):
        raise AssertionError("Unexpected value of 'h_idx', it needs to be from 0 to 9 !")
    
    # Load the function
    myfunction, h_thrs_list, sign = generate_hlevel_MLfunc(bm_function)
    func = myfunction.func
    bounds_all = np.asarray(myfunction.bounds)
    h_thrs = h_thrs_list[h_idx]

    # Generate testing data and compute the true probability function
    for idx in range(1):
        if ('DeepPerf' in bm_function):
            X_MC, Y_MC, bounds_all, model_ML, max_X, max_Y, whole_data = data_generate_DeepPerf(bm_function)
        elif ('Protein' in bm_function):
            X_MC, Y_MC = data_generate_Protein('Protein_data/Protein_data.csv')
        else:
            raise('Not valid bm_function!')

    # Set some algorithmic parameters
    iter_mul = 40
    if ('Protein' in bm_function):
        iter_mul = 25
    n_init_points = 5*myfunction.input_dim
    max_iter = int(iter_mul*myfunction.input_dim / batch_size)

    X_all = []
    Y_all = []
    print('Function: ' + bm_function)
    print('Experiment {}'.format(n_exp))
    print('Acq model: ' + acq_model)
    print('H_idx: {}'.format(h_idx))
    print('Threshold level: {}'.format(h_thrs))
    
    # Declare the filename
    filename = output_dir + 'results_' + bm_function + '_' + acq_model + '_nexp' + str(n_exp) \
               + '_h' + str(h_idx) + '_bs' + str(batch_size) + '.pkl'
    
    # Generate inputs/outputs data
    seed = n_exp
    np.random.seed(seed)
    idx_permutation = np.random.permutation(X_MC.shape[0])
    X_idx = list(idx_permutation[0:n_init_points].copy())
    X_init = list(X_MC[idx_permutation[0:n_init_points], :])
    if ('DeepPerf' in bm_function) | ('UCI' in bm_function):
        Y_init = np.abs(func(X_init, model_ML, max_X, max_Y, whole_data))
    elif ('Protein' in bm_function):
        Y_init = np.abs(func(X_init, X_MC, Y_MC))
    else:
        raise('Not valid bm_function!')

    # Fit the surrogate model
    # Be careful with the fitting methodology (it can be stuck in local minima)
    if (acq_model == 'ExpHLSE'):
        
        X_train = np.array(X_init)
        Y_train = np.array(Y_init)
        
        # Specify model options
        method = 'dropout'
        architecture = 'mlp'  # The architecture of the BNN
        test_level = 1   # Zero -> no testing. One -> testing with real data.
                         # Two is for testing with fake data.
        output_dir = os.path.join(output_dir)
        makedirs(output_dir)
        model_opts = experiment.get_experiment_config(method,
                                                      architecture,
                                                      test_level=test_level,
                                                      output_dir=output_dir)
        
        # Tune hyperparameters using iNAS
        model_opts.predictions_per_example = 200
        model_opts.mlp_layer_sizes = 256  # units per layer
        model_opts.dataset_name = bm_function
        model_opts.batch_size = min(int(np.ceil(X_train.shape[0]/5)), 32)
        model_opts.input_size = X_train.shape[1]
        model_opts.num_layer = 1
        model_opts.learning_rate = 'NA'
        model_opts.dropout_rate = 'NA'

        model, scale_y_bnn, scale_x_bnn, model_opts = bnn_dr_nas(X_train, Y_train, model_opts, func_name=bm_function,
                                                                 acq_model=acq_model, N_train_init=n_init_points,
                                                                 batch_size=batch_size, h_idx=h_idx, n_exp=n_exp,
                                                                 output_dir=output_dir, seed=seed, lse_type='',
                                                                 h_thrs=h_thrs)
        
    else:
        raise AssertionError("Unexpected value of 'acquisition function'!")    

    # Compute the interested statistics (F1 score, Precision, Recall)
    prob_est_iter = []
    F1score_iter = []
    Prec_iter = []
    Rec_iter = []
    h_thrs_iter = []
    eps = 1e-5
    X_idx_TV = list(range(0, X_MC.shape[0]))
    H_idx_TV = []
    L_idx_TV = []
    
    if (acq_model == 'ExpHLSE'):

        poolDataset = (X_MC*scale_x_bnn, X_MC[:, 0:1])
        y_samples = model_predict(model,
                                  poolDataset,
                                  50,
                                  _BATCH_SIZE_FOR_PREDICT = 5000,
                                  _NUM_CLASSES=1)
        y_samples = y_samples/scale_y_bnn
        y_samples_cl = y_samples[0][:, :, 0]
        y_samples_cl[X_idx, :] = np.tile(np.expand_dims(Y_init.ravel(), axis=1), y_samples_cl.shape[1])
        Y_est = np.mean(y_samples_cl, axis = 1)
        y_samples_bool = y_samples_cl >= h_thrs

        Y_est = Y_est.ravel()
        Y_MC_h = Y_MC.ravel()
        prob_est = np.sum(y_samples_bool) / (y_samples_bool.shape[0] * y_samples_bool.shape[1])
        TP = np.sum((Y_est >= h_thrs) & (Y_MC_h >= h_thrs))
        FP = np.sum((Y_est >= h_thrs) & (Y_MC_h < h_thrs))
        FN = np.sum((Y_est < h_thrs) & (Y_MC_h >= h_thrs))
        Prec = (TP + eps)/(TP + FP + eps)
        Rec = (TP + eps)/(TP + FN + eps)
        F1score = 2*Prec*Rec/(Prec+Rec+eps)
    
        prob_est_iter.append(prob_est)
        F1score_iter.append(F1score)
        Prec_iter.append(Prec)
        Rec_iter.append(Rec)
        h_thrs_iter.append(h_thrs)
        print('F1 score: {}'.format(F1score))

    # The sampling process
    epsilon = 0.02*np.max(Y_MC)
    y_ci = np.tile(np.array([[-10.0**10], [10.0**10]]), X_MC.shape[0])
    H_idx = []
    L_idx = []
    U_idx = list(range(0, X_MC.shape[0]))
    r = 0.1
    eta = 1
    delta_bar = 0
    X_idx_TV = list(range(0, X_MC.shape[0]))
    for n_iter in range(1, max_iter+1):

        print('Function: ' + bm_function)
        print('Experiment {}'.format(n_exp))
        print('Acq model: ' + acq_model)
        print('H_idx: {}'.format(h_idx))
        print('Threshold level: {}'.format(h_thrs))
        print("Iteration {}".format(n_iter))
        
        # Find the data points to sample        
        if (acq_model == 'ExpHLSE'):
            
            # Compute the forward passes from the BNN
            x_tries = X_MC.copy()
            poolDataset = (x_tries*scale_x_bnn, x_tries[:, 0:1])
            y_samples = model_predict(model,
                                      poolDataset,
                                      50,
                                      _BATCH_SIZE_FOR_PREDICT=1000,
                                      _NUM_CLASSES=1)
            y_samples = y_samples/scale_y_bnn
            
            y_samples_cl = y_samples[0][:, :, 0]
            y_samples_bool = y_samples_cl >= h_thrs
            probs_pool_all = y_samples_bool.astype(float)
            probs_pool_all = np.concatenate((probs_pool_all[..., np.newaxis],
                                             1-probs_pool_all[..., np.newaxis]), axis=2)              
            probs_class = np.mean(probs_pool_all, axis=1)
            probs_class_log = -np.log2(probs_class, out=np.zeros_like(probs_class), where=(probs_class!=0))
            pred_entropy = np.sum(probs_class*probs_class_log, axis=1)
            
            probs_pool_all_log = -np.log2(probs_pool_all, out=np.zeros_like(probs_pool_all),
                                          where=(probs_pool_all!=0))
            temp = np.sum(probs_pool_all * probs_pool_all_log, axis=1)
            temp = np.sum(temp, axis=1)
            pred_entropy_exp = 1/probs_pool_all.shape[1]*temp
            ys = pred_entropy - pred_entropy_exp

            idx_max = ys.argmax()
            x_max = x_tries[idx_max]
            max_acq = ys.max()
            
            ys_sort_idx = np.argsort(ys)[::-1]
            cnt = 0
            x_max_list = []
            for bs in range(0, batch_size):
                idx_max = ys_sort_idx[cnt]
                x_max = x_tries[idx_max]
                while any((x_max == x).all() for x in X_init):
#                    print('Duplicate samples, pick a sample with next largest value!')
                    cnt += 1
                    idx_max = ys_sort_idx[cnt]
                    x_max = x_tries[idx_max]

                x_max_list.append(x_tries[idx_max])
                X_idx.append(idx_max)
                cnt += 1
            X_next = x_max_list
        
        else:
            raise AssertionError("Unexpected value of 'acquisition function'!")
        
        # Get the labels of the interested points
        if not isinstance(X_next, list):
            X_init.append(X_next)
        else:
            for xnext_idx in range(0, len(X_next)):
                X_init.append(X_next[xnext_idx])
        if ('DeepPerf' in bm_function):
            Y_init = np.abs(func(X_init, model_ML, max_X, max_Y, whole_data))
        elif ('Protein' in bm_function):
            Y_init = np.abs(func(X_init, X_MC, Y_MC))
        else:
            raise('Not valid bm_function!')
        
        # Update the surrogate model with the new labels
        if (acq_model == 'ExpHLSE'):
            X_train = np.array(X_init)
            Y_train = np.array(Y_init)
            model, scale_y_bnn, scale_x_bnn, model_opts = bnn_dr_nas(X_train, Y_train, model_opts, func_name=bm_function,
                                                                     acq_model=acq_model, N_train_init=n_init_points,
                                                                     batch_size=batch_size, h_idx=h_idx, n_exp=n_exp,
                                                                     output_dir=output_dir, seed=seed, lse_type='',
                                                                     h_thrs=h_thrs)
            
        else:
            raise AssertionError("Unexpected value of 'acquisition function'!")
        
        # Compute the interested statistics (F1 score, Precision, Recall)
        eps = 1e-5        
        if (acq_model == 'ExpHLSE'):
            
            poolDataset = (X_MC*scale_x_bnn, X_MC[:, 0:1])
            y_samples = model_predict(model,
                                      poolDataset,
                                      50,
                                      _BATCH_SIZE_FOR_PREDICT = 1000,
                                      _NUM_CLASSES=1)
            y_samples = y_samples/scale_y_bnn
            y_samples_cl = y_samples[0][:, :, 0]
            y_samples_cl[X_idx, :] = np.tile(np.expand_dims(Y_init.ravel(), axis=1), y_samples_cl.shape[1])
            Y_est = np.mean(y_samples_cl, axis = 1)
            y_samples_bool = y_samples_cl >= h_thrs

            Y_est = Y_est.ravel()
            Y_MC_h = Y_MC.ravel()
            prob_est = np.sum(y_samples_bool) / (y_samples_bool.shape[0] * y_samples_bool.shape[1])
            TP = np.sum((Y_est >= h_thrs) & (Y_MC_h >= h_thrs))
            FP = np.sum((Y_est >= h_thrs) & (Y_MC_h < h_thrs))
            FN = np.sum((Y_est < h_thrs) & (Y_MC_h >= h_thrs))
            Prec = (TP + eps)/(TP + FP + eps)
            Rec = (TP + eps)/(TP + FN + eps)
            F1score = 2*Prec*Rec/(Prec+Rec+eps)

        prob_est_iter.append(prob_est)
        F1score_iter.append(F1score)
        Prec_iter.append(Prec)
        Rec_iter.append(Rec)
        h_thrs_iter.append(h_thrs)
        print('F1 score: {}'.format(F1score))
        
        # Create a dictionary to store immediate results
        results = dict()
        results['Prob_est_iter'] = prob_est_iter
        results['F1_score_iter'] = F1score_iter
        results['Precision_iter'] = Prec_iter
        results['Recall_iter'] = Rec_iter

        results['X_samples'] = np.array(X_init)
        results['Y_samples'] = np.array(Y_init)
        results['h_level'] = h_thrs_iter
        
        # Save results every iteration as each iteration takes long time to run
        print('Save results ...')
        save_dict(results, filename)  
    
    



    