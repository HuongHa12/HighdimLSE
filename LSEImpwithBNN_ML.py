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


# Function for computing the frequency of each item in a variable
def return_freq(arr):
    temp = np.unique(arr, return_counts=True)[1]
    res = np.pad(temp, (0, np.abs(len(arr)-len(temp))))
    return res


# Main function
if __name__ == '__main__':
    
    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-func_name",
                        help="name of function to be evaluated: Branin, Hartman6, etc.",
                        type=str)
    parser.add_argument("-h_index",
                        help="percentage of the threshold level to be evaluated: 0, 0.1, 0.2, etc.",
                        type=float)
    parser.add_argument("-n_exp",
                        help="The n_exp experiments: 0, 1, 2, etc.",
                        type=int)
    parser.add_argument("-acq_model",
                        help="name of acquisition function to be evaluated: LSE, Var, etc.",
                        type=str)
    parser.add_argument("-batch_size",
                        help="batch size of the active learning process: 1, 4, 8, etc.",
                        type=int)
    args = parser.parse_args()

    # Getting parameters from argparse
    bm_function = args.func_name
    h_idx = args.h_index
    n_exp = args.n_exp
    acq_model = args.acq_model
    batch_size = args.batch_size
    output_dir = 'results/'
    if (h_idx < 0) or (h_idx > 1):
        raise AssertionError("Unexpected value of 'h_idx', it needs to be from 0 to 1 !")
    
    # Load the function
    myfunction, h_thrs_list, sign = generate_hlevel_MLfunc(bm_function)
    func = myfunction.func
    bounds_all = np.asarray(myfunction.bounds)
    
    # Generate testing data and compute the true probability function
    if ('DeepPerf' in bm_function):
        X_MC, Y_MC, bounds_all, model_ML, max_X, max_Y, whole_data = data_generate_DeepPerf(bm_function)
    elif ('Protein' in bm_function):
        X_MC, Y_MC = data_generate_Protein('Protein_data/Protein_data.csv')
    else:
        raise('Not valid bm_function!')
    max_func = np.max(Y_MC)
    h_thrs_true = h_idx*max_func
    
    # Set some algorithmic parameters
    iter_mul = 30
    if ('Protein' in bm_function):
        iter_mul = 25
    n_init_points = 5*myfunction.input_dim
    max_iter = int(iter_mul*myfunction.input_dim / batch_size)

    # Specify some variables to store the data
    prob_est_all = []
    X_all = []
    Y_all = []
    print('Function: ' + bm_function)
    print('Experiment {}'.format(n_exp))
    print('Acq model: ' + acq_model)
    print('H_idx: {}'.format(h_idx))
    
    # Declare the filename
    filename = output_dir + 'results_imp_' + bm_function + '_' + acq_model + '_nexp' + str(n_exp) \
               + '_h' + str(h_idx) + '_bs' + str(batch_size) + '.pkl'
        
    # Generate inputs/outputs data
    seed = n_exp
    np.random.seed(seed)
    idx_permutation = np.random.permutation(X_MC.shape[0])
    X_idx = list(idx_permutation[0:n_init_points].copy())
    X_init = list(X_MC[idx_permutation[0:n_init_points], :])
    if ('DeepPerf' in bm_function):
        Y_init = np.abs(func(X_init, model_ML, max_X, max_Y, whole_data))
    elif ('Protein' in bm_function):
        Y_init = np.abs(func(X_init, X_MC, Y_MC))
    else:
        raise('Not valid bm_function!')
    
    # Fit the surrogate model
    # Be careful with the fitting methodology (it can be stuck in local minima)
    if (acq_model == 'ImpHLSE'):
        
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
                                                                 output_dir=output_dir, seed=seed, lse_type='imp', h_thrs=0)
        
    else:
        raise AssertionError("Unexpected value of 'acquisition function'!")
    
    # Save the statistics of MCDropoutNAS_BALD
    eps = 1e-5
    prob_est_iter = []
    F1score_iter = []
    Prec_iter = []
    Rec_iter = []
    h_thrs_iter = []
    if (acq_model == 'ImpHLSE'):
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
        h_thrs_est = h_idx*np.max(Y_est)
        y_samples_bool = y_samples_cl >=h_thrs_est

        Y_est = Y_est.ravel()
        Y_MC_h = Y_MC.ravel()
        prob_est = np.sum(y_samples_bool) / (y_samples_bool.shape[0] * y_samples_bool.shape[1])
        TP = np.sum((Y_est >= h_thrs_est) & (Y_MC_h >= h_thrs_true))
        FP = np.sum((Y_est >= h_thrs_est) & (Y_MC_h < h_thrs_true))
        FN = np.sum((Y_est < h_thrs_est) & (Y_MC_h >= h_thrs_true))
        Prec = (TP + eps)/(TP + FP + eps)
        Rec = (TP + eps)/(np.sum(Y_MC_h >= h_idx*max_func) + eps)
        F1score = 2*Prec*Rec/(Prec+Rec+eps)
        prob_est_iter.append(prob_est)
        F1score_iter.append(F1score)
        Prec_iter.append(Prec)
        Rec_iter.append(Rec)
        h_thrs_iter.append(h_thrs_est)
        print('F1 score: {}'.format(F1score))

    # The iterative sampling process
    # Specify some variables used by the GP-LSE method
    y_ci = np.tile(np.array([[-10.0**10], [10.0**10]]), X_MC.shape[0])
    H_idx = []
    L_idx = []
    Mh_idx = []
    Ml_idx = []
    X_idx_lse = list(range(0, X_MC.shape[0]))
    Z_idx = list(range(0, X_MC.shape[0]))
    U_idx = list(range(0, X_MC.shape[0]))
    for n_iter in range(1, max_iter+1):

        print('Function: ' + bm_function)
        print('Experiment {}'.format(n_exp))
        print('Acq model: ' + acq_model)
        print('H_idx: {}'.format(h_idx))
        print("Iteration {}".format(n_iter))
        
        # Find the data points to sample        
        if (acq_model == 'ImpHLSE'):
            
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
            y_samples_mean = np.mean(y_samples_cl, axis=1)

            # Set the maximum value to be the max of the estimations
            # Compute the super-level understanding gain
            # First step, compute the variance of the area of the super-level
            # set corresponding to each data point when its label varies
            max_est = np.max(y_samples_mean)
            h_thrs_est = h_idx*max_est

            # If max_est is actually the max, then compute the variance of the super area
            super_area_l = np.sum(y_samples_mean >= h_thrs_est) - (y_samples_mean >= h_thrs_est)
            y_samples_cl_bool = y_samples_cl >= h_thrs_est
            super_area_h = np.expand_dims(super_area_l, axis=-1) + y_samples_cl_bool
            super_area_h_var = np.var(super_area_h, axis=1)
            
            # If max_est is not actually the max, then compute the variance of the super area
            y_samples_max = y_samples_cl * (y_samples_cl > max_est) + max_est * (y_samples_cl <= max_est)
            h_samples = h_idx*y_samples_max
            
            max_est_bool = (y_samples_cl > max_est)
            max_est_bool_s = np.sum(max_est_bool, axis=1)
            super_area_max = super_area_h.copy()
            for i in range(super_area_h.shape[0]):
                if (max_est_bool_s[i] != 0):
                    for j in range(super_area_h.shape[1]):
                        if (y_samples_max[i, j] > max_est):
                            h_thrs_est_t = h_idx*y_samples_max[i, j]
                            super_area_max[i, j] = 1 + np.sum(y_samples_mean[0:i]>=h_thrs_est_t) + np.sum(y_samples_mean[i+1:]>=h_thrs_est_t)

            
            # Compute the entropy of the variable super_area_max
            super_area_freq = np.apply_along_axis(return_freq, 1, super_area_max)
            super_area_prob = super_area_freq/super_area_max.shape[1]
            super_area_prob_log = -np.log2(super_area_prob, out=np.zeros_like(super_area_prob),
                                           where=(super_area_prob!=0))
            pred_entropy = np.sum(super_area_prob*super_area_prob_log, axis=1)
            
            # Compute the 2nd entropy of the variable super_area_max
            # The second term is always 0 due to the fact that we always
            # set p(G=q) to be 0 or 1
            pred_entropy_exp = 0
            ys = pred_entropy + pred_entropy_exp

            # Select the best batch_size data points
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
        if (acq_model == 'ImpHLSE'):
            X_train = np.array(X_init)
            Y_train = np.array(Y_init)
            model, scale_y_bnn, scale_x_bnn, model_opts = bnn_dr_nas(X_train, Y_train, model_opts, func_name=bm_function,
                                                                     acq_model=acq_model, N_train_init=n_init_points,
                                                                     batch_size=batch_size, h_idx=h_idx, n_exp=n_exp,
                                                                     output_dir=output_dir, seed=seed, lse_type='imp',
                                                                     h_thrs=h_thrs_est) 
            
        else:
            raise AssertionError("Unexpected value of 'acquisition function'!")
        
        # Compute the interested statistics (F1 score, Precision, Recall)
        eps = 1e-5
        if (acq_model == 'ImpHLSE'):

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
            h_thrs_est = h_idx*np.max(Y_est)
            y_samples_bool = y_samples_cl >=h_thrs_est

            Y_est = Y_est.ravel()
            Y_MC_h = Y_MC.ravel()
            prob_est = np.sum(y_samples_bool) / (y_samples_bool.shape[0] * y_samples_bool.shape[1])
            TP = np.sum((Y_est >= h_thrs_est) & (Y_MC_h >= h_thrs_true))
            FP = np.sum((Y_est >= h_thrs_est) & (Y_MC_h < h_thrs_true))
            FN = np.sum((Y_est < h_thrs_est) & (Y_MC_h >= h_thrs_true))
            Prec = (TP + eps)/(TP + FP + eps)
            Rec = (TP + eps)/(np.sum(Y_MC_h >= h_idx*max_func) + eps)
            F1score = 2*Prec*Rec/(Prec+Rec+eps)

        prob_est_iter.append(prob_est)
        F1score_iter.append(F1score)
        Prec_iter.append(Prec)
        Rec_iter.append(Rec)
        h_thrs_iter.append(h_thrs_est)
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
