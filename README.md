# HighdimLSE

This repository contains the code for the method ExpHLSE and ImpHLSE proposed in the paper 'High Dimensional Level Set Estimation with Bayesian Neural Network', AAAI'2021, Ha et al. These are the two methods that solve the high-dimensional Level Set Estimation (LSE) problem via Bayesian Neural Network (BNN).

## Citing ExpHLSE and ImpHLSE
If you find our code useful, please kindly cite our paper. 

```
@inproceedings{Ha2021,
  title={High Dimensional Level Set Estimation with Bayesian Neural Network},
  author={Ha, Huong and Gupta, Sunil and Rana, Santu and Venkatesh, Svetha},
  booktitle={The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  year={2021}
}
```

## Prerequisites

- Python 3 (tested with Python 3.6.x)
- Tensorflow (tested with tensorflow 1.15.0)
- Other packages versions are included in the file LSE_BNN_packages.txt

## Test Cases

ExpHLSE and ImpHLSE have been evaluated on 3 synthetic benchmark functions: Levy10, Ackley10, Alpine10 and three real-world experiments: Material Design, Protein Selection, and Algorithmic Assurance, using various level set thresholds. More details about the experiment setting can be found in the paper. The codes to implement these synthetic functions and ML models can be found in the scripts ```functions.py``` and ```functions_ml_model.py```.

## Usage
The folder has four main scripts to run the explicit and implict algorithms for synthetic and real-world experiments:
1) ```LSEwithBNN.py```: Script to run the proposed explicit algorithm (ExpHLSE) on synthetic functions
2) ```LSEwithBNN_ML.py```: Script to run the proposed explicit algorithm (ExpHLSE) on real-world experiments
3) ```LSEImpwithBNN.py```: Script to run the proposed implicit algorithm (ImpHLSE) on synthetic functions
4) ```LSEImpwithBNN_ML.py```: Script to run the proposed implicit algorithm (ImpHLSE) aon real-world experiments

### A - TO RUN THE EXPLICIT ALGORITHMS
To get the results for a FUNCTION with the explicit level set threshold at the index H_IDX, for experiment N_EXP with batch size BS, use the following command:
python LSEwithBNN.py -func_name FUNCTION -h_index H_IDX -n_exp N_EXP -acq_model METHOD -batch_size BS
OR python LSEwithBNN_ML.py -func_name FUNCTION -h_index H_IDX -n_exp N_EXP -acq_model METHOD -batch_size BS

Possible METHODs: ExpHLSE
Possible FUNCTIONs: Levy10, Ackley10, Alpine10, Protein_wl, DeepPerf_hsmgp_0, DeepPerf_hipacc_0
Possible H_IDX: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 corresponding to the cases of the super_level set being [0, 1, 5, 10, 20, 100, 99, 95, 90, 80]% of the function domain (in the main paper, H_IDX is set as 4)
Possible N_EXP: 0, 1, 2, .... (in the main paper, we show the results of 3 experiments with N_EXP = 0, 1, 2)
Possible BS: 1, 2, .... (in the main paper, for ExpHLSE and gp_TruVar, we set BS=10d where d is the dimension of the input)

Examples (this will reproduce the results in the paper for Ackley10, and Protein problem -- with experiment 0):
python LSEwithBNN.py -func_name Ackley10 -h_index 4 -n_exp 0 -acq_model ExpHLSE -batch_size 100
python LSEwithBNN.py -func_name Protein_wl -h_index 4 -n_exp 0 -acq_model ExpHLSE -batch_size 200

B ----- TO RUN THE IMPLICIT ALGORITHMS
To get the results for a FUNCTION with the implicit level set threshold at the H_IDX, for experiment N_EXP with batch size BS, use the following command:
python LSEImpwithBNN.py -func_name FUNCTION -h_index H_IDX -n_exp N_EXP -acq_model METHOD -batch_size BS
OR python LSEImpwithBNN_ML.py -func_name FUNCTION -h_index H_IDX -n_exp N_EXP -acq_model METHOD -batch_size BS

Possible METHODs: ImpHLSE
Possible FUNCTIONs: Levy10, Ackley10, Alpine10, Protein_wl, DeepPerf_hsmgp_0, DeepPerf_hipacc_0
Possible H_IDX: Any number from 0 to 1 (in the main paper, we set H_IDX such that the level is similiar to the corresponding explicit LSE problem, so our choices of H_IDX are as follows: Ackley10: 0.9689, Levy10: 0.35, Alpine10: 0.5, DeepPerf_hsmgp: 0.091, DeepPerf_hipacc: 0.109, Protein_wl: 0.904)
Possible N_EXP: 0, 1, 2, .... (in the main paper, we show the results of 3 experiments with N_EXP = 0, 1, 2)
Possible BS: 1, 2, .... (in the main paper, for ImpHLSE and gp_TruVar, we set BS=10d where d is the dimension of the input)

Examples:
python LSEImpwithBNN.py -func_name Ackley10 -h_index 0.9689 -n_exp 0 -acq_model ImpHLSE -batch_size 100
python LSEImpwithBNN_ML.py -func_name Protein_wl -h_index 0.904 -n_exp 0 -acq_model ImpHLSE -batch_size 200



