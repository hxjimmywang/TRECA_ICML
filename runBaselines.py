import os
import argparse
import numpy as np
import pandas as pd
from utils import myDataset, torchDataset
import torch.optim as optim
import time

# Lazy import baseline runners so one missing dependency (e.g. tensorflow)
# does not prevent running other methods.
RUNNER_PATHS = {
    "TRECA": "module.TRECA",
}


def load_runner(model_name):
    if model_name not in RUNNER_PATHS:
        raise ValueError(f"Unknown model: {model_name}")
    import importlib
    mod = importlib.import_module(RUNNER_PATHS[model_name])
    return mod.run

def set_args():
    parser = argparse.ArgumentParser(description='DecisionMaking')
    parser.add_argument('--path', type=str, default='./Data/', help='seed of the problem')
    parser.add_argument('--name', type=str, default='Twins_mono', help='seed of the problem')
    parser.add_argument('--exps', type=int, default=5, help='seed of the problem')
    parser.add_argument('--dim', type=int, default=25, help='seed of the problem')
    parser.add_argument('--num',  type=int, default=470, help='seed of the problem')
    parser.add_argument('--vnum',  type=int, default=201, help='seed of the problem')
    parser.add_argument('--tnum', type=int, default=76, help='seed of the problem')
    parser.add_argument('--trep', type=int, default=1, help='seed of the problem')
    parser.add_argument('--seed', type=int, default=2025, help='seed of the problem')
    parser.add_argument('--cvalue', type=float, default=0.5, help='seed of the problem')
    parser.add_argument('--vvalue', type=float, default=1.0, help='seed of the problem')
    parser.add_argument('--model', type=str, default='TRECA', help='seed of the problem')
    parser.add_argument('--theta', default=0.5, type=float, help='Loss Weight.')
    parser.add_argument('--mode', type=int, default=1, help='seed of the problem')
    parser.add_argument('--y0_mean', type=int, default=0, help='seed of the problem')
    parser.add_argument('--y1_mean', type=int, default=0, help='seed of the problem')
    parser.add_argument('--y0_std', type=int, default=2, help='seed of the problem')
    parser.add_argument('--y1_std', type=int, default=4, help='seed of the problem')
    ####################
    # Ours & UA
    parser.add_argument('--dim_in',default=16,type=int,help='Pre-representation layer dimensions.')
    parser.add_argument('--dim_out',default=16,type=int,help='Post-representation layer dimensions.')
    parser.add_argument('--alpha',default=0.05,type=float,help='Hyper-parameter of CFR penalty.')
    parser.add_argument('--q_alpha', default=0.9, type=float, help='Proportion of Retained Samples.')
    parser.add_argument('--abs_type', default='rank', type=str, help='Abstention type.')
    parser.add_argument('--dropout',default=0.0,type=float,help='Mask rate.')
    parser.add_argument('--preTrain',default=100,type=int,help='Mask rate.')
    parser.add_argument('--onlyF',default=1,type=int,help='Learn F separately.')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0,1.0,5.0],help='Loss weight for CFR training.')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.01,1.0,0.0],help='Loss weight for pre-train.')
    parser.add_argument('--ismono', type=int, default=1, help='Whether use monotonicity version.')
    parser.add_argument('--isunmeasured', type=int, default=0, help='Whether unmeasured confounding exists.')
    parser.add_argument('--check_dr', default=0, type=int, help='Check Mis-specificity.')
    parser.add_argument('--misspec_prop', default=0.25, type=float, help='Misspecified Propensity.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[1.0,0.0,0.0],help='Loss weight for f training.')
    parser.add_argument('--search',default=0,type=int,help='Mask rate.')
    parser.add_argument('--I',default=0,type=int,help='theta choice in [0, 1, 2].')
    # Shared Parameters
    parser.add_argument('--iterations',default=200,type=int,help='The num of cluster')
    parser.add_argument('--batch_size',default=500,type=int,help='The num of cluster')
    parser.add_argument('--output_delay',default=10,type=int,help='Mask rate.')
    parser.add_argument('--lrate', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--fiterations',default=3000,type=int,help='The num of cluster')
    parser.add_argument('--foutput_delay',default=100,type=int,help='Mask rate.')

    #############################
    parser.add_argument('--sensitive', type=float, default=1.0, help='seed of the problem')
    args = parser.parse_args()
    return args


args = set_args()

if __name__ == "__main__":
    name, exps, seed = args.name, args.exps, args.seed
    cvalue = args.cvalue
    # dataPath  = args.path
    data_name = f'{name}'


    for exp in range(exps):
        data_path = './Data/{}/{}/'.format(data_name, exp)
        os.makedirs(f'./Result/{data_name}/{exp}/train/', exist_ok=True)
        os.makedirs(f'./Result/{data_name}/{exp}/test/', exist_ok=True)
        train_pd = pd.read_csv(data_path+'train.csv', index_col=False)
        val_pd = pd.read_csv(data_path+'val.csv', index_col=False)
        test_pd  = pd.read_csv(data_path+'test.csv', index_col=False)

        train_np = myDataset(train_pd)
        valid_np = myDataset(val_pd)
        test_np = myDataset(test_pd)

        train = torchDataset(train_np)
        valid = torchDataset(valid_np)
        test = torchDataset(test_np)

  
        runner = load_runner(args.model)
        train_dfs, test_dfs = runner(exp, args, train, valid, test)
        print(train_dfs)
        print(test_dfs)
        if args.search:
            train_dfs.to_csv(f'./Result/{data_name}/{exp}/train/{args.model}_{args.onlyF}_{args.preTrain}_{params}.csv')
            test_dfs.to_csv(f'./Result/{data_name}/{exp}/test/{args.model}_{args.onlyF}_{args.preTrain}_{params}.csv')
        else:
            train_dfs.to_csv(f'./Result/{data_name}/{exp}/train/{args.model}.csv')
            test_dfs.to_csv(f'./Result/{data_name}/{exp}/test/{args.model}.csv')
