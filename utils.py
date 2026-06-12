import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

np.set_printoptions(suppress=True, precision=6)

class myDataset():
    def __init__(self, df):
        self.df = df
        
        self.X = df.filter(regex='^x').values       
        self.C = df.filter(regex='^c').values
        self.C2 = df.filter(regex='^c2').values
        self.C3 = df.filter(regex='^c3').values
        self.S = df.filter(regex='^S').values
        self.V = df.filter(regex='^V').values
        self.W = df.filter(regex='^W').values
        self.T = df.filter(regex='^t').values # {0,1}
        self.Y = df.filter(regex='^y').values # {0,1}
        self.U = df.filter(regex='^u').values
        self.P = df.filter(regex='^P').values
        self.G = df.filter(regex='^G').values # Y0, Y1
        self.P = df.filter(regex='^P').values # (0,1)
        self.Q = df.filter(regex='^Q').values # (0,1)
        self.PR = self.P[:,1:2] # tau = P(Y1|X,T)-P(Y0)

        T = self.T * 2 - 1
        Y = self.Y * 2 - 1
        G1 = self.U[:,1:2] * 2 - 1   # Y1 * 2 - 1
        G0 = self.U[:,0:1] * 2 - 1   # Y0 * 2 - 1

        self.R = G1 > G0              # [0, 1]
        self.Z = ((T * Y) + 1) / 2    # [0, 1]
        self.A = (G0 + G1) / 2        # [-1, 0, 1]

class torchDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data.X, dtype=torch.float32)    
        self.C = torch.tensor(data.C, dtype=torch.float32)
        self.C2 = torch.tensor(data.C2, dtype=torch.float32)
        self.C3 = torch.tensor(data.C3, dtype=torch.float32)
        self.S = torch.tensor(data.S, dtype=torch.float32)
        self.V = torch.tensor(data.V, dtype=torch.float32)
        self.W = torch.tensor(data.W, dtype=torch.float32)
        self.T = torch.tensor(data.T, dtype=torch.float32)
        self.Y = torch.tensor(data.Y, dtype=torch.float32)
        self.U = torch.tensor(data.U, dtype=torch.float32)
        self.P = torch.tensor(data.P, dtype=torch.float32)
        self.G = torch.tensor(data.G, dtype=torch.float32)
        self.Q = torch.tensor(data.Q, dtype=torch.float32)
        self.PR = torch.tensor(data.PR, dtype=torch.float32)
        self.R = torch.tensor(data.R, dtype=torch.float32)
        self.Z = torch.tensor(data.Z, dtype=torch.float32)
        self.A = torch.tensor(data.A, dtype=torch.float32)

        self.variables = ['X','C','T','Y','U','P','R']#,'P','Q','PR','R','Z','A']

    def to_cpu(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cpu()')
            
    def to_cuda(self,n=0):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cuda({n})')
    
    def to_tensor(self):
        for var in self.variables:
            exec(f'self.{var} = torch.Tensor(self.{var})')
            
    def to_double(self):
        for var in self.variables:
            exec(f'self.{var} = torch.Tensor(self.{var}).double()')
            
    def to_numpy(self):
        try:
            self.detach()
            self.to_cpu()
        except:
            self.to_cpu()
        for var in self.variables:
            exec(f'self.{var} = self.{var}.numpy()')
            
    def to_pandas(self):
        var_list = []
        var_dims = []
        var_name = []
        for var in self.variables:
            exec(f'var_list.append(self.{var})')
            exec(f'var_dims.append(self.{var}.shape[1])')
        for i in range(len(self.variables)):
            for d in range(var_dims[i]):
                var_name.append(self.variables[i]+str(d))
        df = pd.DataFrame(np.concatenate(var_list, axis=1),columns=var_name)
        return df
    
    def detach(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.detach()')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'T': self.T[idx],
            'Y': self.Y[idx],
            'U': self.U[idx],
            'R': self.R[idx]
        }