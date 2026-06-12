import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def set_seed(seed=2024):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FNetwork(nn.Module):
    def __init__(self, x_dim, h_dim=32):
        super(FNetwork, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim) 
        self.fc3 = nn.Linear(h_dim, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        output = torch.sigmoid(x)
        return output


def run(exp, args, train, valid, test, variable='S'):
    batch_size = args.batch_size
    lr = args.lr
    num_epoch = args.num_epoch

    set_seed(args.seed)

    train_loader = DataLoader(train, batch_size=batch_size)

    net = FNetwork(args.dim, args.h_dim)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epoch):
        for idx, inputs in enumerate(train_loader):
            X = inputs['X']
            if variable=='S':
                C = inputs['C']
            else:
                C = inputs['V']

            prediction = net(X) 
            loss = loss_func(prediction, C)

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()    

    hat_C_train = net(train.X)
    # MSE_train = loss_func(hat_C_train, train.C)

    hat_C_test = net(test.X)
    # MSE_test = loss_func(hat_C_test, test.C)

    return hat_C_train, hat_C_test