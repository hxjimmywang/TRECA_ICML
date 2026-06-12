import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from .module import Net
from .utils import wasserstein, mmd2_lin
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.float_format', '{:.3f}'.format)


def set_seed(seed=2025):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(exp, args, train, valid, test):
    set_seed(args.seed)
    print(f'The {exp}-th experiments: ', train.X.shape, valid.X.shape, test.T.shape)
    a1, a2, a3 = args.alphas
    b1, b2, b3 = args.betas
    g1, g2, g3 = args.gammas
    ismono = args.ismono
    isunmeasured = args.isunmeasured
    theta = args.theta

    ''' Define model graph '''
    dropout = 0.0
    n, x_dim = train.X.shape
    model = Net(n, x_dim, dropout, args)


    ''' Set up optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.97, last_epoch=-1)

    # Train CFRNet only
    parameterv2 = list(model.rep_net.parameters()) + list(model.y0_net.parameters()) + list(model.y1_net.parameters())
    optimizerv2 = torch.optim.Adam(parameterv2, lr=args.lrate)
    schedulerv2 = torch.optim.lr_scheduler.StepLR(optimizerv2, 100, gamma=0.97, last_epoch=-1)

    # Train propensity only
    parametervp = list(model.prop_net.parameters())
    optimizervp = torch.optim.Adam(parametervp, lr=args.lrate)
    schedulervp = torch.optim.lr_scheduler.StepLR(optimizervp, 100, gamma=0.97, last_epoch=-1)

    # Train f only
    parametervf = list(model.f_cost.parameters())
    optimizervf = torch.optim.Adam(parametervf, lr=args.lrate)
    schedulervf = torch.optim.lr_scheduler.StepLR(optimizervf, 100, gamma=0.97, last_epoch=-1)

    # Train beta only
    parametervb = [model.beta]
    optimizervb = torch.optim.Adam(parametervb, lr=args.lrate)
    schedulervb = torch.optim.lr_scheduler.StepLR(optimizervb, 100, gamma=0.97, last_epoch=-1)


    ''' Compute treatment probability'''
    p_treated = torch.mean(train.T)

    ''' Train for multiple iterations '''
    alpha = args.alpha
    q_alpha = args.q_alpha
    loss_valid = nn.BCELoss()
    train_dfs = pd.DataFrame()
    test_dfs = pd.DataFrame()
    for epoch in range(args.iterations):
        model.train()

        ''' Fetch sample '''
        x_batch = train.X
        theta_x = train.C
        t_batch = train.T
        y_batch = train.Y

        w_t = t_batch / (2 * p_treated)
        w_c = (1 - t_batch) / (2 * 1 - p_treated)
        sample_weight = w_t - w_t + 1

        '''Train Propensity'''
        if epoch <= args.preTrain:

            ''' Train tau '''
            for _ in range(1):
                rep_batch, y0_pred, y1_pred, y_pred, prop_pred, f_x = model(x_batch, t_batch)
                # Propensity
                loss_propn = nn.BCELoss(weight=sample_weight)
                prop_err = loss_propn(prop_pred, t_batch)
                optimizervp.zero_grad()
                prop_err.backward()
                optimizervp.step()
                schedulervp.step()

                # CFR loss
                loss_fn = nn.BCELoss(weight=sample_weight)
                risk = loss_fn(y_pred, y_batch)
                rep_t1, rep_t0 = rep_batch[(t_batch > 0).nonzero()[:, 0]], rep_batch[(t_batch < 1).nonzero()[:, 0]]
                imb_dist = alpha * wasserstein(rep_t0, rep_t1, p_treated) if alpha > 0 else 0
                cfr_loss = risk + imb_dist

                tot_loss = cfr_loss

                optimizerv2.zero_grad()
                tot_loss.backward()
                optimizerv2.step()
                schedulerv2.step()
        else:
            ''' Train beta and F'''
            for _ in range(5):
                rep_batch, y0_pred, y1_pred, y_pred, prop_pred, f_x = model(x_batch, t_batch)
                if args.check_dr:
                    prop_pred = args.misspec_prop
                else:
                    prop_pred = torch.clamp(prop_pred, min=0.01, max=0.99)
                if ismono:
                    if not isunmeasured:
                        # Value at Risk VAR loss
                        v = 0.5 * f_x * (theta - (y1_pred - y0_pred)) + 0.5 * (theta + (1 - 2 * theta) * (y1_pred - y0_pred))
                        uncertainty = v + a3 * 0.25 * (1- 2 * theta - f_x) * (t_batch - prop_pred) / prop_pred / (1 - prop_pred) * 2 * (y_batch - y_pred)

                        aug_loss = - F.sigmoid(model.beta - v) * (model.beta - uncertainty) / q_alpha + model.beta
                        aug_loss = torch.mean(sample_weight * aug_loss)
                    else:
                        # Upper bound on conditional risk
                        lb = 2 * (y1_pred - y0_pred) - 1
                        ub = 2 * (y1_pred - y0_pred) + 1
                        v = (1 - f_x) * 0.5 * 0.5 * 0.5 * ub + (1 + f_x) * 0.5 * 0.5 * (1 - 0.5 * lb)
                        uncertainty = v
                        aug_loss = - F.sigmoid(model.beta - v) * (model.beta - uncertainty) / q_alpha + model.beta
                        aug_loss = torch.mean(sample_weight * aug_loss)
                else:
                    v = 0.5 * (1 - f_x) * (1 - theta) * F.relu(y1_pred - y0_pred) + 0.5 * (1 + f_x) * theta * (1 - 0.5 - 0.5 * (y1_pred - y0_pred))
                    dr = (t_batch - prop_pred) / prop_pred / (1 - prop_pred) * 2 * (y_batch - y_pred)
                    uncertainty = v + a3 * (0.5 * (1 - f_x) * (1-theta) * 0.5 * F.sigmoid(y1_pred - y0_pred) - 0.5 * (1 + f_x) * theta * 0.25) * dr
                    aug_loss = - F.sigmoid(model.beta - v) * (model.beta - uncertainty) / q_alpha + model.beta
                    aug_loss = torch.mean(sample_weight * aug_loss)

                beta_loss = - aug_loss
                optimizervb.zero_grad()
                beta_loss.backward(retain_graph=True)
                optimizervb.step()
                schedulervb.step()

                if args.onlyF:
                    optimizervf.zero_grad()
                    aug_loss.backward()
                    optimizervf.step()
                    schedulervf.step()
                else:
                    optimizer.zero_grad()
                    aug_loss.backward()
                    optimizer.step()
                    scheduler.step()




        if epoch % args.output_delay == 0 or epoch==args.iterations-1:
            model.eval()

            _, _, _, valid_y_pred, _, _ = model(valid.X, valid.T)
            eval_loss = loss_valid(valid_y_pred, valid.Y)

            train_df = evaluation(epoch, eval_loss, tot_loss, q_alpha, model, train, ifprint=True)
            train_dfs = pd.concat([train_dfs, train_df])
            train_dfs = train_dfs.reset_index(drop=True)

            test_df = evaluation(epoch, eval_loss, tot_loss, q_alpha, model, test, ifprint=False)
            test_dfs = pd.concat([test_dfs, test_df])
            test_dfs = test_dfs.reset_index(drop=True)

    return train_dfs.round(4), test_dfs.round(4)

def evaluation(epoch, loss, totloss, q_alpha, model, data, ifprint=True):
    rep_batch, y0_pred, y1_pred, y_pred, prop_pred, f_x = model(data.X, data.T)
    v = 0.25 * f_x * (1 - 2 * (y1_pred - y0_pred)) + 0.25
    R, C, PR = data.R, data.C, data.P

    # responde -> non-responde: 0*(R-C)=0   1*(R-C)=(R-C)   R=1, hatR=0, 1-C ---- regret
    # non-responde -> responde: 0*(R-C)=0   1*(R-C)=(R-C)   R=0, hatR=1, C   ---- regret
    q_est = v.flatten().quantile(q_alpha)
    select = (v < q_est).float()
    abstain = (v >= q_est).float()

    rejection_gap = torch.abs(torch.mean(abstain) - (1 - q_alpha))

    R_pred = y1_pred > y0_pred # We would apply treatments to these samples.
    tau_accuracy = torch.mean((R_pred == R).float())
    tau_PN_samples = (R_pred > R).float()
    tau_NP_samples = (R_pred < R).float()
    tau_PN = torch.mean(tau_PN_samples)
    tau_NP = torch.mean(tau_NP_samples)
    tau_REGRET  =  torch.mean(tau_PN_samples * (C)) + torch.mean(tau_NP_samples * (1-C))

    tau2_pred = y1_pred - y0_pred # We would apply treatments to these samples.
    R2_pred   = tau2_pred > C
    tau2_accuracy = torch.mean((R2_pred == R).float())
    tau2_PN_samples = (R2_pred > R).float()
    tau2_NP_samples = (R2_pred < R).float()
    tau2_PN = torch.mean(tau2_PN_samples)
    tau2_NP = torch.mean(tau2_NP_samples)
    tau2_REGRET = torch.mean(tau2_PN_samples * (C)) + torch.mean(tau2_NP_samples * (1-C))

    PEHE = torch.sqrt(torch.mean((tau2_pred - PR) ** 2))

    f_pred = f_x > 0. # We would apply treatments to these samples.
    f_accuracy = torch.mean((f_pred == R).float())
    f_PN_samples = (f_pred > R).float()
    f_NP_samples = (f_pred < R).float()
    f_PN = torch.sum(f_PN_samples * select) / torch.sum(select)
    f_NP = torch.sum(f_NP_samples * select) / torch.sum(select)
    f_REGRET = torch.mean(f_PN * (C)) + torch.mean(f_NP * (1 - C))
    f_PN_abs = torch.sum(f_PN_samples * abstain) / torch.sum(abstain)
    f_NP_abs = torch.sum(f_NP_samples * abstain) / torch.sum(abstain)
    f_REGRET_abs = torch.mean(f_PN_abs * (C)) + torch.mean(f_NP_abs * (1 - C))
    f_PN_overall = torch.mean(f_PN_samples)
    f_NP_overall = torch.mean(f_NP_samples)
    f_REGRET_overall = torch.mean(f_PN_overall * (C)) + torch.mean(f_NP_overall * (1 - C))



    if ifprint: print('{}, loss:{:.4f}, totloss:{:.4f}, FPR: {:.4f}, FNR: {:.4f}, f_REGRET: {:.4f}, f_REGRET_abs: {:.4f}, f_REGRET_overall: {:.4f}'.format(epoch, loss, totloss, f_PN, f_NP, f_REGRET, f_REGRET_abs, f_REGRET_overall))


    re_dict = {
        'epoch': [epoch],
        'loss':[loss.item()],
        'totloss':[totloss.item()],
        'FPR': [f_PN.item()],
        'FNR':[f_NP.item()],
        'f_REGRET': [f_REGRET.item()],
        'f_REGRET_abs': [f_REGRET_abs.item()],
        'f_REGRET_overall': [f_REGRET_overall.item()],
        'Rejection_gap': [rejection_gap.item()],
    }

    return pd.DataFrame(re_dict)
