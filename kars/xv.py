import pandas as pd
import numpy as np

from .split_data import split_data
from .karsft import karsft

def err_gamma(gma,tpms,tls,pct=.95):
    tn_params, v_params, tn_ls, v_ls, _, _ = split_data(tpms,tls,pct_train=pct)
    kf = karsft(train_params = tn_params,
           train_ls = tn_ls,
           gamma=gma)
    v_pred = kf.predict(param=v_params)
    dff = np.abs(v_ls-v_pred)**2
    v_err = np.mean(dff,axis=0)
    return np.mean(v_err)

def err_gamma_meta(gma,tpms,tls,N_rep=10,pct=.75):
    return [err_gamma(gma=gma,tpms=tpms,tls=tls,pct=pct) for i in range(0,N_rep)]

def xv_gamma(gma_seq,tpms,tls,N_rep=5):
    """
    Use cross-validation to estimate error of karsft using various values of gamma.

    gma_seq - sequence of gamma values to try.
    tpms - matrix of training parameters (rows are different parameter sets and columns are flow parameters)
    tls - matrix of training lineshapes (rows are different lineshapes and columns correspond to wavenumber)
    N_rep - integer indicating how many randomize cross-validation folds to perform for each value of gamma. 
    """
    errs = []
    for i in range(len(gma_seq)):
        g = gma_seq[i]
        errs.append(np.median(err_gamma_meta(gma=g,tpms=tpms,tls=tls,N_rep=N_rep)))
    
    return(errs)
