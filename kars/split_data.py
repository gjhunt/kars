import pandas as pd
import numpy as np

def split_data(pms,ls,pct_train=.1,n_train=None,random=True):
        """
        Split parameters and lineshapes into training/testing datasets. 

        pms - matrix of training parameters (rows are different parameter sets and columns are flow parameters)
        ls - matrix of training lineshapes (rows are different lineshapes and columns correspond to wavenumber)
        pct_train - percentage to put in training set. One of pct_train and n_train must be specified.
        n_train - number to put in training set. One of pct_train and n_train must be specified.
        random - boolean with default True randomizing the split.
        """
        idx = np.arange(0,pms.shape[0])
        if random:
                np.random.shuffle(idx)
        if n_train is None:
                train_ss = idx[np.linspace(0,pms.shape[0]-1,num=int(pct_train*pms.shape[0]),dtype='int')]
        else:
                train_ss = idx[0:n_train]

        val_ss = np.setdiff1d(idx,train_ss)
        tpms = pms.iloc[train_ss].reset_index(drop=True)
        vpms = pms.iloc[val_ss].reset_index(drop=True)
        tls = ls.iloc[train_ss].reset_index(drop=True)
        vls = ls.iloc[val_ss].reset_index(drop=True)

        return tpms,vpms,tls,vls,train_ss,val_ss
