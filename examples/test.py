# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Example using KARS (Kernel Approximation of Raman Spectra)

# First, we read in some example library data

import kars

# +
import pandas as pd
import numpy as np

library_params = pd.read_csv('params.csv',header=None) #gas parameters
library_params.columns = ['Temp','N2','H2','O2','H2O']

library = pd.read_csv('spectra.csv',header=None).T #library spectra
library = np.sqrt(library)

library_wns = pd.read_csv('wn.csv',header=None).T #wavenumber (not strictly nec.)
w = np.array(library_wns.iloc[0])
# -

# we'll split off some of the example data into a validation set and keep some for training

n_train = 750

np.random.seed(65441326)
train_params, val_params, train_ls, val_ls, _, _ = kars.split_data(library_params,library,
                                                        n_train = n_train)

# make the kars fit

kf = kars.karsft(train_params = train_params,
           train_ls = train_ls,
           gamma=1)

# we can then look at some predictions of the validation data. First we look at predicting a single spectra,

from matplotlib import pyplot as plt

truth = val_ls.iloc[0]
pm = np.array([val_params.iloc[0]])

pred = kf.predict(pm).reshape(-1)
plt.plot(w,truth,label='truth');
plt.plot(w,pred,label='prediction')
plt.legend();
plt.xlabel("wavenumber");
plt.ylabel("sqrt(spectra)");

# but we can also predict multiple spectra at a time,

truth = val_ls.iloc[0:3].T
pm = val_params.iloc[0:3]

# +
clr_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
pred = kf.predict(pm).T

for i in range(pred.shape[1]):
    plt.plot(w,truth.iloc[:,i],linestyle='solid',label='truth '+str(i+1),color=clr_cycle[i]);
    plt.plot(w,pred[:,i],linestyle='dashed',label='prediction '+str(i+1),color=clr_cycle[i]);

plt.xlabel("wavenumber");
plt.ylabel("sqrt(spectra)");
plt.legend();
# -

# given our kars fit we can see which validation spectra it predicts worst

dff = (kf.predict(val_params) - val_ls)
ii = np.argmax(np.max(dff,axis=1))
truth = val_ls.iloc[ii]
pm = np.array([val_params.iloc[ii]])
pred = kf.predict(pm).reshape(-1)
plt.plot(w,truth,label='truth');
plt.plot(w,pred,label='prediction')
plt.title("Worst example: val spectra #"+str(ii))
plt.xlabel("wavenumber");
plt.ylabel("sqrt(spectra)");
plt.legend();

# We can improve the fit by choosing $\gamma$ with cross validation. First we set some range of $\gamma$ values to try out

gma_seq = [10**float(x) for x in np.arange(-5,5,2)]

# and then we calculate the cross-validated error using only the training data

xv_err = kars.xv_gamma(gma_seq,tpms=train_params,tls=train_ls)

plt.scatter(np.log10(gma_seq),np.log10(xv_err));
plt.plot(np.log10(gma_seq),np.log10(xv_err));
plt.xlabel("log10(gamma)");
plt.ylabel("log10(xv-error)");

# we then choose $\gamma^*$ as the value which minimizes this cross-validated error

gma_star = gma_seq[np.argmin(xv_err)]
gma_star

# and re-fit our kars fit using this value

kf = kars.karsft(train_params = train_params,
           train_ls = train_ls,
           gamma=gma_star)

# we can now look at this worst example again and see improvement 

dff = (kf.predict(val_params) - val_ls)
ii = np.argmax(np.max(dff,axis=1))
truth = val_ls.iloc[ii]
pm = np.array([val_params.iloc[ii]])
pred = kf.predict(pm).reshape(-1)
plt.plot(w,truth,label='truth');
plt.plot(w,pred,label='prediction')
plt.title("Worst example: val spectra #"+str(ii))
plt.xlabel("wavenumber");
plt.ylabel("sqrt(spectra)");
plt.legend();
