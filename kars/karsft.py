import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

class karsft:
        
        def __init__(self, train_params, train_ls, gamma, scale=True, fit=True):
                """
                Initialize karsft object.
        
                train_params - matrix of training parameters (rows are different parameter sets and columns are flow parameters)
                train_ls - matrix of training lineshapes (rows are different lineshapes and columns correspond to wavenumber)
                gamma - smoothness parameter for kernel fit
                scale - boolean with default True indicating that parameter should be z-scored before fit
                fit - boolean with default True indicating that approximator should be fit upon initialization
                """

                self.train_params = train_params
                self.train_ls = train_ls
                self.gamma = gamma
                self.scale = scale

                if self.scale:
                        self.scaler = StandardScaler()
                        ft = self.scaler.fit_transform(self.train_params)
                        cns = train_params.columns
                        self.train_params = pd.DataFrame(ft)
                        self.train_params.columns = cns

                if fit:
                        self.fit()

        def fit(self):
                """
                Fit karsft object. 
                """
                K = np.exp(-self.gamma*pairwise_distances(self.train_params,self.train_params))
                self.W = np.linalg.solve(K, self.train_ls)

        def predict(self,param):
                """
                Predict using karsft. 

                param - flow parameter at which to predict. If scale=True upon initialization predict will automatically re-scale param passed in.
                """
                if self.scale:
                        param = self.scaler.transform(param)

                b = np.exp(-self.gamma*pairwise_distances(self.train_params,param))
                return b.T.dot(self.W)
