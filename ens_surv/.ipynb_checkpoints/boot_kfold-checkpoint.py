import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from lifelines import KaplanMeierFitter

# models 
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier

# others
from numpy import inf
from random import sample
from collections import Counter
from sklearn.model_selection import KFold
import itertools
import copy
from sklearn.base import clone

from ens_surv.utils import *

import warnings
warnings.filterwarnings("ignore")

# Return bootstrapped Superset, train set(in-bag), oob sample datasets.
# Return bootstrapped Superset, train set(in-bag), oob sample datasets.
class boot_kfold :
    def __init__(self, base_info, train_df_list, test_df_list,model_specifics_1, model_specifics_2) :         
        # base_info : dict with ID_col, T_col, E_col, measure_T_col names, boot(bool), B, K
        self.base_info = base_info
        self.ID_col = base_info['ID_col']
        self.T_col = base_info['T_col']
        self.E_col = base_info['E_col']
        self.measure_T_col = base_info['measure_T_col']
        self.window = base_info['window']
        self.S = base_info['S']
        self.k_bin = base_info['k_bin']
        
        self.boot = base_info['boot']
        self.B = base_info['B']
        self.K = base_info['K']
        
        # sorting dataframes in right order
        temp = [train_df_list[0]]
        for df in train_df_list[1:] :
            temp.append(df.sort_values(['LM',self.ID_col]))
        train_df_list = temp
        
        temp = [test_df_list[0]]
        for df in test_df_list[1:] :
            temp.append(df.sort_values(['LM',self.ID_col]))
        test_df_list = temp
        
        del(temp)

        # list of dataframes :
        ## in train, sequently, original data / lm1 transformed / lm2 transformed(trn form) / lm2 transformed(validation form)
        ## in test, sequently, original data / lm1 transformed/ lm2 transformed(validation form)
        self.train_df_list = train_df_list
        self.test_df_list = test_df_list
        
        # model_specifics(dataframe)
        ## model_specifics_1 : 1st stage models' 1) model name / model_instance / hyperparams grid / type
        ## model_specifics_2 : 2nd stage models' 1) model name / model_instance / hyperparams grid / type
        self.model_specifics_1 = model_specifics_1
        self.model_specifics_2 = model_specifics_2
    
    # boot_stack outputs B stacked super set
    def boot_stack(self,train_df_list = None, test_df_list = None, model_specifics_1 = None, model_specifics_2 = None, 
                   ID_col = None,T_col=None,E_col=None,measure_T_col= None,
                   window = None, S = None, k_bin = None,
                   boot = None, B = None, K= None) : 
        # initiallizing
        if train_df_list is None :
            train_df_list = self.train_df_list
        if test_df_list is None :
            test_df_list = self.test_df_list
            
        if model_specifics_1 is None :
            model_specifics_1 = self.model_specifics_1
        if model_specifics_2 is None :
            model_specifics_2 = self.model_specifics_2

        if ID_col is None :
            ID_col = self.ID_col
        if E_col is None :
            E_col = self.E_col
        if T_col is None :
            T_col = self.T_col
        if measure_T_col is None :
            measure_T_col = self.measure_T_col
        if window is None :
            window = self.window
        if S is None :
            S = self.S
        if k_bin is None :
            k_bin = self.k_bin
        
        if boot is None :
            boot = self.boot
        if B is None :
            B = self.B
        if K is None :
            K = self.K
        
        # censoring model
        KM_cens = KaplanMeierFitter()
                    
 ######################################################################################################################################################################
        # OUTER-LOOP
        BOOTSTRAP_SUPERSETS = []
        IN_BAG_SETS = []
        OUT_BAG_SETS = []
        WEIGHT_BAG_SETS = []
        
        for b in range(B) :
            print('######################################################################')
            print(b+1,'/', B,' Resampled')
            
            # add bootstrap weight and IPC weight column to the inbag sets.
            train_df_list_new, train_df_list_oob = add_weight_column(train_df_list=train_df_list, ID_col = ID_col, T_col=T_col, E_col = E_col, boot=boot, S=S, window= window)
            
            # kfold part - Different IDs are divided into K folds
            kf = kfold(k=K, ID_col=ID_col, df1 = train_df_list_new[0], df2 = train_df_list_new[1], df3_train = train_df_list_new[2], df3_validation = train_df_list_new[3])
            ############################################################################################################
            # INNER-LOOP
            ## b_TH_STACK : 1st column contains true survival status / 2 to end columns contain survival estimates(of training set) from different models. 
            b_TH_STACK = np.array([])
            b_TH_weight = []
            for k in range(K) :
                print(k+1,'/', K,' fold')
                # 
                df1_k_train, df1_k_validation, df2_k_train, df2_k_validation, df3_k_train, df3_k_validation = next(kf)

                # Training 1st stage models
                ## 1) Training 1st stage models with kth training set
                ## 2) Predict kth validation set with trained 1st stage models
                ## Stacking results from 2), forming inputs for 2nd stage models

                out_b_k = level_1_stack(model_specifics_1,ID_col=ID_col, E_col=E_col, T_col = T_col, measure_T_col = measure_T_col, window = window, S = S, k_bin = k_bin, 
                                        train_sets=[df1_k_train, df2_k_train, df3_k_train], 
                                        validation_sets=[df1_k_validation, df2_k_validation, df3_k_validation])
                weight_b_k = df2_k_validation['weight']
                
                b_TH_STACK = b_TH_STACK.reshape(-1, out_b_k.shape[1])
                b_TH_STACK = np.vstack((b_TH_STACK, out_b_k))
                
                b_TH_weight = np.append(b_TH_weight, np.array(weight_b_k).ravel())
            ############################################################################################################    
            
            # append results from bth bootstrapping, b = 1, ... , B
            ## BOOTSTRAP_SUPERSETS : All B (b_TH_STACK) super sets obtained from B bootstrap samples.
            BOOTSTRAP_SUPERSETS.append(b_TH_STACK)
            ## in_bag_train : to fully train 1st stage models
            IN_BAG_SETS.append(train_df_list_new)
        
            ## out_bag_train : to check validity
            OUT_BAG_SETS.append(train_df_list_oob)
            
            # Weights to use for 2nd stage model
            b_TH_weight = b_TH_weight
            WEIGHT_BAG_SETS.append(b_TH_weight)
            
            # Fit 2nd stage model & Store( bootstrap True / False 에 따라 다르겠지?)
            ## To be continue...
            
            # Refit 1st stage model & Store
            ## To be continue...
            
            # 
            
 ###################################################################################################################################################################### 
        
        # store 
        self.supersets = BOOTSTRAP_SUPERSETS
        self.inbags = IN_BAG_SETS
        self.outbags = OUT_BAG_SETS
        self.weights = WEIGHT_BAG_SETS 
        
        # df1_k_train, df1_k_validation, df2_k_train, df2_k_validation, df3_k_train, df3_k_validation
        return BOOTSTRAP_SUPERSETS, IN_BAG_SETS, OUT_BAG_SETS, WEIGHT_BAG_SETS

