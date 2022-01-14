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

## Data Gen
def LM_transformer(df,ID_col, T_col,E_col,window,S,measure_T_col) :
    super_set = pd.DataFrame()
    
    for t in S :
        # LM point 이후 생존자
        # R_t_idx = np.where(df[T_col] > t )
        R_t_idx = np.where( (df[T_col] > t ) & (df[measure_T_col] <= t ) )
        R_t = df.loc[R_t_idx].reset_index(drop=True)
        
        # LM point - 변수로 지정. strata로 나중에 지정하려고
        R_t['LM'] = t
        
        # time & event 수정 필요한 그룹. -> t+w 시점에서 censoring된 것으로 처리
        occurance_out_index = np.where(R_t[T_col] > t+window)
        for idx in occurance_out_index :
            R_t.loc[idx,T_col] = t+window
            R_t.loc[idx,E_col] = 0
            
        super_set = pd.concat([super_set,R_t],axis=0)
        
        # Leave only last measurements per each id & lm points
        super_set = super_set.drop_duplicates([ID_col,'LM'],keep='last')
        
        # Time elapsed from measurement & LM time
        super_set['diff'] = super_set['LM'] - super_set[measure_T_col]
                
    return  super_set.drop(columns = [measure_T_col], axis=1).reset_index(drop=True)

## Data Gen2
def LM_transformer2(df,ID_col, T_col,E_col,window,S,measure_T_col, k_bin, train=True) :
    super_set = df
    
    discretized_set = pd.DataFrame()

    for s in S :
        temp = super_set[super_set['LM'] == s].reset_index(drop=True)
        temp_bin = np.linspace(s, s+window, k_bin)

        temp_digitize = np.digitize(temp[T_col],temp_bin, right =True)
        temp['bin'] = temp_digitize    

        
        for i in range(temp.shape[0]) :
            temp2 = temp.copy().iloc[i,:]
            if train :
                for j in range(1,temp_digitize[i]) :
                    temp2['bin'] = j
                    temp2[E_col] = 0
                    discretized_set = pd.concat([discretized_set,temp2],axis=1)
                    
                temp2['bin'] = temp_digitize[i]
                temp2[E_col] = temp.loc[i,E_col]
                discretized_set = pd.concat([discretized_set,temp2],axis=1)
                
            else :
                for j in range(1,k_bin) :
                    temp2['bin'] = j
                    temp2[E_col] = 0
                    discretized_set = pd.concat([discretized_set,temp2],axis=1)
                
        
    discretized_set = discretized_set.T
    
    return discretized_set.drop(columns = [T_col], axis=1).reset_index(drop=True)

def splitID(data,ID_col,p) :
    # Unique ID names
    unique_ids = np.unique(data[ID_col])

    # Number of samples within each train and test set
    n_train = round(len(unique_ids)*0.7)
    n_test = len(unique_ids) - n_train
    
    # IDs within train set and test set
    train_ids = list(sample(set(unique_ids), n_train))
    test_ids = list(set(unique_ids).difference(set(train_ids)))

    # Row-wise masking for train and test set
    mask_train = data[ID_col].isin(train_ids)
    mask_test = data[ID_col].isin(test_ids)

    # final train and test sets
    data_train = data[mask_train].reset_index(drop=True)
    data_test = data[mask_test].reset_index(drop=True)
    
    return data_train, data_test

def boot_weight(df, ID_col, boot=True) : 
    unique_ids = np.unique(df[ID_col])
    
    train_boot = np.random.choice(a = unique_ids, replace = boot, size =  len(unique_ids))
    boot_counts = pd.DataFrame.from_dict(dict(Counter(train_boot)),orient='index').reset_index()
    boot_counts.columns = [ID_col, 'weight_boot']
    
    return pd.merge(left=pd.DataFrame({ID_col : unique_ids}), right=boot_counts, how='left', on=ID_col).fillna(0)
    
class kfold :
    def __init__(self, k, ID_col, df1, df2, df3_train, df3_validation) :
        self.k = k
        self.ID_col = ID_col
        self.df1 = df1
        self.df2 = df2
        self.df3_train = df3_train
        self.df3_validation = df3_validation
        
        self.kf = KFold(n_splits=k, shuffle=True)
        
        # unique ids in b_th bootstrapped sample        
        self.unique_ids = np.unique(df1[ID_col])
        
        # 
        self.k_fold = 0 
        
        # bootstrap part
        # TO BE ... 
        
        # where ids in each kth train set and validation set is stored 
        fold_train_id = []
        fold_validation_id = []

        for train_unique_id_idx, validation_unique_id_idx in self.kf.split(self.unique_ids) :
            fold_train_id.append(self.unique_ids[train_unique_id_idx])
            fold_validation_id.append(self.unique_ids[validation_unique_id_idx])
        
        self.fold_train_id = fold_train_id
        self.fold_validation_id = fold_validation_id
        
        
    def __iter__(self) : 
        return self
    
    def __next__(self) : 
        if self.k_fold > (self.k) :
            raise StopIteration
            
        else :
            # df1 - original dataset
            mask1_train = self.df1[self.ID_col].isin(self.fold_train_id[self.k_fold])
            mask1_validation = self.df1[self.ID_col].isin(self.fold_validation_id[self.k_fold])

            df1_k_train = self.df1[mask1_train]
            df1_k_validation = self.df1[mask1_validation]

            # df2 - output of LM_transformer1 
            mask2_k_train = self.df2[self.ID_col].isin(self.fold_train_id[self.k_fold])
            mask2_k_validation = self.df2[self.ID_col].isin(self.fold_validation_id[self.k_fold])

            df2_k_train = self.df2[mask2_k_train]
            df2_k_validation = self.df2[mask2_k_validation]

            # df3 - output of LM_transformer2
            mask3_k_train = self.df3_train[self.ID_col].isin(self.fold_train_id[self.k_fold])
            mask3_k_validation = self.df3_validation[self.ID_col].isin(self.fold_validation_id[self.k_fold])

            df3_k_train = self.df3_train[mask3_k_train]
            df3_k_validation = self.df3_validation[mask3_k_validation]
            
            self.k_fold += 1
            print('$$$')
            print('Iteration : ',self.k_fold)
            return df1_k_train, df1_k_validation, df2_k_train, df2_k_validation, df3_k_train, df3_k_validation
        
def Add_IPCW(train_df, evaluation_df, ID_col,E_col,T_col, S, window) :
    
    # fitting KM for censoring from each landmark point
    temp_cens = pd.DataFrame()
    for s in S :
        # leave only risk set at given landmark point s
        risk_at_s = train_df.loc[train_df[T_col]>s]
        # fitting KM for censoring 
        km_cens = KaplanMeierFitter()
        km_cens.fit(risk_at_s[T_col], event_observed = abs(risk_at_s[E_col]-1))

        # predict KM for transform_df
        censor_pred = km_cens.predict(times= evaluation_df[T_col])
        
        # size of risk set. 
        n_risk_at_s = risk_at_s.shape[0]
        # 1/temp_cens is IPCW for uncensored(administrative censoring 제외)
        temp_cens[s] = censor_pred * n_risk_at_s
        
    temp_cens = temp_cens.reset_index(drop=True)
    
    
    # obtain IPC WEIGHTS for each row
    w_list = []
    for i in range(evaluation_df.shape[0]) :
        s_i = evaluation_df.loc[i,'LM']        
        if (evaluation_df.loc[i,E_col] == 0) & (evaluation_df.loc[i,T_col] != s_i+window) :
            w_i =0
        else :
            w_i = 1/temp_cens.loc[i,s_i]
    
        w_list.append(w_i)

    return np.array(w_list)


def v_year_survival_prob_cox(model, ID_col, test_set, S ,window) :
    # predict survival probability in each time grid (given LM points)
    predicted_survival = model.predict_survival_function(test_set.drop(ID_col, axis=1), times= S + window)
    
    # discretized survival probability from each LM points to LM points + window(v)
    time = test_set.LM + window

    v_year_surv_prob = []
    for idx in time.index : 
        value = predicted_survival.loc[time[idx],idx]
        v_year_surv_prob.append(value)
    return np.array(v_year_surv_prob)


def v_year_survival_prob_ml(model, ID_col, E_col, test_set) :
    surv_prob = pd.DataFrame(model.predict_proba(test_set.drop([ID_col, E_col, 'weight'], axis=1))[:,0])

    output = pd.concat([test_set[[ID_col, 'LM', 'bin']].reset_index(drop = True), surv_prob],axis=1)
    output = output.pivot_table(index=['LM',ID_col], columns='bin', values=0)

    output = output.reset_index(drop=True)
    # cumprod from column 3(surv_1) ~ surv_last
    output = np.cumprod(output,axis= 1)
    
    return output.iloc[:,-1]


def level_1_stack(model_specifics_1,ID_col, E_col, T_col, measure_T_col, window, S, k_bin, 
                  train_sets, validation_sets) :

    true_survival_status = np.array(1 - np.array(validation_sets[1][E_col]))
    
    out = true_survival_status
    model_specifics = model_specifics_1.reset_index(drop = True)
    
    for g_1 in range(model_specifics.shape[0]) : 
        model_name = model_specifics.loc[g_1,'model_name'] 
        model_instance = model_specifics.loc[g_1,'model_instance'] 
        model_hyperparams = model_specifics.loc[g_1,'hyperparams']
        model_type = model_specifics.loc[g_1,'type']
        
        print(model_name)

        param_combinations = list(itertools.product(*list(model_hyperparams.values())))
        param_names = list(model_hyperparams.keys())

        if model_type == 'cont' :
            # feed appropriate form of train validation data
            train_data = train_sets[1]
            validation_data = validation_sets[1]
            
            # change hyperparameters according to model_hyperparameter grid
            for g_2 in range(len(param_combinations)) :
                for param_idx in range(len(param_names)) :
                    setattr(model_instance, param_names[param_idx], param_combinations[g_2][param_idx])

                model_instance.fit(df = train_data.drop([ID_col,'weight'],axis=1), duration_col = T_col, event_col = E_col, step_size = 0.01, robust=True)
                # print(model_instance.print_summary())

                surv_prob_est = v_year_survival_prob_cox(model = model_instance,ID_col= ID_col ,test_set = validation_data, S=S ,window = window)
                out = np.c_[out, surv_prob_est]

        elif model_type == 'disc' : 
            # feed appropriate form of train validation data
            train_data = train_sets[2]
            validation_data = validation_sets[2]
            
            # change hyperparameters according to model_hyperparameter grid
            for g_2 in range(len(param_combinations)) :
                for param_idx in range(len(param_names)) :
                    setattr(model_instance, param_names[param_idx], param_combinations[g_2][param_idx])

                model_instance.fit(train_data.drop([ID_col, E_col,'weight'],axis=1),train_data[E_col])
                # print(model_instance.print_summary())

                surv_prob_est = v_year_survival_prob_ml(model = model_instance, ID_col = ID_col, E_col = E_col, test_set = validation_data)
                out = np.c_[out, surv_prob_est]
        
    # out : first column in true value, 
    #       2nd column to end is predicted survival prob from each models with different hyperparam settings
    return out
