import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

# models 
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.base import clone

# others
from numpy import inf
from random import sample
from collections import Counter
from sklearn.model_selection import KFold
import itertools
import copy
from sklearn.base import clone
from sklearn.metrics import mean_squared_error,brier_score_loss
from scipy.optimize import minimize


# squared error 
def sq_error(a,b) : 
    return( np.mean( (a-b)**2 ))

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

## LM_transformer2(discretizer) - outputs Discretized landmarking dataset
## input should be output from basic lm_transformer
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

# Train-test split by ID, p is proportion of train set
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

# boot_weight : outputs boostrapped sample from df
# 'weight_boot' indicates how many times certain ID is selected in boostrapped sample
def boot_weight(df, ID_col, boot=True) : 
    unique_ids = np.unique(df[ID_col])
    
    train_boot = np.random.choice(a = unique_ids, replace = boot, size =  len(unique_ids))
    boot_counts = pd.DataFrame.from_dict(dict(Counter(train_boot)),orient='index').reset_index()
    boot_counts.columns = [ID_col, 'weight_boot']
    
    return pd.merge(left=pd.DataFrame({ID_col : unique_ids}), right=boot_counts, how='left', on=ID_col).fillna(0)

# kfold generator/iterator given ID 
# outputs kfold train and test sets.
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
        
def add_weight_column(train_df_list, ID_col, T_col, E_col, boot, S, window) :
    
    # add bootstrap weight & seperate inbag/outbag samples
    boot_weight_at_b = boot_weight(df = train_df_list[0], ID_col = ID_col, boot=boot)

    train_df_list_inbag = [];train_df_list_outbag = []
    for df_temp in train_df_list :
        df_inbag = pd.merge(pd.DataFrame.copy(df_temp), right = boot_weight_at_b, how='left', on= ID_col); df_inbag = df_inbag[df_inbag.weight_boot !=0]
        df_outbag = pd.merge(pd.DataFrame.copy(df_temp), right = boot_weight_at_b, how='left', on= ID_col); df_outbag = df_outbag[df_outbag.weight_boot ==0]

        train_df_list_inbag.append(df_inbag)
        train_df_list_outbag.append(df_outbag)

    # add IPC weight part
    KM_cens = KaplanMeierFitter()
    df_temp = train_df_list_inbag[0].drop_duplicates([ID_col])

    cens_prob = []
    for s in S :
        df_risk = df_temp[df_temp[T_col]>s]
        df_risk['LM'] = s
        n_risk = df_risk.shape[0]

        KM_cens.fit(durations = df_risk[T_col], event_observed = abs(df_risk[E_col]-1), weights  = df_risk['weight_boot'])

        cens_prob.append(np.array(KM_cens.predict(train_df_list_inbag[1].loc[train_df_list_inbag[1].LM == s].sort_values(ID_col)[T_col]) + 10**(-10))*n_risk)

    cens_prob = [item for sublist in cens_prob for item in sublist]
    IPC_weight_at_b = train_df_list_inbag[1][[ID_col, 'LM',T_col,E_col]].sort_values(['LM',ID_col])
    IPC_weight_at_b['cens_prob'] = cens_prob; IPC_weight_at_b['weight_IPC'] = 1/IPC_weight_at_b['cens_prob']
    IPC_weight_at_b.loc[((IPC_weight_at_b[T_col] < IPC_weight_at_b['LM']+window)&(IPC_weight_at_b[E_col]==0)),'weight_IPC'] = 0 

    IPC_weight_at_b = IPC_weight_at_b[[ID_col, 'LM', 'weight_IPC']]
    
    for i in range(1,len(train_df_list_inbag)) :
        train_df_list_inbag[i] = train_df_list_inbag[i].merge(IPC_weight_at_b,how='left', on = [ID_col, 'LM'])
        train_df_list_inbag[i]['weight'] = train_df_list_inbag[i]['weight_boot']*train_df_list_inbag[i]['weight_IPC']*(10**4) + 10**(-10)
        train_df_list_inbag[i] = train_df_list_inbag[i].drop(['weight_boot','weight_IPC'],axis=1)
    
    return(train_df_list_inbag, train_df_list_outbag)


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
    del_col = [col for col in test_set.columns if "weight" in col]; del_col.append(ID_col) ; del_col.append(E_col)
    surv_prob = pd.DataFrame(model.predict_proba(test_set.drop(del_col, axis=1))[:,0])

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

        if model_type == 'cont' : # Cox model
            # feed appropriate form of train validation data
            train_data = train_sets[1]
            validation_data = validation_sets[1]
            
            # change hyperparameters according to model_hyperparameter grid
            for g_2 in range(len(param_combinations)) :
                for param_idx in range(len(param_names)) :
                    setattr(model_instance, param_names[param_idx], param_combinations[g_2][param_idx])
                
                model_instance.fit(df = train_data.drop([ID_col],axis=1), duration_col = T_col, event_col = E_col,weights_col = 'weight' ,step_size = 0.01, robust=True)
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
                
                if model_name in ['KNN','MLP'] : 
                    model_instance.fit(train_data.drop([ID_col, E_col,'weight'],axis=1),train_data[E_col])                    
                else :     
                    model_instance.fit(train_data.drop([ID_col, E_col,'weight'],axis=1),train_data[E_col], train_data['weight'])


                surv_prob_est = v_year_survival_prob_ml(model = model_instance, ID_col = ID_col, E_col = E_col, test_set = validation_data)
                out = np.c_[out, surv_prob_est]
        
    # out : first column in true value, 
    #       2nd column to end is predicted survival prob from each models with different hyperparam settings
    return out

class nnls_constraint() : 
    def __init__(self, tol = 10**(-5), max_iter = 10^5) : 
        self.tol = tol
        self.max_iter = max_iter
        
        return
        
        
    def fit(self, x, y, w) : 
        n, k = x.shape
        obj = lambda beta, y, x, w : np.dot(w.reshape(-1,), (np.array(y).reshape(-1, ) - x @ beta)**2)/n
        
        # bound(0-1) and constrant(beta sum to 1)
        bnds = list(tuple(itertools.repeat((0,1),k)))
        cons = [{"type": "eq", "fun": lambda beta: np.sum(beta) - 1}]

        # Initial guess for betas
        init = np.repeat(0,k)
        
        # minimization
        res = minimize(obj, args=(y, x, w), x0=init, bounds=bnds, constraints=cons, tol = self.tol, options= {'maxiter':self.max_iter})
        
        self.coef_ = res.x
        self.iter = res['nit']
        self.score = res['fun']
        self.res = res
        
        return 

    def predict(self, x) : 
        return x @ self.coef_
        

class hillclimb() : 
    def __init__(self, max_iter= 2000, early_stop_n = 50, early_stop_eps = 10**(-3)) : 
        self.max_iter = max_iter
        self.early_stop_n = early_stop_n
        self.early_stop_eps = early_stop_eps
        return
        
    def fit(self, x, y, w) : 
        n, k = x.shape
        coef_ = np.zeros(k)
        
        current_score = 10^10
        
        current_iter = 0; early_stop_iter = 0 
        while (current_iter <= self.max_iter)&(early_stop_iter <= self.early_stop_n) :
            
            # search
            next_scores = []
            for i in range(k) : 
                temp_coef_ = copy.copy(coef_); temp_coef_[i] += 1
                temp_score = brier_score_loss(y, x @ (temp_coef_ / sum(temp_coef_)),w)
                next_scores.append(temp_score)
            
            
            # update
            next_score = min(next_scores)
            
            best_ind = next_scores.index(next_score)
            coef_[best_ind] = coef_[best_ind]+1
            
            current_iter += 1
            
            if (current_score - next_score) > self.early_stop_eps :
                early_stop_iter = 0 
            else : 
                early_stop_iter += 1
            
            current_score = next_score
        
        self.coef_ = coef_ / sum(coef_)
        self.iter = current_iter    
        self.score = current_score
            
    def predict(self, x) : 
        return x @ self.coef_
        
    
        