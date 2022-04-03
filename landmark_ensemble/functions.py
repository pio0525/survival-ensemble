import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# from sksurv.util import Surv
# from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

# models 
import lifelines
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# others
from numpy import inf
from random import sample, seed
from collections import Counter
from sklearn.model_selection import KFold
import itertools
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import warnings
from lifelines.utils import concordance_index
import sys 
import os
from collections import Counter

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment',  None)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import itertools
import copy
from sklearn.base import clone
from sklearn.metrics import mean_squared_error,brier_score_loss
from scipy.optimize import minimize


# Given list of IDs, split ids into train with p proportion
# return list of train id and test id
def id_train_test_split(id_list, seed_number = 1, p=0.7) :
    id_list = np.unique(id_list)
    
    n_train = round(len(id_list)*0.7)
    n_test = len(id_list) - n_train
    
    # IDs within train set and test set
    seed(seed_number)
    train_id = list(sample(set(id_list), n_train))
    test_id = list(set(id_list).difference(set(train_id)))
    return train_id, test_id
    
# inpiut : id_list to boostrapping and seed_number
# output : dataframe with id column and counts column
def id_bootstrapping_split(id_list,ID ,seed_number) :
    id_list = np.unique(id_list)
    
    np.random.seed(seed_number)
    pick = np.random.choice(a = id_list, replace = True, size =  len(id_list))
    
    boot_counts = pd.DataFrame.from_dict(dict(Counter(pick)),orient='index').reset_index()
    boot_counts.columns = [ID, 'weight']

    return pd.merge(left=pd.DataFrame({ID : id_list}), right=boot_counts, how='left', on=ID).fillna(0)

# Given list of IDs, split ids into k-fold train/validation set 
class id_kfold :
    def __init__(self,id_list, n_split,seed_number=1) : 
        self.id_list = np.unique(id_list)
        self.n_split = n_split
        self.seed_number=  seed_number

        self.kf = KFold(n_splits = n_split, shuffle =True, random_state = seed_number)
        
        self.n_iter = 0 # initializing iteration
        
        train_fold_id = [] ; validation_fold_id = []
        for train_unique_id_idx, validation_unique_id_idx in self.kf.split(self.id_list) :
                train_fold_id.append(self.id_list[train_unique_id_idx])
                validation_fold_id.append(self.id_list[validation_unique_id_idx])

        self.train_fold_id = train_fold_id
        self.validation_fold_id = validation_fold_id
        
        return
                
    def __iter__(self) : 
        return 
    
    def __next__(self) : 
        if self.n_iter > self.n_split :
            raise StopIteration
            
        else :
            self.n_iter += 1
            return self.train_fold_id[self.n_iter-1], self.validation_fold_id[self.n_iter-1]
          
# Given original form of data,
# Return landmarked dataset in continuous form
def landmarker_cont(data,ID_col, T_col,E_col,window,S,measure_T_col) :
    super_set = pd.DataFrame()
    
    for t in S :
        # LM point 이후 생존자
        # R_t_idx = np.where(data[T_col] > t )
        R_t_idx = np.where( (data[T_col] > t ) & (data[measure_T_col] <= t ) )
        R_t = data.loc[R_t_idx].reset_index(drop=True)
        
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


# Given landmarked dataset in continuous form(output from Landmarker_cont),
# Return discretized landmarked dataset.
## Note that, if arg train == True, then 
def landmarker_disc(data,ID_col, T_col,E_col,window,S,measure_T_col, k_bin, train=True) :
    super_set = data
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


# Given model_specifics(dictionary)
# Create list of model instances with hyperparameters from model_specifics(baseline)
def set_hyperparams(model_specifics) :
    model_list = []
    for g_1 in range(model_specifics.shape[0]) : 
        model_name = model_specifics.loc[g_1,'model_name'] 
        model_hyperparams = model_specifics.loc[g_1,'hyperparams']
        model_type = model_specifics.loc[g_1,'type']

        param_combinations = list(itertools.product(*list(model_hyperparams.values())))
        param_names = list(model_hyperparams.keys())

        # change hyperparameters according to model_hyperparameter grid
        for g_2 in range(len(param_combinations)) :
            model_instance = deepcopy(model_specifics.loc[g_1,'model_instance'])
            for param_idx in range(len(param_names)) :
                setattr(model_instance, param_names[param_idx], param_combinations[g_2][param_idx])
            model_list.append(model_instance)    
    return model_list
    
    
    
####################################################################################################################################
#################################                 CLASS                                              ###############################
####################################################################################################################################

# Given list of IDs, split ids into k-fold train/validation set 
class id_kfold :
    def __init__(self,id_list, n_split,seed_number=1) : 
        self.id_list = np.unique(id_list)
        self.n_split = n_split
        self.seed_number=  seed_number

        self.kf = KFold(n_splits = n_split, shuffle =True, random_state = seed_number)
        
        self.n_iter = 0 # initializing iteration
        
        train_fold_id = [] ; validation_fold_id = []
        for train_unique_id_idx, validation_unique_id_idx in self.kf.split(self.id_list) :
                train_fold_id.append(self.id_list[train_unique_id_idx])
                validation_fold_id.append(self.id_list[validation_unique_id_idx])

        self.train_fold_id = train_fold_id
        self.validation_fold_id = validation_fold_id
        
        return
                
    def __iter__(self) : 
        return 
    
    def __next__(self) : 
        if self.n_iter > self.n_split :
            raise StopIteration
            
        else :
            self.n_iter += 1
            return self.train_fold_id[self.n_iter-1], self.validation_fold_id[self.n_iter-1]
          
# fit KaplanMeier model on each landmarking time point
# return(predict) Inverse Probabliity of Censoring Weight(IPCW) * n(S) on any given dataset

## Note : fit and predict method requires continous type of landmarking dataset. 
## Note2 : censoring될 확률이 높을수록(survival estimate from KM이 작을수록) -> (관측이 되었다면) 관측치의 weight 높아짐.
class ipcw_fitter : 
    def __init__(self, S, window) : 
        self.S = S
        self.window = window
        self.censoring_model = [KaplanMeierFitter() for i in range(len(S))]
        return
    

    # T, E, W 는 해당하는 각각 time, event indcicator, weight에 해당하는 칼럼 네임.
    ## Note : 즉, bagging할 시 먼저 웨이트를 붙여서 들어와야 됨. 
    def fit(self, data, T, E, W = None) : 
        self.T = T
        self.E = E
        data['weight'] = W
        
        for i in range(len(self.S)) : 
            risk_set = data.loc[data['LM'] == self.S[i],]
            
            # Here, event is censoring, so indicator is reversed.
            time = risk_set[T]; event = abs(risk_set[E]-1); 
            if W is  None : 
                self.censoring_model[i] = self.censoring_model[i].fit(durations = np.array(time), event_observed = np.array(event))
            else :
                weight = risk_set['weight']
                self.censoring_model[i] = self.censoring_model[i].fit(durations = time, event_observed = event, weights  = weight)
        return 
    
    def predict(self, data) : 
        eps = 0.000000001
        n_S = [sum(data['LM']==s) for s in self.S]# number of risk sets on each landmark time point
        
        
        ipcw_list = []
        for i in range(data.shape[0]) : 
            lm_time = data['LM'][i]
            lm_index= np.where(self.S==lm_time)[0][0]
            
            ipcw_list.append(1/(self.censoring_model[lm_index].predict(data[self.T][i]- eps) * n_S[lm_index]))
        
        ipcw_list = np.array(ipcw_list)
        ipcw_list[(data[self.E]==0)&(data[self.T] < data['LM']+self.window)] = 0

        return ipcw_list
    
# input : model and specifics
# output : predicted v-year survival estimates
class LM_cox_fitter :
    def __init__(self, model, ID, T, E, S, window, degree= 2, stratified = False) : 
        self.model = deepcopy(model)
        self.ID = ID
        self.T = T
        self.E = E
        self.S = S
        self.window = window
        
        self.degree = degree
        self.stratified = stratified
        
    def fit(self, data, weight = None) : 
        
        temp_data = deepcopy(data)        
        x_cols = list(temp_data.columns)
        x_cols.remove(self.ID);x_cols.remove(self.T);x_cols.remove(self.E);x_cols.remove('LM');x_cols.remove('diff')
        self.x_cols = x_cols

        # making interaction term between Xs and 1, ... , d degree LM terms
        for i in range(len(x_cols)) : 
            for d in range(1,self.degree+1) : 
                col_name = x_cols[i] + '_' + str(d)
                value = temp_data[x_cols[i]] * (temp_data['LM'])**d
                temp_data[col_name] = value

        # Add weight column
        if weight is not None: 
            temp_data['weight'] = weight
            
        if self.stratified :   
            # default : landmarked time has 2nd degree relationship with baseline hazard
            temp_data['LM_2'] = (temp_data['LM'])**2
            
            if weight is None : 
                self.model.fit(df = temp_data.drop([self.ID],axis=1), duration_col = self.T, event_col = self.E, robust =True, step_size= 0.5) # no strata on LM
            else :  
                self.model.fit(df = temp_data.drop([self.ID],axis=1), duration_col = self.T, event_col = self.E, weights_col = 'weight', robust =True, step_size= 0.5) # no strata on LM
        else : 
            if weight is None : 
                self.model.fit(df = temp_data.drop([self.ID],axis=1), duration_col = self.T, event_col = self.E, strata = ['LM'], step_size= 0.5) # strata on LM
            else :
                self.model.fit(df = temp_data.drop([self.ID],axis=1), duration_col = self.T, event_col = self.E, strata = ['LM'], weights_col = 'weight', robust =True, step_size= 0.5) # strata on LM
                
        return self.model
    
    def predict(self, data, v = None) : 
        if v == None : 
            v = self.window
            
        temp_data = deepcopy(data)        

        # making interaction term between Xs and 1, ... , d degree LM terms
        for i in range(len(self.x_cols)) : 
            for d in range(1,self.degree+1) : 
                col_name = self.x_cols[i] + '_' + str(d)
                value = temp_data[self.x_cols[i]] * (temp_data['LM'])**d
                temp_data[col_name] = value
                
        if self.stratified :   
            # default : landmarked time has 2nd degree relationship with baseline hazard
            temp_data['LM_2'] = (temp_data['LM'])**2
            surv_est_mat = self.model.predict_survival_function(X = temp_data, times = self.S + v)
        else : 
            surv_est_mat = self.model.predict_survival_function(X = temp_data, times = self.S + v)
            
        v_year = temp_data.LM + v

        v_year_surv_prob = []
        for idx in v_year.index : 
            value = surv_est_mat.loc[v_year[idx],idx]
            v_year_surv_prob.append(value)
            
        return np.array(v_year_surv_prob)
        
        
# input : model and specifics
# output : predicted v-year survival estimates
class LM_sklearn_fitter : 
    def __init__(self, model, ID, E, k_bin) : 
        self.model = deepcopy(model)
        self.ID = ID
        self.E = E
        self.k_bin = k_bin
        
        
    def fit(self, data, weight = None) : 
        if weight is None : 
            self.model.fit(data.drop([self.E, self.ID], axis=1), data[self.E])
        
        else :
            self.model.fit(data.drop([self.E, self.ID], axis=1), data[self.E], weight)
                        
        return self.model
    
    def predict(self, data) : 
        data = data.drop_duplicates(subset =[self.ID, 'LM'])

        v_year_surv_prob=1
        for i in range(1,self.k_bin) : 
            data['bin'] = i
            v_year_surv_prob = v_year_surv_prob*self.model.predict_proba(data.drop([self.E, self.ID],axis=1))[:,0]

        return np.array(v_year_surv_prob)
        
class nnls_constraint : 
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
        


class hillclimb : 
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
        
# 보고 stacker2가 잘 되면 stacker2만 남겨도 됨!!!!!! 불안해서 남겨놓음;;;ㅎ
class stacker :
    def __init__(self, model_specifics, ID, T, E, S, window, k_bin) : 
        self.model_specifics = model_specifics
        self.ID = ID
        self.T = T
        self.E = E
        self.S = S
        self.window = window
        self.k_bin = k_bin 
        
        self.model_list = [] # initializing model list
        return
    
    # 
    def fit(self, data_cont, data_disc) : 
        new_model_list = []
        for i in range(self.model_specifics.shape[0]) : 
            current_model_specifics = self.model_specifics.iloc[i:(i+1),:].reset_index(drop=True)
            current_model_list = set_hyperparams(current_model_specifics) 

            current_model_name = current_model_specifics['model_name'][0]
            current_model_type = current_model_specifics['type'][0]

            # j for models in current_model_list 
            for j in range(len(current_model_list)) : 
                if current_model_type == 'cox_str' : 
                    fitter = LM_cox_fitter(model = current_model_list[j], ID = self.ID, T = self.T, E = self.E, 
                                           S = self.S, window = self.window, degree= 2, stratified = True)
                    fitter.fit(data= data_cont)

                elif current_model_type == 'cox_no_str' : 
                    fitter = LM_cox_fitter(model = current_model_list[j], ID = self.ID, T = self.T, E = self.E, 
                                           S = self.S, window = self.window, degree= 2, stratified = False)
                    fitter.fit(data= data_cont)

                else : 
                    fitter = LM_sklearn_fitter(model = current_model_list[j], ID = self.ID, E = self.E, k_bin = self.k_bin)
                    fitter.fit(data= data_disc)
                new_model_list.append(fitter)
        
        self.model_list = new_model_list
                
        return self.model_list
    
    def predict(self, data_cont, data_disc) :
        stacked = []
        for fitter in self.model_list : 
            module_tree = getattr(fitter.model,'__module__',None)
            parent = module_tree.split('.')[0] if module_tree else None
            
            if parent == lifelines.__name__:
                stacked.append(fitter.predict(data_cont))
            else :
                stacked.append(fitter.predict(data_disc))
        
        stacked = np.array(stacked).T
        
        return stacked
            
class stacker2 :
    def __init__(self, model_specifics, ID, T, E, S, window, k_bin) : 
        self.model_specifics = model_specifics
        self.ID = ID
        self.T = T
        self.E = E
        self.S = S
        self.window = window
        self.k_bin = k_bin 
        
        self.model_list = [] # initializing model list
        return
    
    # 
    def fit(self, data_cont, data_disc, weight_cont, weight_disc) : 
        new_model_list = []
        for i in range(self.model_specifics.shape[0]) : 
            current_model_specifics = self.model_specifics.iloc[i:(i+1),:].reset_index(drop=True)
            current_model_list = set_hyperparams(current_model_specifics) 

            current_model_name = current_model_specifics['model_name'][0]
            current_model_type = current_model_specifics['type'][0]

            # j for models in current_model_list 
            for j in range(len(current_model_list)) : 
                if current_model_type == 'cox_str' : 
                    fitter = LM_cox_fitter(model = current_model_list[j], ID = self.ID, T = self.T, E = self.E, 
                                           S = self.S, window = self.window, degree= 2, stratified = True)
                    fitter.fit(data= data_cont, weight = weight_cont)

                elif current_model_type == 'cox_no_str' : 
                    fitter = LM_cox_fitter(model = current_model_list[j], ID = self.ID, T = self.T, E = self.E, 
                                           S = self.S, window = self.window, degree= 2, stratified = False)
                    fitter.fit(data= data_cont, weight = weight_cont)

                elif current_model_type in ['mlp', 'knn'] : 
                    fitter = LM_sklearn_fitter(model = current_model_list[j], ID = self.ID, E = self.E, k_bin = self.k_bin)
                    fitter.fit(data= data_disc)
                else :
                    fitter = LM_sklearn_fitter(model = current_model_list[j], ID = self.ID, E = self.E, k_bin = self.k_bin)
                    fitter.fit(data= data_disc, weight = weight_disc)
                new_model_list.append(fitter)
        
        self.model_list = new_model_list
                
        return self.model_list
    
    def predict(self, data_cont, data_disc) :
        stacked = []
        for fitter in self.model_list : 
            module_tree = getattr(fitter.model,'__module__',None)
            parent = module_tree.split('.')[0] if module_tree else None
            
            if parent == lifelines.__name__:
                stacked.append(fitter.predict(data_cont))
            else :
                stacked.append(fitter.predict(data_disc))
        
        stacked = np.array(stacked).T
        
        return stacked
                