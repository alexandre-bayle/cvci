import time

start_time = time.time()

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from math import sqrt
import scipy.stats as stats
from xgboost import XGBRFRegressor, XGBRFClassifier


print("Importing the packages:",time.time()-start_time)

task = str(sys.argv[1])
n = int(sys.argv[2])
k = int(sys.argv[3])
rep = int(sys.argv[4])
path_to_res = str(sys.argv[5])
path_to_data = str(sys.argv[6])
LOOCV = bool(int(sys.argv[7]))

r = int(n/k)

# Clf: classification
algos_Clf = ["LR","RF","NN"] # LR: Logistic Regression, RF: Random Forest, NN: Neural Network

# Reg: regression
algos_Reg = ["RR","RF","NN"] # RR: Ridge Regression, RF: Random Forest, NN: Neural Network

if task=="Clf":
    algos = algos_Clf
    # define loss as the 0-1 classification error loss
    def loss(y1,y2):
        return 1*(y1 != y2)
else: # task=="Reg"
    algos = algos_Reg
    # define the loss as the squared error loss
    def loss(y1,y2):
        return (y1-y2)**2

random_seed = n + 50001*rep
np.random.seed(random_seed)

def model_choice(algo,random_seed_add):
    if algo == "RF" and task == "Clf":
        model = XGBRFClassifier(n_estimators=100,subsample=0.5,max_depth=1,random_state=99+random_seed_add)
    elif algo == "RF" and task == "Reg":
        model = XGBRFRegressor(n_estimators=100,subsample=0.5,max_depth=1,random_state=99+random_seed_add)
    elif algo == "LR": # task=="Clf"
        model = LogisticRegression(solver="lbfgs",C=0.001,random_state=99+random_seed_add)
    elif algo == "RR": # task=="Reg"
        model = Ridge(alpha=1000000.0,random_state=99+random_seed_add)
    elif algo == "NN" and task == "Clf":
        model = MLPClassifier(hidden_layer_sizes=(8,4,),alpha=100.0,random_state=99+random_seed_add)
    elif algo == "NN" and task == "Reg":
        model = MLPRegressor(hidden_layer_sizes=(8,4,),alpha=100.0,random_state=99+random_seed_add)
    else:
        pass
    return model

data_file_name = os.path.join(path_to_data,"reps","rep"+str(rep)+".h5")

test_data_file_name = os.path.join(path_to_data,"test_data.h5")

def create_folder(folder):
    if not os.path.isdir(folder):
        try:
            os.mkdir(folder)
        except:
            pass

res_folder = path_to_res
create_folder(res_folder)

res_folder = os.path.join(res_folder,task)
create_folder(res_folder)

res_folder_dict = {}

for algo in algos:
    res_folder_algo = os.path.join(res_folder,algo)
    create_folder(res_folder_algo)
    res_folder_algo = os.path.join(res_folder_algo,"n"+str(n))
    create_folder(res_folder_algo)
    res_folder_dict[algo] = res_folder_algo

for algoA in algos:
    for algoB in algos:
        if algoA != algoB:
            comp = algoA + "_" + algoB
            res_folder_comp = os.path.join(res_folder,comp)
            create_folder(res_folder_comp)
            res_folder_comp = os.path.join(res_folder_comp,"n"+str(n))
            create_folder(res_folder_comp)
            res_folder_dict[comp] = res_folder_comp

current_time = time.time()
print(data_file_name)
df = pd.read_hdf(data_file_name,"chunk")
df = df.iloc[:n,:] # to keep the first n Z_i's out of the n_max samples where n_max is the maximal sample size
data = df.values
print("Reading replication data:",time.time()-current_time)

current_time = time.time()
test_data_df = pd.read_hdf(test_data_file_name,"test_data")
cond_err_data = test_data_df.values
print("Reading test data:",time.time()-current_time)

indiv_err_kfoldCV_dict = {} # to store the n h_n for k-fold CV for all algos
cond_fold_err_kfoldCV_dict = {} # to store the 10 cond fold errors for k-fold CV for all algos
cond_fold_std_kfoldCV_dict = {} # to store the 10 cond fold standard deviations for k-fold CV for all algos
cond_indiv_err_kfoldCV_dict = {} # to store the cond indiv errors for k-fold CV for all algos (but only save 10,000 of them)
cond_std_kfoldCV_dict = {} # to store the cond standard deviation for k-fold CV for all algos
fold_err_5x2CV_dict = {} # to store the 5x2 fold errors for 5x2CV for all algos
avg_cond_err_5x2CV_dict = {} # to store the average of the 5x2 cond fold errors for 5x2CV for all algos
J_err_rep_tt_dict = {} # to store the J validation errors for the repeated train-validation splitting procedure for all algos (J will be equal to 10)
avg_cond_err_rep_tt_dict = {} # to store the average of the J cond errors for the repeated train-validation splitting procedure for all algos

# Organize data

X = data[:,1:]
y = data[:,0]
if task == "Reg":
    y = np.sign(y)*np.log(1+np.abs(y))

X_cond_err = cond_err_data[:,1:]
y_cond_err = cond_err_data[:,0]
if task == "Reg":
    y_cond_err = np.sign(y_cond_err)*np.log(1+np.abs(y_cond_err))

assert(X.shape[1]==X_cond_err.shape[1])

nb_features = X.shape[1]
test_size = X_cond_err.shape[0]

current_time = time.time()

for algo in algos:
    
    start_algo_time = time.time()
    
    indiv_err_kfoldCV_dict[algo] = []
    cond_fold_err_kfoldCV_dict[algo] = []
    cond_fold_std_kfoldCV_dict[algo] = []
    cond_indiv_err_kfoldCV_dict[algo] = np.zeros(test_size)
    cond_std_kfoldCV_dict[algo] = []
    fold_err_5x2CV_dict[algo] = []
    avg_cond_err_5x2CV_dict[algo] = []
    J_err_rep_tt_dict[algo] = []
    avg_cond_err_rep_tt_dict[algo] = []
    
    # k-fold CV (we will store quantities to compute confidence intervals and perform tests for our procedures (sigma_in, sigma_out), the hold-out procedure and the CV t procedure) # and also LOOCV for RR
    current_algo_time = time.time()
    
    kfold_split = KFold(n_splits=k, shuffle=False)

    for fold, (train_index, test_index) in enumerate(kfold_split.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if not LOOCV:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        print(X_test[0,0])
        model = model_choice(algo,0)
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        if not LOOCV:
            X_cond_err_temp = scaler.transform(X_cond_err)
        else:
            X_cond_err_temp = X_cond_err
        pred_values_cond_err = model.predict(X_cond_err_temp)
        score_all = loss(y_test,pred_values)
        indiv_err_kfoldCV_dict[algo].extend(score_all)
        score_all_cond_err = loss(y_cond_err,pred_values_cond_err)
        cond_fold_err_kfoldCV_dict[algo].append(np.mean(score_all_cond_err))
        cond_fold_std_kfoldCV_dict[algo].append(np.std(score_all_cond_err,ddof=1))
        cond_indiv_err_kfoldCV_dict[algo] = cond_indiv_err_kfoldCV_dict[algo] + np.array(score_all_cond_err)/k
    cond_std_kfoldCV_dict[algo].append(cond_indiv_err_kfoldCV_dict[algo].std())
    
    print(algo,"k-fold CV:",time.time()-current_algo_time)
    
    # LOOCV only for Ridge Regression
    if algo == "RR" and LOOCV:
        indiv_err_LOOCV = []
        indiv_cond_err_LOOCV = []
        pen = 1000000.0
        X_inter = np.append(np.ones((n,1)), X, axis=1)
        X_cond_err_inter = np.append(np.ones((X_cond_err.shape[0],1)), X_cond_err, axis=1)
        assert X_inter.shape[1] == (1+nb_features)
        assert X_cond_err_inter.shape[1] == (1+nb_features)
        D = np.eye(1+nb_features)
        D[0,0] = 0
        M_temp = np.dot(X_inter.T,X_inter) + pen * D
        M = np.linalg.inv(M_temp)
        v = np.dot(X_inter.T,y)
        w = np.dot(M,v)
        for i in range(n):
            x_i = X_inter[i,:]
            h_i = np.dot(x_i, np.dot(M, x_i))
            w_i = w + np.dot(M,x_i) * (np.dot(w,x_i)-y[i])/(1-h_i)
            new_pred_minus_i = np.dot(X_cond_err_inter,w_i)
            score_i = loss(np.dot(w_i,x_i),y[i])
            indiv_err_LOOCV.append(score_i)
            score_cond_err_i = loss(new_pred_minus_i,y_cond_err)
            indiv_cond_err_LOOCV.append(np.mean(score_cond_err_i))
        indiv_err_LOOCV = np.array(indiv_err_LOOCV)
        indiv_cond_err_LOOCV = np.array(indiv_cond_err_LOOCV) 
   
    # 5x2CV procedure (we will store quantities to compute confidence intervals and perform tests for the 5x2 CV procedure)
    current_algo_time = time.time()
    
    scores_cond_err = []
    
    for i in range(5):
        
        two_fold_split = KFold(n_splits=2, shuffle=True, random_state=i)
        
        for fold, (train_index, test_index) in enumerate(two_fold_split.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if not LOOCV:
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            print(X_test[0,0])
            model = model_choice(algo,0)
            model.fit(X_train, y_train)
            pred_values = model.predict(X_test)
            if not LOOCV:
                X_cond_err_temp = scaler.transform(X_cond_err)
            else:
                X_cond_err_temp = X_cond_err
            pred_values_cond_err = model.predict(X_cond_err_temp)
            score_all = loss(y_test,pred_values)
            score_current = np.mean(score_all)
            fold_err_5x2CV_dict[algo].append(score_current)
            score_all_cond_err = loss(y_cond_err,pred_values_cond_err)
            score_current_cond_err = np.mean(score_all_cond_err)
            scores_cond_err.append(score_current_cond_err)
            
    avg_cond_err_5x2CV_dict[algo].append(np.mean(scores_cond_err))
    print(algo,"5x2CV:",time.time()-current_algo_time)
    
    # Repeated train-validation splitting procedure (we will store quantities to compute confidence intervals and perform tests for the repeated train-validation splitting procedure)
    current_algo_time = time.time()
    
    scores_cond_err = []
    
    J = 10
    
    for j in range(J):
        if k!=n:
            size = 1/k
        else:
            size = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, shuffle=True, random_state=j)
        if not LOOCV:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        print(X_test[0,0])
        model = model_choice(algo,0)
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        if not LOOCV:
            X_cond_err_temp = scaler.transform(X_cond_err)
        else:
            X_cond_err_temp = X_cond_err
        pred_values_cond_err = model.predict(X_cond_err_temp)
        score_all = loss(y_test,pred_values)
        score_current = np.mean(score_all)
        J_err_rep_tt_dict[algo].append(score_current)
        score_all_cond_err = loss(y_cond_err,pred_values_cond_err)
        score_current_cond_err = np.mean(score_all_cond_err)
        scores_cond_err.append(score_current_cond_err)
        
    avg_cond_err_rep_tt_dict[algo].append(np.mean(scores_cond_err))
    
    print(algo,"rep tt:",time.time()-current_algo_time)
    
    indiv_err_kfoldCV_dict[algo] = np.array(indiv_err_kfoldCV_dict[algo])
    cond_fold_err_kfoldCV_dict[algo] = np.array(cond_fold_err_kfoldCV_dict[algo])
    cond_fold_std_kfoldCV_dict[algo] = np.array(cond_fold_std_kfoldCV_dict[algo])
    # cond_indiv_err_kfoldCV_dict[algo] is already a np.array
    cond_std_kfoldCV_dict[algo] = np.array(cond_std_kfoldCV_dict[algo])
    fold_err_5x2CV_dict[algo] = np.array(fold_err_5x2CV_dict[algo])
    avg_cond_err_5x2CV_dict[algo] = np.array(avg_cond_err_5x2CV_dict[algo])
    J_err_rep_tt_dict[algo] = np.array(J_err_rep_tt_dict[algo])
    avg_cond_err_rep_tt_dict[algo] = np.array(avg_cond_err_rep_tt_dict[algo])
    
    print(algo,"total time for all procedures:",time.time()-start_algo_time)


for algoA in algos:
    for algoB in algos:
        if algoA != algoB:
            comp = algoA + "_"  + algoB
            indiv_err_kfoldCV_dict[comp] = indiv_err_kfoldCV_dict[algoA] - indiv_err_kfoldCV_dict[algoB]
            cond_fold_err_kfoldCV_dict[comp] = cond_fold_err_kfoldCV_dict[algoA] - cond_fold_err_kfoldCV_dict[algoB]
            cond_fold_std_kfoldCV_dict[comp] = cond_fold_std_kfoldCV_dict[algoA] - cond_fold_std_kfoldCV_dict[algoB] # not to be used for comparisons (just for simplicity when saving results)
            cond_indiv_err_kfoldCV_dict[comp] = cond_indiv_err_kfoldCV_dict[algoA] - cond_indiv_err_kfoldCV_dict[algoB]
            cond_std_kfoldCV_dict[comp] = np.array([cond_indiv_err_kfoldCV_dict[comp].std()])
            fold_err_5x2CV_dict[comp] = fold_err_5x2CV_dict[algoA] - fold_err_5x2CV_dict[algoB]
            avg_cond_err_5x2CV_dict[comp] = avg_cond_err_5x2CV_dict[algoA] - avg_cond_err_5x2CV_dict[algoB]
            J_err_rep_tt_dict[comp] = J_err_rep_tt_dict[algoA] - J_err_rep_tt_dict[algoB]
            avg_cond_err_rep_tt_dict[comp] = avg_cond_err_rep_tt_dict[algoA] - avg_cond_err_rep_tt_dict[algoB]

# We now have all the quantities needed to compute the 2-sided confidence intervals for the different methods for all algos (for estimating coverage probability and width with all replications later on). We also have what we need to define the tests for comparing pairs of algos (for estimating size and power with all replications later on).

def aggr_ours(indiv_err_kfoldCV, cond_fold_err_kfoldCV, version="out"):
    squared_indiv_err_kfoldCV = indiv_err_kfoldCV ** 2
    fold_err_kfoldCV = []
    for i in range(k):
        fold_err_kfoldCV.append(np.mean(indiv_err_kfoldCV[(i*r):((i+1)*r)])) # works when k divides n (which is the case for the plots we present)
    fold_err_kfoldCV = np.array(fold_err_kfoldCV)
    squared_fold_err_kfoldCV = fold_err_kfoldCV ** 2
    kfoldCV_err = np.mean(fold_err_kfoldCV)
    kfold_test_err = np.mean(cond_fold_err_kfoldCV)
    if version=="in":
        if k!=n:
            sigma_est = np.sqrt((n/(n-k))*(np.mean(squared_indiv_err_kfoldCV) - np.mean(squared_fold_err_kfoldCV)))
        else: # LOOCV, sigma_in is not defined in this case
            sigma_est = 0
    else: # version=="out"
        sigma_est = np.sqrt(np.mean(squared_indiv_err_kfoldCV) - kfoldCV_err ** 2)
    return kfoldCV_err, kfold_test_err, sigma_est

def aggr_hold_out(indiv_err_kfoldCV, cond_fold_err_kfoldCV):
    ho_indiv_err_kfoldCV = indiv_err_kfoldCV[:r]
    ho_squared_indiv_err_kfoldCV = ho_indiv_err_kfoldCV ** 2
    ho_fold_err_kfoldCV = np.mean(ho_indiv_err_kfoldCV)
    ho_kfold_test_err = cond_fold_err_kfoldCV[0]
    sigma_est = np.sqrt(np.mean(ho_squared_indiv_err_kfoldCV) - ho_fold_err_kfoldCV ** 2)
    return ho_fold_err_kfoldCV, ho_kfold_test_err, sigma_est

def aggr_CV_t(indiv_err_kfoldCV, cond_fold_err_kfoldCV):
    squared_indiv_err_kfoldCV = indiv_err_kfoldCV ** 2
    fold_err_kfoldCV = []
    for i in range(k):
        fold_err_kfoldCV.append(np.mean(indiv_err_kfoldCV[(i*r):((i+1)*r)])) # works when k divides n (which is the case for the plots we present)
    fold_err_kfoldCV = np.array(fold_err_kfoldCV)
    squared_fold_err_kfoldCV = fold_err_kfoldCV ** 2
    kfoldCV_err = np.mean(fold_err_kfoldCV)
    kfold_test_err = np.mean(cond_fold_err_kfoldCV)
    sigma_est = np.std(fold_err_kfoldCV, ddof=1)
    return kfoldCV_err, kfold_test_err, sigma_est

def aggr_5x2CV(fold_err_5x2CV, avg_cond_err_5x2CV):
    p_5x2CV = fold_err_5x2CV[0] # fold error of the first fold on the first iteration of the 5x2 CV
    cond_err_5x2CV = avg_cond_err_5x2CV[0] # it is an array of size 1
    variances = np.zeros(5)
    for i in range(5):
        variances[i] = np.var(fold_err_5x2CV[(2*i):(2*i+2)], ddof=1)
    sigma_est = np.sqrt(np.mean(variances))
    return p_5x2CV, cond_err_5x2CV, sigma_est

def aggr_rep_tt(J_err_rep_tt, avg_cond_err_rep_tt, corrected=True):
    rep_tt_mean = np.mean(J_err_rep_tt)
    rep_tt_cond_err = avg_cond_err_rep_tt[0] # it is an array of size 1
    if not corrected: # repeated train-validation splitting procedure
        sigma_est = np.std(J_err_rep_tt, ddof=1)
    else: # corrected repeated train-validation splitting procedure
        adj_factor = (1/J)+(1/(k-1)) # J = 10
        sigma_est = np.std(J_err_rep_tt, ddof=1) * np.sqrt(J * adj_factor)
    return rep_tt_mean, rep_tt_cond_err, sigma_est

# Constructing the 2-sided and 1-sided confidence intervals

def CI_2sided(center, scale, df=None, distrib="normal", alpha=0.05):
    if distrib=="normal":
        q = stats.norm.ppf(1-alpha/2,0,1)
    else:
        q = stats.t.ppf(1-alpha/2,df)
    variation = q * scale
    l_bound = center - variation
    u_bound = center + variation
    return (l_bound,u_bound)

def CI_1sided(center, scale, df=None, distrib="normal", side="upper", alpha=0.05):
    if side=="upper":
        level = alpha
    else: # side=="lower"
        level = 1-alpha
    if distrib=="normal":
        q = stats.norm.ppf(level,0,1)
    else:
        q = stats.t.ppf(level,df)
    cutoff = abs(q * scale)
    u_bound = center + cutoff
    l_bound = center - cutoff
    if side=="upper":
        bound = u_bound
    else:
        bound = l_bound
    return bound, cutoff

def CI_2sided_results(conf_int,cond_err):
    l_bound, u_bound = conf_int
    is_contained = (l_bound<=cond_err) and (cond_err<=u_bound)
    width = u_bound-l_bound
    return (is_contained,width)

def CI_1sided_results(conf_bound,cond_err, side="upper"):
    bound, cutoff = conf_bound
    # TP, TN, FP, FN: T=True, F=False, P=Positive (rejected), N=Negative (not rejected)
    # The following 4 quantities only apply to the "upper" case (we compute them for lower as well but it does not matter since we do not use them)
    is_TN = (bound>=0) and (cond_err>=0)
    is_TP = (bound<0) and (cond_err<0)
    is_FN = (bound>=0) and (cond_err<0)
    is_FP = (bound<0) and (cond_err>=0)
    if side=="upper":
        is_contained = (cond_err <= bound)
    else: # side=="lower"
        is_contained = (bound <= cond_err)
    return (is_TN,is_TP,is_contained,cutoff)

# Now that we defined the functions, let's apply them appropriately to single algos and comparisons between algos and store all the results we need

current_time = time.time()
for key in list(indiv_err_kfoldCV_dict.keys()): # loop on all single algos and all comparisons
    print(key)
    R_hat_in, R_cond_in, sigma_in = aggr_ours(indiv_err_kfoldCV_dict[key], cond_fold_err_kfoldCV_dict[key], version="in")
    R_hat_out, R_cond_out, sigma_out = aggr_ours(indiv_err_kfoldCV_dict[key], cond_fold_err_kfoldCV_dict[key], version="out")
    R_hat_ho, R_cond_ho, sigma_ho = aggr_hold_out(indiv_err_kfoldCV_dict[key], cond_fold_err_kfoldCV_dict[key])
    R_hat_CV_t, R_cond_CV_t, sigma_CV_t = aggr_CV_t(indiv_err_kfoldCV_dict[key], cond_fold_err_kfoldCV_dict[key])
    R_hat_5x2CV, R_cond_5x2CV, sigma_5x2CV = aggr_5x2CV(fold_err_5x2CV_dict[key], avg_cond_err_5x2CV_dict[key])
    R_hat_rep_tt, R_cond_rep_tt, sigma_rep_tt = aggr_rep_tt(J_err_rep_tt_dict[key], avg_cond_err_rep_tt_dict[key],corrected=False)
    R_hat_rep_tt_adj, R_cond_rep_tt_adj, sigma_rep_tt_adj = aggr_rep_tt(J_err_rep_tt_dict[key], avg_cond_err_rep_tt_dict[key], corrected=True)
    if key == "RR" and LOOCV:
        R_hat_LOOCV, R_cond_LOOCV, sigma_LOOCV = aggr_ours(indiv_err_LOOCV, indiv_cond_err_LOOCV, version="out")
    
    res_array = np.concatenate([indiv_err_kfoldCV_dict[key],cond_fold_err_kfoldCV_dict[key],cond_fold_std_kfoldCV_dict[key],fold_err_5x2CV_dict[key],avg_cond_err_5x2CV_dict[key],J_err_rep_tt_dict[key],avg_cond_err_rep_tt_dict[key],cond_std_kfoldCV_dict[key]])
    res_array = np.append(res_array,[R_hat_in, R_cond_in, sigma_in, R_hat_out, R_cond_out, sigma_out, R_hat_ho, R_cond_ho, sigma_ho, R_hat_CV_t, R_cond_CV_t, sigma_CV_t, R_hat_5x2CV, R_cond_5x2CV, sigma_5x2CV, R_hat_rep_tt, R_cond_rep_tt, sigma_rep_tt, R_hat_rep_tt_adj, R_cond_rep_tt_adj, sigma_rep_tt_adj])
    assert len(res_array)==(n+k+k+10+1+J+1+1+3*7)
    if key == "RR" and LOOCV:
        res_array = np.append(res_array, [R_hat_LOOCV, R_cond_LOOCV, sigma_LOOCV])
        assert len(res_array)==(n+k+k+10+1+J+1+1+3*8)
    
    CI_2sided_in = CI_2sided(R_hat_in,sigma_in/sqrt(n),distrib="normal")
    CI_2sided_out = CI_2sided(R_hat_out,sigma_out/sqrt(n),distrib="normal")
    CI_2sided_ho = CI_2sided(R_hat_ho,sigma_ho/sqrt(n/k),distrib="normal")
    CI_2sided_CV_t = CI_2sided(R_hat_CV_t,sigma_CV_t/sqrt(k),df=k-1,distrib="t")
    CI_2sided_5x2CV = CI_2sided(R_hat_5x2CV,sigma_5x2CV,df=5,distrib="t")
    CI_2sided_rep_tt = CI_2sided(R_hat_rep_tt,sigma_rep_tt/sqrt(J),df=J-1,distrib="t")
    CI_2sided_rep_tt_adj = CI_2sided(R_hat_rep_tt_adj,sigma_rep_tt_adj/sqrt(J),df=J-1,distrib="t")
    
    is_contained_in,width_in = CI_2sided_results(CI_2sided_in,R_cond_in)
    is_contained_out,width_out = CI_2sided_results(CI_2sided_out,R_cond_out)
    is_contained_ho,width_ho = CI_2sided_results(CI_2sided_ho,R_cond_ho)
    is_contained_CV_t,width_CV_t = CI_2sided_results(CI_2sided_CV_t,R_cond_CV_t)
    is_contained_5x2CV,width_5x2CV = CI_2sided_results(CI_2sided_5x2CV,R_cond_5x2CV)
    is_contained_rep_tt,width_rep_tt = CI_2sided_results(CI_2sided_rep_tt,R_cond_rep_tt)
    is_contained_rep_tt_adj,width_rep_tt_adj = CI_2sided_results(CI_2sided_rep_tt_adj,R_cond_rep_tt_adj)
        
    res_array = np.append(res_array,[is_contained_in,width_in,is_contained_out,width_out,is_contained_ho,width_ho,is_contained_CV_t,width_CV_t,is_contained_5x2CV,width_5x2CV,is_contained_rep_tt,width_rep_tt,is_contained_rep_tt_adj,width_rep_tt_adj])
    if key == "RR" and LOOCV:
        CI_2sided_LOOCV = CI_2sided(R_hat_LOOCV,sigma_LOOCV/sqrt(n),distrib="normal")
        is_contained_LOOCV,width_LOOCV = CI_2sided_results(CI_2sided_LOOCV,R_cond_LOOCV)
        res_array = np.append(res_array,[is_contained_LOOCV,width_LOOCV])

    CI_1sided_in = CI_1sided(R_hat_in,sigma_in/sqrt(n),distrib="normal",side="upper")
    CI_1sided_out = CI_1sided(R_hat_out,sigma_out/sqrt(n),distrib="normal",side="upper")
    CI_1sided_ho = CI_1sided(R_hat_ho,sigma_ho/sqrt(n/k),distrib="normal",side="upper")
    CI_1sided_CV_t = CI_1sided(R_hat_CV_t,sigma_CV_t/sqrt(k),df=k-1,distrib="t",side="upper")
    CI_1sided_5x2CV = CI_1sided(R_hat_5x2CV,sigma_5x2CV,df=5,distrib="t",side="upper")
    CI_1sided_rep_tt = CI_1sided(R_hat_rep_tt,sigma_rep_tt/sqrt(J),df=J-1,distrib="t",side="upper")
    CI_1sided_rep_tt_adj = CI_1sided(R_hat_rep_tt_adj,sigma_rep_tt_adj/sqrt(J),df=J-1,distrib="t",side="upper")
    
    is_TN_in,is_TP_in,is_contained_in,cutoff_in = CI_1sided_results(CI_1sided_in,R_cond_in,side="upper")
    is_TN_out,is_TP_out,is_contained_out,cutoff_out = CI_1sided_results(CI_1sided_out,R_cond_out,side="upper")
    is_TN_ho,is_TP_ho,is_contained_ho,cutoff_ho = CI_1sided_results(CI_1sided_ho,R_cond_ho,side="upper")
    is_TN_CV_t,is_TP_CV_t,is_contained_CV_t,cutoff_CV_t = CI_1sided_results(CI_1sided_CV_t,R_cond_CV_t,side="upper")
    is_TN_5x2CV,is_TP_5x2CV,is_contained_5x2CV,cutoff_5x2CV = CI_1sided_results(CI_1sided_5x2CV,R_cond_5x2CV,side="upper")
    is_TN_rep_tt,is_TP_rep_tt,is_contained_rep_tt,cutoff_rep_tt = CI_1sided_results(CI_1sided_rep_tt,R_cond_rep_tt,side="upper")
    is_TN_rep_tt_adj,is_TP_rep_tt_adj,is_contained_rep_tt_adj,cutoff_rep_tt_adj = CI_1sided_results(CI_1sided_rep_tt_adj,R_cond_rep_tt_adj,side="upper")
    
    res_array = np.append(res_array,[is_TN_in,is_TP_in,is_contained_in,cutoff_in,is_TN_out,is_TP_out,is_contained_out,cutoff_out,is_TN_ho,is_TP_ho,is_contained_ho,cutoff_ho,is_TN_CV_t,is_TP_CV_t,is_contained_CV_t,cutoff_CV_t,is_TN_5x2CV,is_TP_5x2CV,is_contained_5x2CV,cutoff_5x2CV,is_TN_rep_tt,is_TP_rep_tt,is_contained_rep_tt,cutoff_rep_tt,is_TN_rep_tt_adj,is_TP_rep_tt_adj,is_contained_rep_tt_adj,cutoff_rep_tt_adj])
    
    CI_1sided_in = CI_1sided(R_hat_in,sigma_in/sqrt(n),distrib="normal",side="lower")
    CI_1sided_out = CI_1sided(R_hat_out,sigma_out/sqrt(n),distrib="normal",side="lower")
    CI_1sided_ho = CI_1sided(R_hat_ho,sigma_ho/sqrt(n/k),distrib="normal",side="lower")
    CI_1sided_CV_t = CI_1sided(R_hat_CV_t,sigma_CV_t/sqrt(k),df=k-1,distrib="t",side="lower")
    CI_1sided_5x2CV = CI_1sided(R_hat_5x2CV,sigma_5x2CV,df=5,distrib="t",side="lower")
    CI_1sided_rep_tt = CI_1sided(R_hat_rep_tt,sigma_rep_tt/sqrt(J),df=J-1,distrib="t",side="lower")
    CI_1sided_rep_tt_adj = CI_1sided(R_hat_rep_tt_adj,sigma_rep_tt_adj/sqrt(J),df=J-1,distrib="t",side="lower")
    
    is_TN_in,is_TP_in,is_contained_in,cutoff_in = CI_1sided_results(CI_1sided_in,R_cond_in,side="lower")
    is_TN_out,is_TP_out,is_contained_out,cutoff_out = CI_1sided_results(CI_1sided_out,R_cond_out,side="lower")
    is_TN_ho,is_TP_ho,is_contained_ho,cutoff_ho = CI_1sided_results(CI_1sided_ho,R_cond_ho,side="lower")
    is_TN_CV_t,is_TP_CV_t,is_contained_CV_t,cutoff_CV_t = CI_1sided_results(CI_1sided_CV_t,R_cond_CV_t,side="lower")
    is_TN_5x2CV,is_TP_5x2CV,is_contained_5x2CV,cutoff_5x2CV = CI_1sided_results(CI_1sided_5x2CV,R_cond_5x2CV,side="lower")
    is_TN_rep_tt,is_TP_rep_tt,is_contained_rep_tt,cutoff_rep_tt = CI_1sided_results(CI_1sided_rep_tt,R_cond_rep_tt,side="lower")
    is_TN_rep_tt_adj,is_TP_rep_tt_adj,is_contained_rep_tt_adj,cutoff_rep_tt_adj = CI_1sided_results(CI_1sided_rep_tt_adj,R_cond_rep_tt_adj,side="lower")
    
    res_array = np.append(res_array,[is_TN_in,is_TP_in,is_contained_in,cutoff_in,is_TN_out,is_TP_out,is_contained_out,cutoff_out,is_TN_ho,is_TP_ho,is_contained_ho,cutoff_ho,is_TN_CV_t,is_TP_CV_t,is_contained_CV_t,cutoff_CV_t,is_TN_5x2CV,is_TP_5x2CV,is_contained_5x2CV,cutoff_5x2CV,is_TN_rep_tt,is_TP_rep_tt,is_contained_rep_tt,cutoff_rep_tt,is_TN_rep_tt_adj,is_TP_rep_tt_adj,is_contained_rep_tt_adj,cutoff_rep_tt_adj])
    
    if key == "RR" and LOOCV:
        assert len(res_array)==(n+k+k+10+1+J+1+1+3*8+2*8+4*7+4*7)
    else:
        assert len(res_array)==(n+k+k+10+1+J+1+1+3*7+2*7+4*7+4*7)
    
    # finally we store everything in a DataFrame in the appropriate files
    
    df = pd.DataFrame([res_array])
    res_file_name = os.path.join(res_folder_dict[key],"rep"+str(rep)+".h5")
    df.to_hdf(res_file_name,"res")

    assert len(cond_indiv_err_kfoldCV_dict[key])==test_size
    df2 = pd.DataFrame([cond_indiv_err_kfoldCV_dict[key][:10000]])
    assert df2.shape[1]==10000
    res_sigma_file_name = os.path.join(res_folder_dict[key],"rep"+str(rep)+"_sigma.h5")
    df2.to_hdf(res_sigma_file_name,"res_sigma")

print("Final results and storing:",time.time()-current_time)

print(" ")
print("Total run time:",time.time()-start_time)
