import time
import os
import sys
import numpy as np
import pandas as pd

start_time=time.time()

task = str(sys.argv[1])
algo = str(sys.argv[2])
n = int(sys.argv[3])
n_reps = int(sys.argv[4])
path_to_res = str(sys.argv[5])

test_size = 10000
n_reps_10perc = int(n_reps/10)

path = os.path.join(path_to_res,task,algo,"n"+str(n))

list_df = []
cond_indiv_err = np.zeros(test_size)
for i in range(n_reps):
    rep_file = os.path.join(path,"rep"+str(i+1)+".h5")
    rep_sigma_file = os.path.join(path,"rep"+str(i+1)+"_sigma.h5")    
    assert os.path.isfile(rep_file) and os.path.isfile(rep_sigma_file)
    list_df.append(pd.read_hdf(rep_file,"res"))
    temp_df = pd.read_hdf(rep_sigma_file,"res_sigma")
    assert temp_df.shape[0]==1
    assert temp_df.shape[1]==test_size
    cond_indiv_err = cond_indiv_err + np.array(temp_df.iloc[0,:])/n_reps
    try:
        if (i+1)%n_reps_10perc == 0:
            print(str(i+1)+" replications out of "+str(n_reps)+ " combined")
    except:
        pass

df = pd.concat(list_df)
df.to_hdf(os.path.join(path,"all_reps.h5"),"all_reps")
df2 = pd.DataFrame([cond_indiv_err])
df2.to_hdf(os.path.join(path,"cond_indiv_err.h5"),"cond_indiv_err")

print(" ")
print("Finished combining:",time.time()-start_time)
