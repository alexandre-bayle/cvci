import os
import sys
import time
import numpy as np
import pandas as pd

start_time = time.time()

task = str(sys.argv[1])
n = int(sys.argv[2])
n_reps = int(sys.argv[3])
path_to_data = str(sys.argv[4])

np.random.seed(99)

Clf_test_size = 5000000

data_file_name = os.path.join(path_to_data,"processed_dataset.csv")

if task == "Clf":
    data = pd.read_csv(data_file_name,index_col=0,skiprows=Clf_test_size+1,header=None,iterator=True)
else: # task == "Reg"
    data = pd.read_csv(data_file_name,index_col=0)

new_folder = os.path.join(path_to_data,"reps")
if not os.path.isdir(new_folder):
    try:
        os.mkdir(new_folder)
    except:
        pass

for i in range(1,n_reps+1):
    chunk_of_data_file_name = os.path.join(new_folder,"rep"+str(i)+".h5")
    if task == "Clf":
        current_df = data.get_chunk(n)
    else: # task == "Reg"
        current_df = data.sample(n=n,replace=True,random_state=i)
    current_df.to_hdf(chunk_of_data_file_name,"chunk")

print("Chunks have been saved")

print(" ")
print("Total run time:",time.time()-start_time)
