import os
import sys
import time
import numpy as np
import pandas as pd

start_time = time.time()

task = str(sys.argv[1])
path_to_data = str(sys.argv[2])

Clf_test_size = 5000000

processed_data_file_name = os.path.join(path_to_data,"processed_dataset.csv")
test_data_file_name = os.path.join(path_to_data,"test_data.h5")

if task == "Clf":
    data_file_name = os.path.join(path_to_data,"HIGGS.csv.gz")
    data = pd.read_csv(data_file_name,header=None,compression="gzip")
    np.random.seed(99)
    df = data.sample(frac=1)
    test_df = df.iloc[:Clf_test_size,:]
    df.to_csv(processed_data_file_name)
    print("Processed dataset has been saved")
    test_df.to_hdf(test_data_file_name,"test_data")
    print("Test set has been saved")
else: # task == "Reg"
    data_file_name = os.path.join(path_to_data,"810_1496_compressed_flights.csv.zip")
    data = pd.read_csv(data_file_name,compression="zip")
    variables_to_keep = ["AIRLINE","SCHEDULED_DEPARTURE","SCHEDULED_TIME",
                         "DISTANCE","ARRIVAL_DELAY"]
    df = data[variables_to_keep].copy()
    df.dropna(inplace = True)
    def to_minutes(scheduled_departure):
        s = str(scheduled_departure)
        l = [c for c in s]
        res = 0
        res += int("".join(l[(-2):])) # reading the minutes and adding them
        if len(l)>2:
            res += 60*int("".join(l[:(-2)])) # reading the hours, converting them into minutes and adding them
        return res
    df['SCHEDULED_DEPARTURE'] =  df['SCHEDULED_DEPARTURE'].apply(to_minutes)
    df2 = pd.get_dummies(df,columns=["AIRLINE"],drop_first=True,sparse=False)
    cols = list(df2)
    cols[0],cols[3] = cols[3],cols[0]
    df2 = df2.reindex(cols,axis=1)
    test_df = df2.sample(frac=1,replace=False,random_state=0)
    df2.to_csv(processed_data_file_name)
    print("Processed dataset has been saved")
    test_df.to_hdf(test_data_file_name,"test_data")
    print("Test set has been saved")

print(" ")
print("Total run time:",time.time()-start_time)
