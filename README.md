#### This code accompanies the paper "Cross-validation Confidence Intervals for Test Error" by Pierre Bayle, Alexandre Bayle, Lucas Janson, and Lester Mackey.

Required Python packages:
- os
- sys
- time
- math
- numpy
- pandas
- sklearn
- scipy
- xgboost
- matplotlib

Instructions to download datasets:
- for Higgs dataset, go to UCI repository (link in the References section of the paper) and download HIGGS.csv.gz
- for FlightsDelay dataset, go to Kaggle (link in the References section of the paper) and download 810_1496_compressed_flights.csv.zip (by downloading only the flights data, not the whole dataset)

Note: you should save them in two different folders as we will be creating a subfolder for saving the replications and they will collide if the data folder is the same for both datasets.

In the root folder, we provide 4 Python scripts :

- processing_dataset.py: performs the initial processing of the dataset; takes as inputs the task ("Clf" or "Reg") and the path to the folder where you saved the downloaded dataset (Higgs dataset for "Clf" and FlightsDelay dataset for "Reg"),

- create_replications.py: performs the creation of the independent replications for the dataset and saves them in the dataset folder; takes as inputs the task ("Clf" or "Reg"), the maximal sample size, the number of replications and the path to the folder where you saved the downloaded dataset (Higgs dataset for "Clf" and FlightsDelay dataset for "Reg"),

- cv_num_exper.py: main script, computes for all procedures the 2-sided confidence interval for each algorithm and the test for each comparison of a pair of algorithms, also computes and saves the quantities needed in the next scripts to estimate and plot coverage probability, width, size and power using all replications; takes as inputs the task ("Clf" or "Reg"), the sample size, the number of folds, the replication index, the path to the folder where you want to save the results, the path to the folder where you saved the downloaded dataset (Higgs dataset for "Clf" and FlightsDelay dataset for "Reg") and a boolean that you should set to 1 if you want the LOOCV results presented in the paper or to 0 otherwise,

- combine_results.py: performs the combining of the results of cv_num_exper.py for all replications; takes as inputs the task ("Clf" or "Reg"), the algorithm or comparison of a pair of algorithms you want to combine results for, the sample size, the number of replications and the path to the folder where you want to save the results.

Note: the signed log transform is applied to the target variable for the FlightsDelay dataset in cv_num_exper.py, not in processing_dataset.py.

We also provide bash scripts we used to run these Python scripts on the Harvard FASRC Cluster.

With the files provided, you can recover our plots by following the instructions below provided you adapt the bash scripts to the infrastructure you have access to.

Start a new Terminal and run the code below:
cd <path_to_folder_containing_the_scripts>
module load <my_Anaconda_installation>
source activate <my_environment> # where you installed the required packages listed earlier
python processing_dataset.py Clf <path_to_clf_data>
python processing_dataset.py Reg <path_to_reg_data>
python create_replications.py Clf 11000 500 <path_to_clf_data> # 11000 is the largest sample size
python create_replications.py Reg 11000 500 <path_to_reg_data> # 11000 is the largest sample size

Start a new Terminal and run the code below:
./runAllExper.sh Clf 500 <path_to_res> <path_to_clf_data> 0
./runAllExper.sh Reg 500 <path_to_res> <path_to_reg_data> 0 # change 0 to 1 if you want the LOOCV results
./combineAllExper.sh Clf 500 <path_to_res>
./combineAllExper.sh Reg 500 <path_to_res>

Use the Jupyter Notebook we provide for outputting plots.

Note: in runOneExper.sh and combineOneExper.sh, add the Anaconda installation and environment you use.

Note: we ask to open a new Terminal to avoid an issue with the Anaconda environment not loading properly in the cluster.

You will need 20-30 GB to store results. You also need enough RAM to process the datasets (16 GB is more than enough).
