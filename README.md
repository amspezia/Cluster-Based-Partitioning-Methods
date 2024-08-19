# K-Fold-Partitioning-Methods

## Using the Splitters
The code for the splitters are in `kfoldmethods/splitters` and the can be used in the same way as one would use sklearn splitters.
The available methods are DBSCV, DOBSCV, CBCV and CBCV using Minibatch k-means.
Simply copy the folder to your project to use them.

## Selected Hyperparameters 
The selected hyperparameters can be seen in the file `appendix/summary hyperparameters.csv`. When not indicated, the default value in sklearn was used.

## Results with 5 folds.
Plots for the 5-folds case that were omitted from the paper because of space are stored in `appendix/results 5 folds`.
Note that all the raw results can be download in this [link](https://drive.google.com/file/d/1Pc8f4Hbx9VGOPj2FYhVw-g5BL_tKl7mK/view?usp=sharing), as mentioned below.

## Reproducing the Experiments
The simplest way to re-execute the experiments is to run `source run_all.sh`, if you are using GNU/Linux.
To guarantee that you are using the same python libraries that we used, consider creating a new conda environment using the env file we provide. To do this, run `conda env create -f environment_file.yml`.
Running `source run_all.sh` will create and activate the env automatically if conda is installed.

If you are interested only in getting the outputs from the experiments, check the folder `appendix/results`.
The raw outputs (including joblib files) can be downloaded [here](https://drive.google.com/file/d/1Pc8f4Hbx9VGOPj2FYhVw-g5BL_tKl7mK/view?usp=sharing).

Finally, it's also possible to run each part of the experiments separately, following the instructions in the sections below.

### Hyperparameter Tuning
We tune each classifier using the full dataset prior to the main experiments.
The tuning only serves to guarantee that the classifiers have appropriate hyperparameters for each dataset.
To reproduce our results on the hyperparameters search, run `python main.py hp-search` followed by `python main.py hp-search -s` to obtain a summary of the hyperparameters selected for each dataset-classifier pair (the parameters that are not specified in the file were set to the default values of sklearn).

The outputs are stored in the folder `classifier_hyperparameters`.

### Defining the Number of Clusters
The number of clusters in each dataset is computed prior to the main experiments and used as input to the cluster-based splitting strategies.
To reproduce these experiments, run

`python main.py n-clusters-estimate`, followed by

`python main.py n-clusters-estimate -a` to obtain a CSV file with the number of clusters in each dataset. 

The outputs are stored in the folder `n_clusters_estimate`.

### Estimates of the True Performance
This step requires that the instructions from *Hyperparameter Tuning* have been performed first.
We use the hyperparameters to find estimates of the true performance of each classifier in each dataset.
To reproduce our results, run `python main.py true-estimate`, followed by `python main.py true-estimate -s` to retrieve CSV tables with the performance estimates for each dataset, classifier, and iteration.
Finally, `python main.py true-estimate -a` produces some CSV files with summary results as well as some figures omitted from the paper because of space. 

### Splitters Estimates of the Performance
This step requires that the instructions from *Hyperparameter Tuning* and *Defining the Number of Clusters* have been performed first.
To reproduce this experiment, run `python main.py compare-splitters`.
Note that this may take up to a few days to run.
After it, run `python main.py compare-splitters -s` to extract to CSV files the estimates from the metadata created through the previous command.
Finally, `python main.py compare-splitters -a` generates the box plots of the performances. 
