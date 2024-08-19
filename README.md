# Cluster-Based-Partitioning-Methods

## Using the Splitters
The code for the splitters are in `kfoldmethods/splitters` and the can be used in the same way as one would use sklearn splitters.

The available splitters are ACBCV, CBCV, CBCV Mini, DBSCV, DBSCANBCV, DOBSCV, ROCBCV, SCBCV, SCBCV Mini, SPECTRAL, and SVC.

The cluster-based splitters are:

* **CBCV**: Cluster-Based Cross-Validation, using K-Means.
* **CBCV Mini**: Cluster-Based Cross-Validation, using Mini Batch K-Means.
* **DBSCANCV**: DBSCAN-Based Cross-Validation, using DBSCAN.
* **ROCBCV**: Robust-Oriented Cluster-Based Cross-Validation, using K-Means with each cluster attributed to a fold.
* **SCBCV**: Stratified Cluster-Based Cross Validation, using intra-class K-Means.
* **SCBCV Min**i: Stratified Cluster-Based Cross-Validation, using intra-class Mini Batch K-Means.
* **SPECTRAL**: SPECTRAL-Based Cross-Validation, using SPECTRAL.

Simply copy the folder to your project to use them.

## Selected Hyperparameters 
The selected hyperparameters can be seen in the file `appendix/summary hyperparameters.csv`. When not indicated, the default value in sklearn was used.

## Reproducing the Experiments
#### Windows:

Run the following commands in order:

``````
python main.py hp-search
python main.py hp-search -s
python main.py estimate-clustering-parameters
python main.py estimate-clustering-parameters -a
python main.py true-estimate
python main.py true-estimate -s
python main.py true-estimate -a
python main.py compare-splitters
python main.py compare-splitters -s
python main.py compare-splitters -a
``````
 

#### Linux 

The simplest way to re-execute the experiments is to run `source run_all.sh`, if you are using GNU/Linux.
To guarantee that you are using the same python libraries that we used, consider creating a new conda environment using the env file we provide. To do this, run `conda env create -f environment_file.yml`.
Running `source run_all.sh` will create and activate the env automatically if conda is installed. 


Finally, it's also possible to run each part of the experiments separately, following the instructions in the sections below.

### Learning Algorithms Hyperparameter Tuning
We tune each classifier using the full dataset prior to the main experiments.
The tuning only serves to guarantee that the classifiers have appropriate hyperparameters for each dataset.
To reproduce our results on the hyperparameters search, run `python main.py hp-search` followed by `python main.py hp-search -s` to obtain a summary of the hyperparameters selected for each dataset-classifier pair (the parameters that are not specified in the file were set to the default values of sklearn).

The outputs are stored in the folder `classifier_hyperparameters`.

### Cluster-Based Algorithms Hyperparameter Tuning
The hyperparameters in K-Means and DBSCAN in each dataset is computed prior to the main experiments and used as input to the cluster-based splitting strategies.
To reproduce these experiments, run

`python main.py estimate-clustering-parameters`, followed by

`python main.py estimate-clustering-parameters -a` to obtain a CSV file with the number of clusters in each dataset. 

The outputs are stored in the folder `estimate_clustering_parameters`.

### Estimates of the True Performance
This step requires that the instructions from *Hyperparameter Tuning* have been performed first.
We use the hyperparameters to find estimates of the true performance of each classifier in each dataset.
To reproduce our results, run `python main.py true-estimate`, followed by `python main.py true-estimate -s` to retrieve CSV tables with the performance estimates for each dataset, classifier, and iteration.
Finally, `python main.py true-estimate -a` produces some CSV files with summary results as well as some figures omitted from the paper because of space. 

### Splitters Estimates of the Performance
This step requires that the previous steps have been performed first.

By default, it only executes the splitters of the following experiments:

**1st Set:** SCBCV, SCBCV2, SCBCV3, SCBCV4, and SCBCV5. 

- **Objective**: Define how the n_clusters parameter used on the intra-class K-Means on SCBCV affects the algorithm.

**2nd Set:** SVC, SCBCV, and SCBCV Mini

- **Objective**: Define whether Stratified Cross Validation or SCBCV and its Mini version performs best in general scenarios of different datasets and number of folds.

**3rd Set:** ACBCV, CBCV, CBCV Mini, DBSCANBCV, SCBCV, and SCBCV Mini.

- **Objective**: Define whether a Cluster-Based Cross Validation method overperforms the others in general mixed scenarios.

To reproduce these experiments, run `python main.py compare-splitters`.
Note that this may take up to a few hours to run.
After it, run `python main.py compare-splitters -s` to extract to CSV files the estimates from the metadata created through the previous command.
Finally, `python main.py compare-splitters -a` generates the box plots of the performances. 

The Splitter methods that are not used on these experiments are commented and can be tested by editing the code in `configs.py`.
