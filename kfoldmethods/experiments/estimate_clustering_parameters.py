from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pmlb import fetch_data
import time
import matplotlib.pyplot as plt

from kfoldmethods.experiments import configs
from kfoldmethods.experiments.utils import estimate_n_clusters 
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def find_suitable_eps(data, n_neighbors=4, ds_name=''):
    # Step 1: Calculate distances
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    # Step 2: Find minimum distance to the nearest 3 neighbors for each point
    min_distances = np.sort(distances[:, n_neighbors - 1])
    
    # Step 3: Sort distances ascending and plot to find each value
    # Step 4: Use the KneeLocator to detect the elbow point
    kneedle = KneeLocator(range(len(min_distances)), min_distances, S=1.0, curve='convex', direction='increasing')
    
    suitable_eps = min_distances[kneedle.elbow] if kneedle.elbow is not None else min_distances[-1]
    
    # Plot the results
    plt.plot(min_distances)
    if kneedle.elbow is not None:
        plt.scatter(kneedle.elbow, suitable_eps, color='red')  # Mark the elbow point
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Distance to nearest neighbor')
    plt.title(f'Elbow Method for Finding Suitable Eps {ds_name}')
    plt.show()
    
    return suitable_eps

class ClusteringParametersEstimateResults:
    def __init__(self):
        self.records = []

    def insert_estimate(self, ds_name, iter, sample_size, execution_time, n_clusters, eps, min_samples):
        self.records.append({
            'ds_name': ds_name,
            'iter': iter,
            'sample_size': sample_size,
            'execution_time': execution_time,
            'n_clusters': n_clusters,
            'eps': eps,
            'min_samples': min_samples
        })
    
    def select_estimate_results(self):
        results = pd.DataFrame.from_records(self.records)
        return results


class ClusteringParametersEstimate:
    def __init__(self, output_dir=None, ds_idx_0=None, ds_idx_last=None):
        self.results = ClusteringParametersEstimateResults()
        self.ds_idx_0 = ds_idx_0 if ds_idx_0 is not None else 0
        self.ds_idx_last = ds_idx_last if ds_idx_last is not None else len(configs.datasets)-1

        if output_dir is None:
            self.path_results = Path(output_dir) / \
                Path('true_estimate') / \
                datetime.now().isoformat(timespec='seconds') / \
                'results_{}_to_{}.joblib'.format(self.ds_idx_0, self.ds_idx_last)
        else:
            self.path_results = Path(output_dir) / \
                'results_{}_to_{}.joblib'.format(self.ds_idx_0, self.ds_idx_last)

    def estimate_clustering_parameters(self):
        for ds_idx, ds_name in enumerate(configs.datasets):
            if self.ds_idx_0 <= ds_idx <= self.ds_idx_last:
                print("Estimating number of clusters for dataset {}".format(ds_name))
                self.estimate_clustering_parameters_dataset(ds_name)

                joblib.dump(self.results, self.path_results)

    def estimate_clustering_parameters_dataset(self, ds_name):
        X, y = fetch_data(ds_name, return_X_y=True)
        
        min_samples = X.shape[1] * 2 
        eps = find_suitable_eps(X, min_samples - 1, ds_name)
        print(f"Estimated parameters of DBSCAN for dataset {ds_name}: eps={eps} min_samples={min_samples}")

        sample_size = min(100, X.shape[0] - 1)
        n_iters = configs.estimate_clustering_parameters_n_iters
        
        start = time.perf_counter()
        n_clusters_list = estimate_n_clusters(
            X, n_iters=n_iters, sample_size=sample_size, return_all=True, 
            random_state=configs.estimate_clustering_parameters_random_state)
        execution_time = time.perf_counter() - start
        
        for iter, n_clusters in enumerate(n_clusters_list):
            self.results.insert_estimate(
                ds_name, iter, sample_size, execution_time / len(n_clusters_list), n_clusters, eps, min_samples)
        

def run_clustering_parameters_estimate(output_dir, idx_first, idx_last):
    print("Running datasets %d to %d" % (idx_first, idx_last))
    ClusteringParametersEstimate(
        output_dir=output_dir, ds_idx_0=idx_first, ds_idx_last=idx_last).estimate_clustering_parameters()
    print("Finished datasets %d to %d" % (idx_first, idx_last))


def analyze(args):
    results_df = pd.DataFrame()
    path_results = Path(configs.estimate_clustering_parameters__output)
    
    path_outputs = path_results / 'analysis'
    path_outputs.mkdir(exist_ok=True, parents=True)

    for path_run in path_results.glob("*.joblib"):
        run_results = joblib.load(path_run)
        run_results_df = run_results.select_estimate_results()
        results_df = pd.concat((results_df, run_results_df), axis=0)
    
    summary = results_df.groupby(by=['ds_name']).agg(
        sample_size=('sample_size', lambda x: np.unique(x)[0]),
        n_iters=('iter', lambda x: np.max(x) + 1), 
        n_clusters_estimate=('n_clusters', 'mean'),
        n_clusters_std=('n_clusters', 'std'),
        execution_time=('execution_time', 'sum'),
        eps=('eps', 'mean'),
        min_samples=('min_samples', 'mean'))
    
    path_estimate_clustering_parameters_summary = path_outputs / 'estimate_clustering_parameters.csv'
    summary.to_csv(path_estimate_clustering_parameters_summary, float_format='%.4f')
    

def main(args):
    if args.analyze:
        analyze(args)
        return

    output_dir = Path(configs.estimate_clustering_parameters__output)
    output_dir.mkdir(exist_ok=True, parents=True)
    n_datasets = len(configs.datasets)
    step = 3
    joblib.Parallel(n_jobs=configs.estimate_clustering_parameters_n_jobs)(
        joblib.delayed(run_clustering_parameters_estimate)(
            output_dir, i, min(i+step-1, n_datasets-1)) for i in range(0, n_datasets, step)
    )
