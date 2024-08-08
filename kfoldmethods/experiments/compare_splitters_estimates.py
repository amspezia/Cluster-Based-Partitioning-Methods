from datetime import datetime
from itertools import combinations, product
from pathlib import Path
import joblib
import os
import numpy as np
import pandas as pd
from pmlb import fetch_data
import time
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

from kfoldmethods.experiments import analyze_running_times, configs, statistical_tests
from kfoldmethods.experiments import utils
from kfoldmethods.experiments.utils import _compare_plot_overall, _make_samples_for_tests, bootstrap_step, estimate_n_clusters, load_best_classifier_for_dataset
from kfoldmethods.experiments import statistical_tests, analyze_running_times


class CompareSplittersEstimatesResults:
    def __init__(self):
        # this design could be normalized, but maybe this won't improve usability in python
        self.records_splits = []
        self.records_classifiers = []
        self.records_metrics = []
        self.records_splitters_running_time = []

    def insert_dataset_split(
            self, ds_name, clf_name, splitter_method, repeat_id, n_splits, split_id, indices_r, 
            train, test):
        # train and test are indices wrt to the repetition indices.
        # to recover the original instance of train instance 0 one would do:
        # X[repetition_indices[train[0]], :]
        self.records_splits.append({
            'dataset_name': ds_name,
            'classifier_name': clf_name,
            'splitter_method': splitter_method,
            'repeat_id': repeat_id,
            'n_splits': n_splits,
            'split_id': split_id,
            'repetition_indices': indices_r,
            'train': train,
            'test': test
        })

    def insert_splitter_running_time(
            self, ds_name, clf_name, splitter_method, repeat_id, n_splits, splitter_object, running_time):
        self.records_splitters_running_time.append({
            'dataset_name': ds_name,
            'classifier_name': clf_name,
            'splitter_method': splitter_method,
            'repeat_id': repeat_id,
            'n_splits': n_splits,
            'splitter_object': splitter_object,
            'running_time': running_time
        })

    def insert_classifier(self, ds_name, clf_name, splitter_method, repeat_id, split_id, classifier_object):
        # storing the classifier will make the disk full
        pass
        # self.records_classifiers.append({
        #     'dataset_name': ds_name,
        #     'classifier_name': clf_name,
        #     'splitter_method': splitter_method,
        #     'repeat_id': repeat_id,
        #     'split_id': split_id,
        #     'classifier_object': classifier_object
        # })

    def insert_metric_result(
            self, ds_name, clf_name, splitter_method, repeat_id, n_splits, split_id, metric_name, metric_value):
        self.records_metrics.append({
            'dataset_name': ds_name,
            'classifier_name': clf_name,
            'splitter_method': splitter_method,
            'repeat_id': repeat_id,
            'n_splits': n_splits,
            'split_id': split_id,
            'metric_name': metric_name,
            'metric_value': metric_value
        })

    def insert_metric_results(self, ds_name, clf_name, splitter_method, repeat_id, n_splits, split_id, metrics_name_value):
        for metric_name, metric_value in metrics_name_value:
            self.insert_metric_result(
                ds_name, clf_name, splitter_method, repeat_id, n_splits, split_id, metric_name, metric_value)

    def select_metric_results(self):
        df = pd.DataFrame.from_records(self.records_metrics)
        return df

    def select_running_time_results(self, return_splitter_obj: bool = False):
        df = pd.DataFrame.from_records(self.records_splitters_running_time)
        if not return_splitter_obj:
            df = df.drop(columns=['splitter_object'])
        return df


class CompareSplittersEstimates:
    def __init__(self, output_dir=None, ds_idx_0=None, ds_idx_last=None, splitter=''):
        self.results = CompareSplittersEstimatesResults()
        self.ds_idx_0 = ds_idx_0 if ds_idx_0 is not None else 0
        self.ds_idx_last = ds_idx_last if ds_idx_last is not None else len(configs.datasets)-1
        self.splitter = splitter

        if output_dir is None:
            self.path_results = Path(output_dir) / \
                Path('compare_splitters_estimates') / \
                datetime.now().isoformat(timespec='seconds') / \
                'results_{}_to_{}.joblib'.format(self.ds_idx_0, self.ds_idx_last)
        else:
            self.path_results = Path(output_dir) / \
                'results_{}_to_{}.joblib'.format(self.ds_idx_0, self.ds_idx_last)

    def _compute_metrics(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        metric_results = [
            ('accuracy', accuracy), ('precision', precision), 
            ('recall', recall), ('f1', f1), ('balanced_accuracy', balanced_accuracy)
        ]
        return metric_results

    def _compare_splitters(self, ds_name, clf_name):
        X, y = fetch_data(ds_name, return_X_y=True)
        df_clustering_parameters = pd.read_csv(configs.compare_splitters__path_clustering_parameters)

        for splitter_name, splitter_class, splitter_params in configs.splitter_methods:
            if self.splitter and splitter_name != self.splitter:
                continue
            
            print(f"{os.getpid()}: -- Running {splitter_name}")

            repeat_splitter = StratifiedShuffleSplit(
                n_splits=configs.compare_splitters__n_repeats, 
                test_size=configs.compare_splitters__repeat_test_size,
                random_state=configs.compare_splitters__repeats_random_state)
            
            for repeat_id, (indices_r, _) in enumerate(repeat_splitter.split(X, y)):
                try:
                    print(f"{os.getpid()}: ---- Repeat [{repeat_id + 1}/{configs.compare_splitters__n_repeats}]")
                    X_r, y_r = X[indices_r, :], y[indices_r]

                    for n_splits in configs.compare_splitters__n_splits:
                        splitter_params['n_splits'] = n_splits

                        if splitter_name in configs.need_n_clusters:
                            n_clusters = round(df_clustering_parameters.loc[
                                df_clustering_parameters['ds_name'] == ds_name, 'n_clusters_estimate'].values[0])
                            print(f"{os.getpid()}: ---- Using {n_clusters} clusters")

                            splitter_params['n_clusters'] = n_clusters
                        
                        if 'DBSCAN' in splitter_name:
                            eps = df_clustering_parameters.loc[df_clustering_parameters['ds_name'] == ds_name, 'eps'].values[0]
                            min_samples = df_clustering_parameters.loc[df_clustering_parameters['ds_name'] == ds_name, 'min_samples'].values[0]
                            print(f"{os.getpid()}: ---- Using eps={eps} min_samples={min_samples}")

                            splitter_params['eps'] = eps
                            splitter_params['min_samples'] = min_samples

                        split_start = time.perf_counter()
                        splitter = splitter_class(**splitter_params)
                        splits = [(split_id, train, test) \
                                for split_id, (train, test) in enumerate(splitter.split(X_r, y_r))]
                                
                        split_execution_time = time.perf_counter() - split_start

                        self.results.insert_splitter_running_time(
                            ds_name, clf_name, splitter_name, repeat_id, n_splits, splitter, split_execution_time)

                        for split_id, train, test in splits:
                            clf = load_best_classifier_for_dataset(ds_name, clf_name)
                            clf.fit(X_r[train], y_r[train])
                            y_pred = clf.predict(X_r[test])

                            metric_results = self._compute_metrics(y_r[test], y_pred)

                            self.results.insert_dataset_split(
                                ds_name, clf_name, splitter_name, repeat_id, n_splits, split_id, indices_r, train, test)
                            # self.results.insert_classifier(
                            #     ds_name, clf_name, splitter_name, repeat_id, split_id, clf)
                            self.results.insert_metric_results(
                                ds_name, clf_name, splitter_name, repeat_id, n_splits, split_id, metric_results)
                except Exception as e:
                    exception_str = f"Exception while executing {splitter_name} on {ds_name} with {clf_name}: {e}\n"
                    print(exception_str)
                    with open('exceptions.txt', 'a') as file:
                        file.write(exception_str)

    def compare_splitters_estimates(self):
        for ds_idx, ds_name in enumerate(configs.datasets):
            if self.ds_idx_0 <= ds_idx <= self.ds_idx_last:
                for params in configs.pipeline_params:
                    clf_class_name = params['clf'][0].__class__.__name__
                    print(f"{os.getpid()}: Estimating metrics for {ds_name} with {clf_class_name}.")

                    self._compare_splitters(ds_name, clf_class_name)

                joblib.dump(self.results, self.path_results)


def run_compare_splitters_estimates(output_dir, idx_first, idx_last, splitter):
    print(f"{os.getpid()}: Running datasets {idx_first} to {idx_last}")
    CompareSplittersEstimates(
        output_dir=output_dir, ds_idx_0=idx_first, ds_idx_last=idx_last, splitter=splitter).compare_splitters_estimates()
    print(f"{os.getpid()}: Finished datasets {idx_first} to {idx_last}")


def analyze(args):
    path_run = Path(configs.compare_splitters__output) / 'outputs'
    path_true_estimates_summary = Path(configs.true_estimates__output_summary)
    
    analyze_running_time = True
    analyze_metrics = True
    make_plots = True
    make_tests = True

    # running time
    if analyze_running_time:
        df_rt = pd.read_csv(path_run / 'running_time_df.csv')
        summary_rt = df_rt.groupby(by=['dataset_name', 'splitter_method','n_splits']).agg(
            running_time=('running_time', 'mean')).reset_index()

        summary_rt = summary_rt.pivot(
            index='dataset_name', columns=['splitter_method', 'n_splits'], values='running_time')
        summary_rt = summary_rt.sort_index(key=lambda ind: ind.str.lower())
        summary_rt.to_csv(path_run / 'summary_running_time.csv', float_format='%.5f')

        analyze_running_times.analyze()

    if analyze_metrics:
        df_m = pd.read_csv(path_run / 'metrics_df.csv')
        df_estimates = df_m.groupby(
            by=['dataset_name', 'classifier_name', 'splitter_method', 'repeat_id', 'n_splits', 'metric_name']).agg(
                estimate=('metric_value', 'mean'))
        df_estimates.to_csv(path_run / 'splitters_estimate.csv', float_format='%.5f')

        df_estimates_summary = df_estimates.groupby(
            by=['dataset_name', 'classifier_name', 'splitter_method', 'n_splits', 'metric_name']).agg(
                expected_estimate=('estimate', 'mean'), estimate_std=('estimate', 'std'))
        df_estimates_summary.to_csv(path_run / 'splitters_estimate_summary.csv', float_format='%.5f')

        df_estimates_summary.reset_index(inplace=True)
        df_true_estimates_summary = pd.read_csv(path_true_estimates_summary).rename(columns={'ds_name': 'dataset_name'})

        df_bias_std_summary = pd.merge(
            df_estimates_summary, df_true_estimates_summary, 
            on=['dataset_name', 'classifier_name', 'metric_name'], how='inner')
        df_bias_std_summary['bias'] = df_bias_std_summary['expected_estimate'] - df_bias_std_summary['true_value']
        df_bias_std_summary.to_csv(path_run / 'bias_variance_tradeoff.csv', float_format='%.5f')
    
    if make_plots:
        sns.set_theme()
        df_base = pd.read_csv(path_run / 'bias_variance_tradeoff.csv')
        df_base = df_base[~df_base['splitter_method'].str.contains('Shuffle')]
    
        for experiment, datasets in configs.experiments.items():
            df = df_base[df_base['splitter_method'].isin(datasets)]

            output_dir = path_run / f'plots/{experiment}'
            output_dir.mkdir(exist_ok=True, parents=True)

            utils._compare_plot_balance(df, output_dir)

    if make_tests:
        statistical_tests.analyze()

def select_df_results(args):
    path_run = Path(configs.compare_splitters__output)
    path_outputs = path_run / 'outputs'
    path_outputs.mkdir(exist_ok=True, parents=True)

    running_time_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    for run_file in path_run.glob("*.joblib"):
        print("Append data from file {}".format(run_file))
        run = joblib.load(run_file)
        run_running_time_df = run.select_running_time_results()
        run_metrics_df = run.select_metric_results()

        running_time_df = pd.concat((running_time_df, run_running_time_df), axis=0)
        metrics_df = pd.concat((metrics_df, run_metrics_df), axis=0)

    metrics_df.to_csv(str(path_outputs / 'metrics_df.csv'), float_format='%.4f')
    running_time_df.to_csv(str(path_outputs / 'running_time_df.csv'), float_format='%.4f')


def main(args):
    # TODO: refactor a bit the args
    if args.analyze:
        analyze(args)
        return

    if args.select:
        select_df_results(args)
        return

    if args.name:
        splitter = args.name
    else:
        splitter = ''

    output_dir = Path(configs.compare_splitters__output)
    output_dir.mkdir(exist_ok=True, parents=True)
    n_datasets = len(configs.datasets)
    step = 1

    joblib.Parallel(n_jobs=configs.compare_splitters__n_jobs)(
        joblib.delayed(run_compare_splitters_estimates)(
            output_dir, i, min(i+step-1, n_datasets-1), splitter) for i in range(0, n_datasets, step)
    )
