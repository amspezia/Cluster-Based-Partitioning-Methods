from itertools import combinations, product
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
from scipy import stats
from .configs import need_n_clusters

from kfoldmethods.experiments import utils
from kfoldmethods.experiments import configs

path_run = Path(configs.compare_splitters__output) / 'outputs'
path_artifacts = Path(configs.compare_splitters__output) / 'artifacts'
path_statistical_tests = path_artifacts / 'statistical_tests'
path_wins = path_artifacts / 'wins'


def test_splitter_pairs(splitters: list, biases: list, stds: list):
    n_splitters = len(splitters)
    records_b = np.zeros((n_splitters, n_splitters))
    records_std = np.zeros((n_splitters, n_splitters))

    for i, j in combinations([k for k in range(len(splitters))], 2):
        _, p_bias = stats.wilcoxon(biases[i], biases[j])
        _, p_std = stats.wilcoxon(stds[i], stds[j])
        
        records_b[i, j] = p_bias
        records_std[i, j] = p_std

    df_bias_wilc = pd.DataFrame(records_b, index=splitters, columns=splitters)
    df_std_wilc = pd.DataFrame(records_std, index=splitters, columns=splitters)

    return df_bias_wilc, df_std_wilc


def test_splitters_friedman(biases, stds):
    _, p_bias = stats.friedmanchisquare(*biases)
    _, p_std = stats.friedmanchisquare(*stds)
    
    return p_bias, p_std


def count_winners(splitters: list, samples: list):
    """Count number of times each splitter had the best result.
    Minimum absolute value is the best result.

    Args:
        splitters (list): list of splitters
        samples (list): list of samples (list of lists). First sample corresponds to first splitter, etc
    """
    samples_array = np.array(samples).transpose()
    wins = np.bincount(np.argmin(np.abs(samples_array), axis=1))
    idx = np.arange(0, len(samples))
    record = {splitters[k]: v for k, v in zip(idx, wins)}
    return record


def apply_tests_by__balance(df: pd.DataFrame, metrics, n_splits_list, ds_balances):
    records_friedman = []
    df_wins = pd.DataFrame()

    for metric, n_splits, ds_balance in product(metrics, n_splits_list, ds_balances):
        ds_list = configs.datasets_balanced if ds_balance == 'balanced' else configs.datasets_imb

        df_group = df[
            (df['metric_name'] == metric) & \
            (df['n_splits'] == n_splits) & \
            (df['dataset_name'].isin(ds_list))]

        splitters, biases, stds = utils._make_samples_for_tests(df_group)
        if len(biases) > 2:
            p_bias, p_std = test_splitters_friedman(biases, stds)
            records_friedman.append(
                {'metric': metric, 'n_splits': n_splits, 'balance': ds_balance, 'p_bias': p_bias, 'p_std': p_std})

        pairs_bias, pairs_std = test_splitter_pairs(splitters, biases, stds)
        pairs_bias.to_csv(
            path_statistical_tests / 'wc_bias_{}_{}_{}.csv'.format(metric, n_splits, ds_balance), 
            float_format="%.5f")
        pairs_std.to_csv(
            path_statistical_tests / 'wc_std_{}_{}_{}.csv'.format(metric, n_splits, ds_balance), 
            float_format="%.5f")

        wins_bias = count_winners(splitters, biases)
        wins_std = count_winners(splitters, stds)
        indexes = pd.MultiIndex.from_product([(metric, ), (n_splits, ), (ds_balance, ), ('bias', 'std')])
        df_wins_group = pd.DataFrame.from_records([wins_bias, wins_std], index=indexes)
        df_wins = pd.concat((df_wins, df_wins_group), axis=0)

    df_wins.to_csv(path_wins / 'wins_by_balance.csv')
    df_fr = pd.DataFrame.from_records(records_friedman)
    df_fr.to_csv(path_statistical_tests / 'fr_by_balance.csv', float_format="%.5f")


def apply_tests_overall(df: pd.DataFrame, metrics, n_splits_list, subdir: Union[None, Path] = None):
    if subdir is not None:
        path_out_wins = path_wins / subdir
        path_out_st_tests = path_statistical_tests / subdir

        path_out_wins.mkdir(exist_ok=True, parents=True)
        path_out_st_tests.mkdir(exist_ok=True, parents=True)
    else:
        path_out_wins = path_wins
        path_out_st_tests = path_statistical_tests
    
    records_friedman = []
    df_wins = pd.DataFrame()
    for metric, n_splits in product(metrics, n_splits_list):
        df_group = df[(df['metric_name'] == metric) & (df['n_splits'] == n_splits)]

        splitters, biases, stds = utils._make_samples_for_tests(df_group)
        if len(biases) > 2:
            p_bias, p_std = test_splitters_friedman(biases, stds)
            records_friedman.append(
                {'metric': metric, 'n_splits': n_splits, 'p_bias': p_bias, 'p_std': p_std})

        pairs_bias, pairs_std = test_splitter_pairs(splitters, biases, stds)
        pairs_bias.to_csv(
            path_out_st_tests / 'wc_bias_{}_{}_overall.csv'.format(metric, n_splits), 
            float_format="%.5f")
        pairs_std.to_csv(
            path_out_st_tests / 'wc_std_{}_{}_overall.csv'.format(metric, n_splits), 
            float_format="%.5f")

        wins_bias = count_winners(splitters, biases)
        wins_std = count_winners(splitters, stds)
        indexes = pd.MultiIndex.from_product([(metric, ), (n_splits, ), ('bias', 'std')])
        df_wins_group = pd.DataFrame.from_records([wins_bias, wins_std], index=indexes)
        df_wins = pd.concat((df_wins, df_wins_group), axis=0)

    df_wins.to_csv(path_out_wins / 'wins_overall.csv')
    df_fr = pd.DataFrame.from_records(records_friedman)
    df_fr.to_csv(path_out_st_tests / 'fr_overall.csv', float_format="%.5f")


def apply_tests_by__balance_clf(df: pd.DataFrame, metrics, n_splits_list, ds_balances, classifiers):
    for metric, n_splits, ds_balance in product(metrics, n_splits_list, ds_balances):
        df_results = pd.DataFrame()
        for clf in classifiers:
            ds_list = configs.datasets_balanced if ds_balance == 'balanced' else configs.datasets_imb

            df_group = df[
                (df['metric_name'] == metric) & \
                (df['n_splits'] == n_splits) & \
                (df['classifier_name'] == clf) & \
                (df['dataset_name'].isin(ds_list))]

            splitters, biases, stds = utils._make_samples_for_tests(df_group)
            wb = count_winners(splitters, biases)
            ws = count_winners(splitters, stds)

            indexes = pd.MultiIndex.from_product([[clf], ['bias', 'std']])
            df_w = pd.DataFrame([wb, ws], index=indexes)
            df_results = pd.concat((df_results, df_w), axis=0)
        
        df_results.to_csv(path_artifacts / "balance_clf/{}_{}_{}.csv".format(metric, n_splits, ds_balance))


def apply_tests_by__clf(df: pd.DataFrame, metrics, n_splits_list, classifiers):
    for metric, n_splits, clf in product(metrics, n_splits_list, classifiers):
        df_group = df[
            (df['metric_name'] == metric) & \
            (df['n_splits'] == n_splits) & \
            (df['classifier_name'] == clf)]

        splitters, biases, stds = utils._make_samples_for_tests(df_group)
        wb = count_winners(splitters, biases)
        ws = count_winners(splitters, stds)

        wdf = pd.DataFrame([wb, ws], index=['bias', 'std'])
        print(metric, n_splits, clf)
        print(wdf)


def cluster_based_only():
    df = pd.read_csv(path_run / 'bias_variance_tradeoff.csv')
    df = df[df['splitter_method'].str.contains('|'.join(need_n_clusters))]
    metrics = ['accuracy', 'f1']
    n_splits_list = [2, 5, 10]  

    apply_tests_overall(df, metrics=metrics, n_splits_list=n_splits_list, subdir=Path('cluster_based'))


def analyze():
    path_wins.mkdir(exist_ok=True, parents=True)
    path_statistical_tests.mkdir(exist_ok=True, parents=True)
    path_artifacts.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(path_run / 'bias_variance_tradeoff.csv')
    df = df[~df['splitter_method'].str.contains('Shuffle')]
    metrics = ['accuracy', 'f1', 'recall', 'precision', 'balanced_accuracy']
    n_splits_list = [2, 5, 10]
    ds_balances = ['balanced', 'imbalanced']

    apply_tests_by__balance(df, metrics, n_splits_list, ds_balances)
    cluster_based_only()
