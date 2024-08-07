from genericpath import exists
from pathlib import Path
import numpy as np
from sklearn import clone
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kfoldmethods.experiments import configs


def load_best_classifier_for_dataset(
        ds, clf_class_name, hp_dir=configs.classifier_hyperparameters_output):
    path_search = Path(hp_dir) / ds / clf_class_name / 'search.joblib'
    search = joblib.load(path_search)
    return clone(search.best_estimator_)


def estimate_n_clusters(X, n_iters=50, sample_size=None, random_state=123, return_all=False):
    """Estimate number of clusters based on agglomerative clustering.

    The number of clusters is estimated by retaining only merges that had a distance
    smaller than a threshold. The threshold is the mean distance plus two times the 
    standard deviation of the distances.

    This process is repeated multiple times with small sample sizes.
    """
    if sample_size is None:
        sample_size = min(100, X.shape[0] - 1)
    
    splitter = ShuffleSplit(n_splits=n_iters, train_size=sample_size, random_state=random_state)
    
    n_clusters_list = []
    for train, test in splitter.split(X):
        model = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage='average', 
            compute_full_tree=True)
        model = model.fit(X[train])
        distances = model.distances_

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # stop when distance > mean + 2*std
        th = mean_distance + 2*std_distance
        n_clusters = np.sum(1*(distances > th)) + 1
        n_clusters_list.append(n_clusters)

    if return_all:
        return n_clusters_list
    
    return np.mean(n_clusters_list), np.std(n_clusters_list)


def bootstrap_step(X: np.ndarray, random_state=123):
    n_samples = X.shape[0]
    indices = np.arange(0, n_samples)
    indices_r = resample(indices, n_samples=n_samples, random_state=random_state)
    return indices_r


def g_name_to_str(g_name):
    return "_".join([str(s) for s in g_name])


def _compare_plot_overall(df, output_dir):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for i, (g_name, g) in enumerate(groups):
        print(g_name)
        fig, ax = plt.subplots()
        sns.boxplot(data=g, y='splitter_method', x='bias', ax=ax)
        fig.savefig(output_dir / 'bias_splitter_{}'.format(g_name_to_str(g_name)))
        fig.tight_layout()
        plt.close(fig)


def _compare_plot_std_overall(df, output_dir):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for i, (g_name, g) in enumerate(groups):
        print(g_name)
        fig, ax = plt.subplots()
        sns.boxplot(data=g, y='splitter_method', x='estimate_std', ax=ax)
        fig.savefig(output_dir / 'std_splitter_{}'.format(g_name_to_str(g_name)))
        fig.tight_layout()
        plt.close(fig)


def _plot_distributions(df_group, output_dir: Path, g_name: str, measure: str, balanced: bool, colored_by: str):
    bal_str = 'balanced' if balanced else 'imbalanced'

    base_name = "{}_{}_{}".format(bal_str, measure, g_name_to_str(g_name))
    base_name = "{}_{}".format(colored_by, base_name) if colored_by is not None else base_name

    output_path_jpg = (output_dir / "jpgs") / (base_name + ".jpg")
    output_path_jpg.parent.mkdir(exist_ok=True, parents=True)
    output_path_pdf = (output_dir / "pdfs") / (base_name + ".pdf")
    output_path_pdf.parent.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    if colored_by is None:
        sns.boxplot(data=df_group, y='splitter_method', x=measure, ax=ax, whis=20.0)
    else:
        sns.stripplot(data=df_group, y='splitter_method', x=measure, ax=ax, hue=colored_by)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel('Splitter')
    fig.tight_layout()
    fig.savefig(output_path_jpg)
    fig.savefig(output_path_pdf)
    plt.close(fig)


def _compare_plot_balance(df, output_dir):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for g_name, g in groups:
        if g_name[0] not in ['accuracy', 'f1']:
            continue
        print(g_name)

        if g_name[0] == 'accuracy':
            g_balanced = g.loc[g['dataset_name'].isin(configs.datasets_balanced), :]
            _plot_distributions(g_balanced, output_dir, g_name, 'bias', balanced=True, colored_by=None)
            _plot_distributions(g_balanced, output_dir, g_name, 'bias', balanced=True, colored_by='classifier_name')
            _plot_distributions(g_balanced, output_dir, g_name, 'bias', balanced=True, colored_by='dataset_name')
            _plot_distributions(g_balanced, output_dir, g_name, 'estimate_std', balanced=True, colored_by=None)
            _plot_distributions(g_balanced, output_dir, g_name, 'estimate_std', balanced=True, colored_by='classifier_name')
            _plot_distributions(g_balanced, output_dir, g_name, 'estimate_std', balanced=True, colored_by='dataset_name')

        if g_name[0] == 'f1':
            g_imbalanced = g.loc[g['dataset_name'].isin(configs.datasets_imb), :]
            _plot_distributions(g_imbalanced, output_dir, g_name, 'bias', balanced=False, colored_by=None)
            _plot_distributions(g_imbalanced, output_dir, g_name, 'bias', balanced=False, colored_by='classifier_name')
            _plot_distributions(g_imbalanced, output_dir, g_name, 'bias', balanced=False, colored_by='dataset_name')
            _plot_distributions(g_imbalanced, output_dir, g_name, 'estimate_std', balanced=False, colored_by=None)
            _plot_distributions(g_imbalanced, output_dir, g_name, 'estimate_std', balanced=False, colored_by='classifier_name')
            _plot_distributions(g_imbalanced, output_dir, g_name, 'estimate_std', balanced=False, colored_by='dataset_name')
        
        # color: clf
        # fig, ax = plt.subplots()
        # sns.stripplot(data=g_balanced, y='splitter_method', x='bias', ax=ax, hue='classifier_name')
        # ax.set_ylabel('Splitter')
        # fig.tight_layout()
        # fig.savefig(output_dir / 'clf_balanced_bias_splitter_{}.jpg'.format(g_name_to_str(g_name)))
        # plt.close(fig)

        # # color: dataset
        # fig, ax = plt.subplots()
        # sns.stripplot(data=g_balanced, y='splitter_method', x='bias', ax=ax, hue='dataset_name')
        # ax.set_ylabel('Splitter')
        # fig.tight_layout()
        # fig.savefig(output_dir / 'ds_balanced_bias_splitter_{}.jpg'.format(g_name_to_str(g_name)))
        # plt.close(fig)

        # fig.savefig(output_dir / 'balanced_bias_splitter_{}.pdf'.format(g_name_to_str(g_name)))
        

        # fig, ax = plt.subplots()
        # g_imbalanced = g.loc[g['dataset_name'].isin(configs.datasets_imb), :]
        # sns.boxplot(data=g_imbalanced, y='splitter_method', x='bias', ax=ax, whis=20.0)
        # ax.set_ylabel('Splitter')
        # fig.tight_layout()
        # fig.savefig(output_dir / 'imbalanced_bias_splitter_{}.jpg'.format(g_name_to_str(g_name)))
        # # fig.savefig(output_dir / 'imbalanced_bias_splitter_{}.pdf'.format(g_name_to_str(g_name)))
        # plt.close(fig)



def _compare_plot_std_balance(df, output_dir):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for g_name, g in groups:
        print(g_name)
        g_balanced = g.loc[g['dataset_name'].isin(configs.datasets_balanced), :]

        fig, ax = plt.subplots()
        sns.boxplot(data=g_balanced, y='splitter_method', x='estimate_std', ax=ax, whis=20.0)
        ax.set_ylabel('Splitter')
        ax.set_xlabel('std')
        fig.tight_layout()
        fig.savefig(output_dir / 'balanced_std_splitter_{}.jpg'.format(g_name_to_str(g_name)))
        plt.close(fig)

        fig, ax = plt.subplots()
        g_imbalanced = g.loc[g['dataset_name'].isin(configs.datasets_imb), :]
        sns.boxplot(data=g_imbalanced, y='splitter_method', x='estimate_std', ax=ax, whis=20.0)
        ax.set_ylabel('Splitter')
        ax.set_xlabel('std')
        fig.tight_layout()
        fig.savefig(output_dir / 'imbalanced_std_splitter_{}.jpg'.format(g_name_to_str(g_name)))
        fig.savefig(output_dir / 'imbalanced_std_splitter_{}.pdf'.format(g_name_to_str(g_name)))
        plt.close(fig)


def _make_samples_for_tests(df):
    bias_list = []
    std_list = []
    splitters_list = []
    for splitter, df_splitter in df.groupby('splitter_method'):
        bias = df_splitter['bias']
        estimate_std = df_splitter['estimate_std']
        
        splitters_list.append(splitter)
        bias_list.append(bias)
        std_list.append(estimate_std)
    
    return splitters_list, bias_list, std_list