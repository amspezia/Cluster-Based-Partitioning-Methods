from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from kfoldmethods.experiments.statistical_tests import count_winners
from kfoldmethods.experiments import configs


path_rt = Path(configs.compare_splitters__output) / "outputs" / "summary_running_time.csv"
path_rt_artifacts = Path(configs.compare_splitters__output) / "artifacts" /"running_time"


def get_df_rt_10folds():
    df_rt = pd.read_csv(path_rt, header=[0, 1], skip_blank_lines=True, index_col=0)
    if 'ShuffleSplit' in df_rt.columns:
        df_rt.drop(columns=['ShuffleSplit'], level=0, inplace=True)
    if 'StratifiedShuffleSplit' in df_rt.columns:
        df_rt.drop(columns=['StratifiedShuffleSplit'], level=0, inplace=True)
    df_rt.drop(columns=["2", "5"], level=1, inplace=True)
    df_rt = df_rt.droplevel(1, axis=1)
    
    return df_rt


def savefig(fig, basename, output_dir: Path):
    jpg_dir = output_dir / 'jpgs'
    jpg_dir.mkdir(exist_ok=True, parents=True)
    pdf_dir = output_dir / 'pdfs'
    pdf_dir.mkdir(exist_ok=True, parents=True)

    fig.savefig(jpg_dir / "{}.jpg".format(basename))
    fig.savefig(pdf_dir / "{}.pdf".format(basename))


def winners_running_time():
    df_rt = get_df_rt_10folds()
    df_rt.drop(inplace=True, columns=['KFold', 'StratifiedKFold'])
    print(df_rt)
    a = np.argmin(df_rt.to_numpy(), axis=1)
    print(np.bincount(a))
    datasets = np.array(configs.datasets)
    print(datasets[a == 1])

    df_rt.drop(inplace=True, columns=['DBSCV', 'DOBSCV'])
    print(df_rt)
    a = np.argmin(df_rt.to_numpy(), axis=1)
    print(np.bincount(a))


def cluster_based__how_many_times_faster():
    df_rt = get_df_rt_10folds()
    df_rt.drop(inplace=True, columns=['KFold', 'StratifiedKFold'])
    df_rt.drop(inplace=True, columns=['DBSCV', 'DOBSCV', 'SPECTRAL'])
    
    df_CBDSCV = df_rt.drop(columns=['SCBDSCV', 'SCBDSCV_Mini'])
    ratio_CBDSCV = df_CBDSCV['CBDSCV'] / df_CBDSCV['CBDSCV_Mini']

    print("CBDSCV vs CBDSCV_Mini running times")
    print(ratio_CBDSCV)
    print("Mean CBDSCV: {:.4f}".format(np.mean(ratio_CBDSCV)))

    df_SCBDSCV = df_rt#.drop(columns=['CBDSCV', 'CBDSCV_Mini'])
    ratio_SCBDSCV = df_SCBDSCV['SCBDSCV'] / df_SCBDSCV['SCBDSCV_Mini']

    print("SCBDSCV vs SCBDSCV_Mini running times")
    print(ratio_SCBDSCV)
    print("Mean SCBDSCV: {:.4f}".format(np.mean(ratio_SCBDSCV)))


def plot_rt_distribution():
    sns.set_theme()
    df_rt = get_df_rt_10folds()

    fig, ax = plt.subplots()
    sns.stripplot(data=df_rt, ax=ax, orient='h')
    ax.set_ylabel("Splitter")
    ax.set_xlabel("Running time (s)")
    fig.tight_layout()

    savefig(fig, "running_times", path_rt_artifacts)
    

def analyze():
    if not path_rt_artifacts.exists():
        path_rt_artifacts.mkdir(exist_ok=True, parents=True)
    cluster_based__how_many_times_faster()
    plot_rt_distribution()
    winners_running_time()