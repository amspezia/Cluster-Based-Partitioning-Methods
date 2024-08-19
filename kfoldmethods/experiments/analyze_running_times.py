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


def get_df_rt_10folds(dataset_names=None):
    df_rt = pd.read_csv(path_rt, header=[0, 1], skip_blank_lines=True, index_col=0)
    if 'ShuffleSplit' in df_rt.columns:
        df_rt.drop(columns=['ShuffleSplit'], level=0, inplace=True)
    if 'StratifiedShuffleSplit' in df_rt.columns:
        df_rt.drop(columns=['StratifiedShuffleSplit'], level=0, inplace=True)
    df_rt.drop(columns=["2"], level=1, inplace=True)
    df_rt = df_rt.droplevel(1, axis=1)
    
    # Filter by dataset names if provided
    if dataset_names:
        df_rt = df_rt[dataset_names]
        
    return df_rt


def savefig(fig, basename, output_dir: Path):
    jpg_dir = output_dir / 'jpgs'
    jpg_dir.mkdir(exist_ok=True, parents=True)
    pdf_dir = output_dir / 'pdfs'
    pdf_dir.mkdir(exist_ok=True, parents=True)

    fig.savefig(jpg_dir / "{}.jpg".format(basename))
    fig.savefig(pdf_dir / "{}.pdf".format(basename))




def plot_rt_distribution():
    sns.set_theme()
    
    for experiment, datasets in configs.experiments.items():
        df_rt = get_df_rt_10folds(datasets)

        fig, ax = plt.subplots()
        sns.stripplot(data=df_rt, ax=ax, orient='h')
        ax.set_ylabel("Splitter")
        ax.set_xlabel("Running time (s)")
        fig.tight_layout()
        
        savefig(fig, f"running_times_{experiment}", path_rt_artifacts)
    

def analyze():
    if not path_rt_artifacts.exists():
        path_rt_artifacts.mkdir(exist_ok=True, parents=True)
    plot_rt_distribution()