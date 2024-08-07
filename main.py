import argparse

from kfoldmethods.experiments import tune_classifiers, \
    estimate_true_metrics, estimate_clustering_parameters, compare_splitters_estimates, datasets_info


def build_compare_splitters_subparser(subparsers):
    parser_experiment = subparsers.add_parser('compare-splitters', help='Run experiments.')

    parser_experiment.add_argument(
        "-a", "--analyze", action="store_true", help="Analyze tables of run results.")

    parser_experiment.add_argument(
        "-s", "--select", action="store_true", help="Select tables from run results.")
    
    parser_experiment.add_argument(
        "-n", "--name", type=str, help="Name of a specific splitter to compare.")
    
    return parser_experiment


def build_hyperparameters_search_subparsers(subparsers):
    parser_hp_search = subparsers.add_parser('hp-search', help='Analyze results of experiments.')
    parser_hp_search.add_argument(
        "-s", "--select", action="store_true", help="Select hyperparameters for each dataset and classifier.")
    return parser_hp_search


def build_estimate_true_metrics_subparsers(subparsers):
    parser_true_estimate = subparsers.add_parser('true-estimate', help='Estimate true metrics for each dataset and classifier.')
    parser_true_estimate.add_argument("-a", "--analyze", action="store_true", help="Analyze results of run")

    parser_true_estimate.add_argument(
        "-s", "--select-metric-results", action="store_true", 
        help="Generate csv files containing only the metrics from the joblib files.")
    parser_true_estimate.add_argument(
        "-p", "--path-input", type=str, help="Path to true-estimate run raw artifacts")
    return parser_true_estimate


def build_estimate_clustering_parameters_subparsers(subparsers):
    parser_estimate_clustering_parameters = subparsers.add_parser('estimate-clustering-parameters', help='Estimate clustering parameters in each dataset.')
    parser_estimate_clustering_parameters.add_argument("-a", "--analyze", action="store_true", help="Analyze results of run")

    return parser_estimate_clustering_parameters


def build_datasets_info_subparsers(subparsers):
    parser_ds_info = subparsers.add_parser('datasets-info', help='Generate tables with information about the datasets.')

    return parser_ds_info


def main():
    parser = argparse.ArgumentParser("k-Fold Partitioning Methods")
    subparsers = parser.add_subparsers(dest='subparser_name')
    parser_experiment = build_compare_splitters_subparser(subparsers)
    parser_hp = build_hyperparameters_search_subparsers(subparsers)
    parser_true_estimate = build_estimate_true_metrics_subparsers(subparsers)
    parser_estimate_clustering_parameters = build_estimate_clustering_parameters_subparsers(subparsers)
    parser_ds_info = build_datasets_info_subparsers(subparsers)

    args = parser.parse_args()

    if args.subparser_name == "compare-splitters":
        compare_splitters_estimates.main(args)
        return

    if args.subparser_name == "hp-search":
        tune_classifiers.main(args)
        return

    if args.subparser_name == "true-estimate":
        estimate_true_metrics.main(args)
        return

    if args.subparser_name == "estimate-clustering-parameters":
        estimate_clustering_parameters.main(args)
        return

    if args.subparser_name == "datasets-info":
        datasets_info.main(args)
        return
    
    

if __name__ == '__main__':
    main()
