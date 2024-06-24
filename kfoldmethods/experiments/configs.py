from decimal import ROUND_HALF_DOWN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from kfoldmethods.datasets.pmlb_api import pmlb_get_ds_list
from kfoldmethods.splitters import CBDSCV, SCBDSCV, DBSVC, DOBSCV, SPECTRAL, ACBCV


run_data_dir = 'results_bracis22'
pipeline = Pipeline([('scaler', MinMaxScaler()), ('clf', LogisticRegression())])
pipeline_params = [
    {'clf': [LogisticRegression(max_iter=10010, random_state=0, class_weight='balanced')], 
    'clf__C': [0.003, 0.03, 0.3, 3.0, 30.0]},

    {'clf': [SVC(kernel='rbf', max_iter=10010, random_state=0, class_weight='balanced')], 
        'clf__C': [0.3, 3.0, 30.0, 300.0],
        'clf__gamma': [0.00003, 0.0003, 0.003, 0.03, 0.3]},
        
    {'clf': [RandomForestClassifier(random_state=0, class_weight='balanced')], 
        'clf__max_depth': [1, 5, 10, 15, 50]},

    {'clf': [DecisionTreeClassifier(random_state=0, class_weight='balanced')], 
        'clf__max_depth': [1, 5, 10, 15, 50]}
]

n_jobs = 4
tuning_folds = 10
tuning_grid_seach_n_jobs = n_jobs
tuning_grid_search_scoring = 'balanced_accuracy'
classifier_hyperparameters_output = "%s/classifier_hyperparameters" % run_data_dir

datasets = [
    'analcatdata_germangss', 'chess', 'analcatdata_happiness', 'analcatdata_japansolvent', 'vote', 'colic', 'dna',
    'vowel', 'movement_libras', 'analcatdata_dmft', 'allrep', 'appendicitis', 'page_blocks', 
    'new_thyroid', 'backache', 'flare', 'postoperative_patient_data',
    'hepatitis', 'analcatdata_cyyoung8092', 'car']

datasets_balanced = [
    'analcatdata_germangss', 'chess',  'analcatdata_happiness', 'analcatdata_japansolvent', 'vote', 'colic', 'dna',
    'vowel', 'movement_libras', 'analcatdata_dmft']

datasets_imb = [
    'allrep', 'appendicitis', 'page_blocks', 'new_thyroid', 'backache', 'flare', 'postoperative_patient_data',
    'hepatitis', 'analcatdata_cyyoung8092', 'car']

dataset_info__output_dir = '%s/dataset_info' % run_data_dir
dataset_info__pmlb_list_path = "kfoldmethods/datasets/pmlb_datasets.csv"

true_estimates_n_splits = 100
true_estimates_test_size = 0.1
true_estimates_n_jobs = n_jobs
true_estimates_random_state = 123
true_estimates__output = "%s/true_estimate" % run_data_dir
true_estimates__output_summary = "%s/true_estimate/analysis/true_estimates_summary.csv" % run_data_dir


estimate_n_clusters_n_iters = 50
estimate_n_clusters_random_state = 123
estimate_n_clusters_n_jobs = 5
estimate_n_clusters__output = "%s/n_clusters_estimate" % run_data_dir

compare_splitters__n_repeats = 20
compare_splitters__repeat_test_size = 0.1
compare_splitters__repeats_random_state = 456
compare_splitters__n_splits = [2, 5, 10]
compare_splitters__n_jobs = n_jobs
compare_splitters__path_n_clusters = "%s/analysis/estimate_n_clusters.csv" % estimate_n_clusters__output
compare_splitters__output = "%s/compare_splitters_estimates" % run_data_dir


splitter_methods = [
    ('DBSCV', DBSVC.DBSCVSplitter, {
        'shuffle': True, 'bad_case': False, 'random_state': 123}),
    ('DOBSCV', DOBSCV.DOBSCVSplitter, {
        'shuffle': True, 'bad_case': False, 'random_state': 123}),
    ('CBDSCV', CBDSCV.CBDSCVSplitter, {
        'shuffle': True, 'random_state': 123, 'minibatch_kmeans': False}),
    ('CBDSCV_Mini', CBDSCV.CBDSCVSplitter, {
        'shuffle': True, 'random_state': 123, 'minibatch_kmeans': True}),
    ('SCBDSCV', SCBDSCV.SCBDSCVSplitter, {
        'shuffle': True, 'random_state': 123, 'minibatch_kmeans': False}),
    ('SCBDSCV_Mini', SCBDSCV.SCBDSCVSplitter, {
        'shuffle': True, 'random_state': 123, 'minibatch_kmeans': True}),
    ('SPECTRAL_rbf', SPECTRAL.SPECTRALSplitter, {
        'shuffle': True, 'affinity': 'rbf', 'random_state': 123}),
    ('SPECTRAL_nn', SPECTRAL.SPECTRALSplitter, {
        'shuffle': True, 'affinity': 'nearest_neighbors', 'random_state': 123}),
    ('ACBCV', ACBCV.ACBCVSplitter, {
        'shuffle': True, 'random_state': 123})
    ('StratifiedKFold', StratifiedKFold, {
        'shuffle': True, 'random_state': 123}),
    ('KFold', KFold, {
        'shuffle': True, 'random_state': 123})
]

need_n_clusters = ['CBDSCV', 'CBDSCV_Mini', 'SCBDSCV', 'SCBDSCV_Mini', 'SPECTRAL_rbf', 'SPECTRAL_nn', 'ACBCV']