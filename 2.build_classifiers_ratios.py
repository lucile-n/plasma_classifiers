'''
2.build_classifiers_ratios.py
created on March 24 2021
lucile.neyton@ucsf.edu

This script aims at building classifiers to allocate samples to one of the two
groups (sepsis vs non-sepsis) given gene expression ratios.
The final models are tested on a held-out set.

Input files (data folder):
    - CSV-formatted file containing raw
        gene counts (samples x genes)
    - CSV-formatted file containing sepsis labels
        (samples x)
    - CSV-formatted file containing
        the list of differentially expressed genes (genes x)

Output files (results folder):
    - One dump file per classifier
'''

# load libraries and functions
#
import sys
sys.path.append("/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_VALID/de_analysis")
#
from custom_classes import VstTransformer, CustomRFECV, CustomBaggingClassifier, DGEA_filter
#
from itertools import product
#
import os
# 1.19.1
import numpy as np
# 1.1.3
import pandas as pd
# 0.23.2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# 1.2.1
from xgboost import XGBClassifier
# 0.17.0
from joblib import dump, load

# set paths
data_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_plasma/data/"
paxgene_data_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_VALID/data/"
results_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_plasma/results/"

# define functions
# build ratio object
def build_ratios_df(expr_data, name_vars, name_samps):
    # convert numpy array to df
    expr_data = pd.DataFrame(expr_data, index=name_samps, columns=name_vars)

    # cols should be = ncol! / (2! * (ncol-2)!)
    ratios_df = pd.DataFrame(index=name_samps)
    for gene_ind_1 in range(0, expr_data.shape[1]):
        for gene_ind_2 in range(0, expr_data.shape[1]):
            if gene_ind_1 < gene_ind_2:
                gene_data_1 = expr_data[name_vars[gene_ind_1]]
                gene_data_2 = expr_data[name_vars[gene_ind_2]]

                gene_ratio = gene_data_1 / gene_data_2

                ratios_df[name_vars[gene_ind_1] + "/" + name_vars[gene_ind_2]] = gene_ratio

    return ratios_df

# set params
mode_ = "create"

# list parameter values
# limit to some values
min_cnts_per_sample_vals = ['50000']
min_non_zero_counts_per_genes_vals = ['20', '30', '40', '50']
fdr_thresh_vals = ['0.1']
age_sex_model_vals = ['TRUE', 'FALSE']

comb_list = list(product(min_cnts_per_sample_vals, min_non_zero_counts_per_genes_vals,
                         fdr_thresh_vals, age_sex_model_vals))

for comb_ in comb_list:
    results_prefix = comb_[0] + "_" + comb_[1] + "_" + comb_[2] + "_" + comb_[3]

    print("=====")
    print(results_prefix)
    print("=====")

    #########################
    # DATA LOADING
    #########################
    cnt_data = pd.read_csv(data_path + "processed/" + results_prefix + "_cnts.csv", index_col=0)
    meta_data = pd.read_csv(data_path + "processed/" + results_prefix + "_metadata_1vs4.csv", index_col=0)
    dgea_results = pd.read_csv(results_path + results_prefix + "_DGEA_results.csv", index_col=0)

    #########################
    # DATA PREPROCESSING
    #########################
    # keep only DE genes
    cnt_data = cnt_data.loc[cnt_data.hgnc_symbol.isin(dgea_results.hgnc_symbol.values), :]

    cnt_data.index = cnt_data.hgnc_symbol
    cnt_data = cnt_data.loc[dgea_results.hgnc_symbol.values, :]
    cnt_data = cnt_data.drop("hgnc_symbol", axis=1)

    # drop duplicate gene symbols
    cnt_data = cnt_data.loc[~cnt_data.index.duplicated(), :]

    # filter samples not in the meta_data file and make sure vst_data and meta_data have the same rows order
    cnt_data = cnt_data.T
    cnt_data = cnt_data.reindex(meta_data.SampleID.values).dropna(axis=0, how='any')

    # split data into train and test sets
    cnt_data_train, cnt_data_test, sepsis_cat_train, sepsis_cat_test = train_test_split(cnt_data, meta_data.sepsis_cat,
                                                                                        test_size=0.3, random_state=123,
                                                                                        stratify=meta_data.sepsis_cat)

    # store variable labels
    name_vars = cnt_data_train.columns

    # store sample ids
    name_samps_train = cnt_data_train.index
    name_samps_test = cnt_data_test.index

    # apply vst to counts
    vst_transformer = VstTransformer()
    vst_transformer = vst_transformer.fit(cnt_data_train)
    vst_data_train = vst_transformer.transform(cnt_data_train)
    vst_data_test = vst_transformer.transform(cnt_data_test)

    # build ratio objects
    vst_data_ratios_train = build_ratios_df(vst_data_train, name_vars, name_samps_train)
    vst_data_ratios_test = build_ratios_df(vst_data_test, name_vars, name_samps_test)

    #########################
    # XGBOOST - CV within a CV for grid search and RFE as part of a pipeline
    # n_jobs should be set to 1/None (or at least not -1 in both) to avoid nested parallelism
    #########################
    rfecv = CustomRFECV(estimator=XGBClassifier(random_state=123, nthread=1), cv=5, n_jobs=1, scoring='roc_auc',
                        step=0.1, min_features_to_select=2, verbose=True, max_features=100)

    # pipeline steps
    pipe = Pipeline([('rfe', rfecv), ('xgbc', XGBClassifier())])

    # create the parameter grid
    param_grid = {
        'xgbc__random_state': [123],
        'xgbc__nthread': [1],
        'xgbc__learning_rate': [0.01, 0.1, 0.3],  # boosting learning rate 0 to 1, default 0.3
        'xgbc__objective': ['binary:logistic'],  # learning objective
        'xgbc__max_depth': [10, 50, 100],  # max number of levels in each tree
        'xgbc__n_estimators': [100, 1000]  # number of trees
    }

    # set the grid search object
    search = GridSearchCV(estimator=pipe, cv=5, n_jobs=1, scoring='roc_auc', param_grid=param_grid, verbose=True)

    if mode_ == "create":
        # fit the chosen model
        search.fit(vst_data_ratios_train, sepsis_cat_train)
        dump(search, results_path + results_prefix + "_ratios" + "_dump_xgb_de_genes.joblib")
    else:
        if mode_ == "load":
            search = load(results_path + results_prefix + "_ratios" + "_dump_xgb_de_genes.joblib")

    print(search.best_params_)
    print(search.best_score_)
    print(vst_data_ratios_train.columns[search.best_estimator_.named_steps["rfe"].support_])

    # save list of predictors
    best_vars = vst_data_ratios_train.columns[search.best_estimator_.named_steps["rfe"].support_]
    pd.DataFrame(best_vars).to_csv(results_path + results_prefix + "_ratios" + "_best_vars_xgb_de_genes.csv", header=False,
                                   index=False)

    # evaluate on test data
    probs = search.predict_proba(vst_data_ratios_test)
    probs = probs[:, 1]
    roc_auc_xgb = roc_auc_score(sepsis_cat_test, probs)
    print(roc_auc_xgb)

    #########################
    # BagSVM - CV within a CV for grid search and RFE as part of a pipeline
    # n_jobs should be set to 1/None (or at least not -1 in both) to avoid nested parallelism
    #########################
    rfecv = CustomRFECV(estimator=CustomBaggingClassifier(base_estimator=LinearSVC(max_iter=10000), random_state=123),
                        cv=5, n_jobs=1, scoring='roc_auc',
                        step=0.1, min_features_to_select=2, verbose=True, max_features=100)

    # pipeline steps
    pipe = Pipeline([('rfe', rfecv),
                     ('bsvmc', CustomBaggingClassifier(base_estimator=LinearSVC(max_iter=10000)))])

    # create the parameter grid
    param_grid = {
        'bsvmc__random_state': [123],
        'bsvmc__base_estimator__C': [0.01, 0.1, 1, 10],  # regularisation parameter
        'bsvmc__max_features': [0.1, 0.5, 0.8],  # max number of features to draw (without replacement)
        'bsvmc__n_estimators': [100, 1000]  # number of base estimators
    }

    # set the grid search object
    search = GridSearchCV(estimator=pipe, cv=5, n_jobs=1, scoring='roc_auc', param_grid=param_grid, verbose=True)

    if mode_ == "create":
        # fit the chosen model
        search.fit(vst_data_ratios_train, sepsis_cat_train)
        dump(search, results_path + results_prefix + "_ratios" + "_dump_bsvm_de_genes.joblib")
    else:
        if mode_ == "load":
            search = load(results_path + results_prefix + "_ratios" + "_dump_bsvm_de_genes.joblib")

    print(search.best_params_)
    print(search.best_score_)
    print(vst_data_ratios_train.columns[search.best_estimator_.named_steps["rfe"].support_])

    # save list of predictors
    best_vars = vst_data_ratios_train.columns[search.best_estimator_.named_steps["rfe"].support_]
    pd.DataFrame(best_vars).to_csv(results_path + results_prefix + "_ratios" + "_best_vars_bsvm_de_genes.csv", header=False,
                                   index=False)

    # evaluate on test data
    probs = search.predict_proba(vst_data_ratios_test)
    probs = probs[:, 1]
    roc_auc_bsvm = roc_auc_score(sepsis_cat_test, probs)
    print(roc_auc_bsvm)
