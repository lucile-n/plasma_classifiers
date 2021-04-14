'''
2.build_classifiers_paxgene.py
created on April 6 2021
lucile.neyton@ucsf.edu

This script aims at building classifiers to allocate samples to one of the two
groups (sepsis vs non-sepsis) given gene expression values generated from PAXgene tubes.
The final models are tested on a held-out set.

Input files (data folder):
    - CSV-formatted file containing raw PAXgene
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
from custom_classes import VstTransformer, CustomRFECV, CustomBaggingClassifier, DGEA_filter, RatiosCalculator
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

# set params
mode_ = "create"

# list parameter values
# sample filter / gene filter / DE FDR / covariates / ratios / algo
comb_list = [('50000', '30', '0.1', 'TRUE', 'FALSE', 'xgb'),
             ('50000', '20', '0.1', 'TRUE', 'FALSE', 'bsvm'),
             ('50000', '50', '0.1', 'TRUE', 'TRUE', 'xgb')]??

for comb_ in comb_list:
    results_prefix = comb_[0] + "_" + comb_[1] + "_" + comb_[2] + "_" + comb_[3]
    ratios_ = comb_[4]
    algo_ = comb_[5]

    print("=====")
    print(results_prefix)
    print("=====")

    #########################
    # DATA LOADING
    #########################
    # paxgene data
    cnt_data = pd.read_csv(data_path + "processed/" + "paxgene_cnts.csv", index_col=0)

    # plasma counts data (to use the same genes)
    plasma_cnt_data = pd.read_csv(data_path + "processed/" + results_prefix + "_cnts.csv", index_col=0)

    # group labels
    meta_data = pd.read_csv(
        paxgene_data_path + "processed/" + "EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20.csv")

    plasma_meta_data = pd.read_csv(data_path + "processed/" + results_prefix + "_metadata_1vs4.csv", index_col=0)
    dgea_results = pd.read_csv(results_path + results_prefix + "_DGEA_results.csv", index_col=0)

    #########################
    # DATA PREPROCESSING
    #########################
    # keep only samples of interest (groups 1 and 4, from the metadata file)
    meta_data["EARLI_Barcode"] = ["EARLI_" + str(x) for x in meta_data["Barcode"]]
    meta_data = meta_data.loc[(meta_data.EARLI_Barcode.isin(plasma_meta_data.SampleID)), :]
    cnt_data = cnt_data.loc[:, cnt_data.columns.isin(np.append(meta_data.HOST_PAXgene_filename, "hgnc_symbol"))]

    # format count data column and row names
    cnt_data.columns = [x.split("_")[0] + "_" + x.split("_")[1] for x in cnt_data.columns]

    # keep only pre-filtered genes
    cnt_data = cnt_data.loc[cnt_data.index.isin(plasma_cnt_data.index), :]

    # DE genes indexes
    name_vars = cnt_data.index.values
    symbol_vars = cnt_data.hgnc_symbol.values
    cnt_data = cnt_data.drop("hgnc_symbol", axis=1)

    y = dgea_results.index.values
    y = [name_vars.tolist().index(x) for x in y]

    # filter samples not in the meta_data file and make sure vst_data and meta_data have the same rows order
    cnt_data = cnt_data.T
    cnt_data = cnt_data.reindex(plasma_meta_data.SampleID.values).dropna(axis=0, how='any')

    # split data into train and test sets
    cnt_data_train, cnt_data_test, sepsis_cat_train, sepsis_cat_test = train_test_split(cnt_data,
                                                                                        meta_data.Group,
                                                                                        test_size=0.3,
                                                                                        random_state=123,
                                                                                        stratify=meta_data.Group)

    # store sample ids
    name_samps_train = cnt_data_train.index
    name_samps_test = cnt_data_test.index

    # make and save a vst version of the complete dataset
    vst_transformer_full = VstTransformer()
    vst_transformer_full = vst_transformer_full.fit(cnt_data)
    vst_data_full = vst_transformer_full.transform(cnt_data)
    vst_data_full = pd.DataFrame(vst_data_full, index=cnt_data.index, columns=cnt_data.columns)
    vst_data_full.to_csv(data_path + "processed/" + results_prefix + "_paxgene_" + algo_ + "_vsd.csv")

    if (algo_ == "xgb"):
        #########################
        # XGBOOST - CV within a CV for grid search and RFE as part of a pipeline
        # n_jobs should be set to 1/None (or at least not -1 in both) to avoid nested parallelism
        #########################
        rfecv = CustomRFECV(estimator=XGBClassifier(random_state=123, nthread=1), cv=5, n_jobs=1,
                            scoring='roc_auc', step=0.1,
                            min_features_to_select=2, verbose=True, max_features=100)

        if ratios_:
            # pipeline steps
            pipe = Pipeline([('norm', VstTransformer()),
                     ('filt', DGEA_filter(vars_to_keep=y)),
                     ('ratios', RatiosCalculator(name_vars=name_vars[y])),
                     ('rfe', rfecv),
                     ('xgbc', XGBClassifier())])
        else:
            pipe = Pipeline([('norm', VstTransformer()), ('scale', StandardScaler()),
                      ('filt', DGEA_filter(vars_to_keep=y)), ('rfe', rfecv),
                      ('xgbc', XGBClassifier())])

        # create the parameter grid
        param_grid = {
            'xgbc__random_state': [123],
            'xgbc__nthread': [1],
            'xgbc__learning_rate': [0.01, 0.1, 0.3],  # boosting learning rate 0 to 1, default 0.3
            'xgbc__objective': ['binary:logistic'],  # learning objective
            'xgbc__max_depth': [10, 50, 100],  # max number of levels in each tree
            'xgbc__n_estimators': [100, 1000]  # number of trees
        }

    else:
        if algo_ == "bsvm":
            #########################
            # BagSVM - CV within a CV for grid search and RFE as part of a pipeline
            # n_jobs should be set to 1/None (or at least not -1 in both) to avoid nested parallelism
            #########################
            rfecv = CustomRFECV(estimator=CustomBaggingClassifier(base_estimator=LinearSVC(max_iter=10000),
                                                                  random_state=123),
                                cv=5, n_jobs=1, scoring='roc_auc',
                                step=0.1, min_features_to_select=2, verbose=True, max_features=100)

            if ratios_:
                # pipeline steps
                pipe = Pipeline([('norm', VstTransformer()),
                                 ('filt', DGEA_filter(vars_to_keep=y)),
                                 ('ratios', RatiosCalculator(name_vars=name_vars[y])),
                                 ('rfe', rfecv),
                                 ('bsvmc', CustomBaggingClassifier(base_estimator=LinearSVC(max_iter=10000)))])
            else:
                pipe = Pipeline([('norm', VstTransformer()), ('scale', StandardScaler()),
                                 ('filt', DGEA_filter(vars_to_keep=y)), ('rfe', rfecv),
                                 ('bsvmc', CustomBaggingClassifier(base_estimator=LinearSVC(max_iter=10000)))])

            # create the parameter grid
            param_grid = {
                'bsvmc__random_state': [123],
                'bsvmc__base_estimator__C': [0.01, 0.1, 1, 10],  # regularisation parameter
                'bsvmc__max_features': [0.1, 0.5, 0.8],  # max number of features to draw (without replacement)
                'bsvmc__n_estimators': [100, 1000]  # number of base estimators
            }

    # add ratio prefix
    if ratios_:
        algo_ = "ratios_" + algo_

    # set the grid search object
    search = GridSearchCV(estimator=pipe, cv=5, n_jobs=1, scoring='roc_auc', param_grid=param_grid,
                          verbose=True)

    if mode_ == "create":
        # fit the chosen model
        search.fit(cnt_data_train, sepsis_cat_train)
        dump(search, results_path + results_prefix + "_paxgene" + "_dump_" + algo_ + "_de_genes.joblib")
    else:
        if mode_ == "load":
            search = load(results_path + results_prefix + "_paxgene" + "_dump_" + algo_ + "_de_genes.joblib")

    print(search.best_params_)
    print(search.best_score_)
    print(name_vars[y][search.best_estimator_.named_steps["rfe"].support_])

    # save list of predictors
    best_vars = name_vars[y][search.best_estimator_.named_steps["rfe"].support_]
    pd.DataFrame(best_vars).to_csv(results_path + results_prefix + "_paxgene" + "_best_vars_" + algo_ + "_de_genes.csv",
                                   header=False,
                                   index=False)

    # evaluate on test data
    probs = search.predict_proba(cnt_data_test)
    probs = probs[:, 1]
    roc_auc = roc_auc_score(sepsis_cat_test, probs)
    print(roc_auc)
