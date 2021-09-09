'''
2.build_classifiers.py
created on April 23 2021
lucile.neyton@ucsf.edu

This script aims at building classifiers to allocate samples to one of the two
groups (sepsis vs non-sepsis or viral non-viral) given gene expression values.
The final models are tested on a held-out set.

Input files (data folder):
    - CSV-formatted file containing gene counts (samples x genes)
    - CSV-formatted file containing raw plasma gene counts (samples x genes)
    - CSV-formatted file containing target labels (samples x)
    - CSV-formatted file containing the list of differentially expressed genes (genes x)

Output files (results folder):
    - One dump file per classifier
    - One list of predictors per classifier
'''

# load libraries and functions
# to load custom functions
import sys
sys.path.append("/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_VALID/de_analysis")
from custom_classes import VstTransformer, CustomRFECV, CustomBaggingClassifier, DGEA_filter
# 3.3.2
import matplotlib.pyplot as plt
# 1.19.1
import numpy as np
# 1.1.3
import pandas as pd
# 0.23.2
from sklearn.metrics import roc_auc_score, roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# 1.2.1
from xgboost import XGBClassifier
# 0.17.0
from joblib import dump, load

# set paths
plasma_data_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_plasma/data/"
paxgene_data_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_VALID/data/"
results_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_plasma/results/"

# set params
mode_ = "load"
# all_genes or overlap
genes_to_use = "all_genes"
num_cv = 5
comb_to_run = int(sys.argv[1])

# list parameter values
# data type / sample filter / gene filter / DE FDR / DE genes list generated from / covariates / algo /cv train test / comp
comb_list = [('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'xgb', True, "1vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'xgb', True, "1vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'bsvm', True, "1vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'bsvm', True, "1vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'xgb', True, "12vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'xgb', True, "12vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'bsvm', True, "12vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'bsvm', True, "12vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'xgb', True, "12"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'xgb', True, "12"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'bsvm', True, "12"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'bsvm', True, "12"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'xgb', True, "124"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'xgb', True, "124"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'bsvm', True, "124"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'bsvm', True, "124")]

# select comb to run
comb_list = [comb_list[comb_to_run]]

# external datasets and meta data files
gse28750_cnt_data = pd.read_csv(plasma_data_path + "processed/" + "GSE28750_geneLevel_data.csv", index_col=0)
gse9960_cnt_data = pd.read_csv(plasma_data_path + "processed/" + "GSE9960_geneLevel_data.csv", index_col=0)

gse28750_meta_data = pd.read_csv(plasma_data_path + "processed/" + "GSE28750_class_labels.csv", index_col=0)
gse9960_meta_data = pd.read_csv(plasma_data_path + "processed/" + "GSE9960_class_labels.csv", index_col=0)

plasma_data_path = plasma_data_path + genes_to_use + "/"
results_path = results_path + genes_to_use + "/"

#########################
# FOR EACH COMBINATION
#########################
for comb_ in comb_list:
    print("=====")
    print(comb_)
    print("=====")

    # infer the target variable
    comp_ = comb_[8]
    if comp_ in ['1vs4', '12vs4']:
        target_ = 'sepsis'
    else:
        if comp_ in ['12', '124']:
            target_ = 'virus'

    # build the results prefix that will be used to load the relevant files
    # and extract other parameters
    results_prefix = comb_[1] + "_" + comb_[2] + "_" + comb_[3] + "_" + comb_[5] + "_" + comb_[8] + "_" + target_
    data_from_ = comb_[0]
    de_genes_from_ = comb_[4]
    algo_ = comb_[6]
    cv_train_test_ = comb_[7]

    #########################
    # DATA LOADING
    #########################
    # counts data
    paxgene_cnt_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_paxgene_cnts.csv", index_col=0)
    plasma_cnt_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_plasma_cnts.csv", index_col=0)

    if comp_ == "12" and data_from_ == "plasma":
        G4_plasma_cnt_data = pd.read_csv(plasma_data_path + "../processed/" + "50000_plasma_G4_unfiltered_cnts.csv", index_col=0)
        G4_plasma_cnt_data = G4_plasma_cnt_data.loc[G4_plasma_cnt_data.index.isin(plasma_cnt_data.index), :]
        G4_plasma_cnt_data = G4_plasma_cnt_data.drop("hgnc_symbol", axis=1)
        G4_plasma_cnt_data = G4_plasma_cnt_data.T

    if data_from_ == "plasma":
        G3_plasma_cnt_data = pd.read_csv(plasma_data_path + "../processed/" + "50000_plasma_G3_unfiltered_cnts.csv",
                                         index_col=0)
        G3_plasma_cnt_data = G3_plasma_cnt_data.loc[G3_plasma_cnt_data.index.isin(plasma_cnt_data.index), :]
        G3_plasma_cnt_data = G3_plasma_cnt_data.drop("hgnc_symbol", axis=1)
        G3_plasma_cnt_data = G3_plasma_cnt_data.T

        G5_plasma_cnt_data = pd.read_csv(plasma_data_path + "../processed/" + "50000_plasma_G5_unfiltered_cnts.csv",
                                         index_col=0)
        G5_plasma_cnt_data = G5_plasma_cnt_data.loc[G5_plasma_cnt_data.index.isin(plasma_cnt_data.index), :]
        G5_plasma_cnt_data = G5_plasma_cnt_data.drop("hgnc_symbol", axis=1)
        G5_plasma_cnt_data = G5_plasma_cnt_data.T

    # plasma meta data (to order main metadata frame)
    meta_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_metadata.csv", index_col=0)
    meta_data_low = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_metadata_low.csv", index_col=0)

    # low counts data
    paxgene_cnt_data_low = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_paxgene_cnts_low.csv", index_col=0)
    plasma_cnt_data_low = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_plasma_cnts_low.csv", index_col=0)

    # choose the list of DE genes (paxgene or plasma)
    if de_genes_from_ == "paxgene":
        dgea_results = pd.read_csv(results_path + results_prefix + "_paxgene_DGEA_results.csv", index_col=0)
    else:
        if de_genes_from_ == "plasma":
            dgea_results = pd.read_csv(results_path + results_prefix + "_plasma_DGEA_results.csv", index_col=0)

    #########################
    # DATA PREPROCESSING
    #########################
    # drop the gene symbol column
    paxgene_cnt_data = paxgene_cnt_data.drop("hgnc_symbol", axis=1)
    paxgene_cnt_data_low = paxgene_cnt_data_low.drop("hgnc_symbol", axis=1)
    plasma_cnt_data = plasma_cnt_data.drop("hgnc_symbol", axis=1)
    plasma_cnt_data_low = plasma_cnt_data_low.drop("hgnc_symbol", axis=1)

    # keep only samples of interest (groups 1 and 4, from the metadata file)
    if comp_ == "12vs4":
        meta_data.sepsis_cat = ["1_Sepsis+BldCx+" if x=="2_Sepsis+OtherCx+" else x for x in meta_data.sepsis_cat]
        meta_data_low.sepsis_cat = ["1_Sepsis+BldCx+" if x=="2_Sepsis+OtherCx+" else x for x in meta_data_low.sepsis_cat]

    # add an extra column that will match the sample names from the counts data frames
    meta_data["EARLI_Barcode"] = ["EARLI_" + str(x) for x in meta_data["Barcode"]]
    meta_data_low["EARLI_Barcode"] = ["EARLI_" + str(x) for x in meta_data_low["Barcode"]]

    # order meta data frame given the counts plasma data frame
    meta_data = meta_data.set_index("EARLI_Barcode")
    meta_data = meta_data.loc[plasma_cnt_data.columns, :]

    meta_data_low = meta_data_low.set_index("EARLI_Barcode")
    meta_data_low = meta_data_low.loc[plasma_cnt_data_low.columns, :]

    # keep only genes overlapping
    if genes_to_use == "overlap":
        paxgene_cnt_data = paxgene_cnt_data.loc[paxgene_cnt_data.index.isin(plasma_cnt_data.index), :]

        # reindex paxgene data so that order is the same as for plasma data
        paxgene_cnt_data = paxgene_cnt_data.reindex(plasma_cnt_data.index)

        # reindex low count data frame
        paxgene_cnt_data_low = paxgene_cnt_data_low.reindex(plasma_cnt_data_low.index)

    # list genes used as input
    if data_from_ == "paxgene":
        name_vars = paxgene_cnt_data.index.values

    else:
        if data_from_ == "plasma":
            name_vars = plasma_cnt_data.index.values

    # DE genes indexes
    y = dgea_results.index.values
    y = [name_vars.tolist().index(x) for x in y]

    if algo_ == "xgb":
        #########################
        # XGBOOST - CV within a CV for grid search and RFE as part of a pipeline
        # n_jobs should be set to 1/None (or at least not -1 in both) to avoid nested parallelism
        #########################
        rfecv = CustomRFECV(estimator=XGBClassifier(random_state=123, nthread=1), cv=num_cv, n_jobs=1,
                            scoring='roc_auc', step=0.1,
                            min_features_to_select=2, verbose=True, max_features=100)

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
                                cv=num_cv, n_jobs=1, scoring='roc_auc',
                                step=0.1, min_features_to_select=2, verbose=True, max_features=100)

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

    # set the grid search object
    search = GridSearchCV(estimator=pipe, cv=num_cv, n_jobs=1, scoring='roc_auc', param_grid=param_grid, verbose=True)

    # add parameters prefix that will be used to save output files and figures
    output_prefix = comb_[0] + "_" + comb_[1] + "_" + comb_[2] + "_" + comb_[3] + "_" + comb_[4] + "_" + comb_[5] + \
                    "_" + comb_[6] + "_" + comb_[8] + "_" + target_

    # transpose data for classifiers
    paxgene_cnt_data = paxgene_cnt_data.T
    plasma_cnt_data = plasma_cnt_data.T
    paxgene_cnt_data_low = paxgene_cnt_data_low.T
    plasma_cnt_data_low = plasma_cnt_data_low.T

    # identify target data
    if data_from_ == "paxgene":
        target_data = paxgene_cnt_data
    else:
        if data_from_ == "plasma":
            target_data = plasma_cnt_data

    # target classes for count data
    if target_ == 'sepsis':
        target_cat = meta_data.sepsis_cat
    else:
        if target_ == 'virus':
            target_cat = meta_data.viruspos

    # split data to have a held-out set
    cnt_data_train_full, cnt_data_test_full, \
    target_cat_train_full, target_cat_test_full = train_test_split(target_data, target_cat,
                                                         test_size=0.3, random_state=123,
                                                         stratify=target_cat)

    #########################
    # FOR EACH CV
    #########################
    if cv_train_test_:
        # 10 CV
        cvs_id = range(0, 10)
    else:
        cvs_id = 0

    np.random.seed(123)
    cvs_aucroc_train = []
    cvs_aucroc_test1 = []
    cvs_aucroc_test2 = []
    cvs_aucroc_test_full = []
    cvs_aucroc_G4 = []
    cvs_aucroc_G3 = []
    cvs_aucroc_G5 = []

    # target classes for low count data
    if target_ == 'sepsis':
        target_cat_low = meta_data_low.sepsis_cat
    else:
        if target_ == 'virus':
            target_cat_low = meta_data_low.viruspos

    # for each CV set
    for i in cvs_id:
        cv_id = str(i+1)

        # split data into train and test sets
        cnt_data_train, cnt_data_test, \
            target_cat_train, target_cat_test = train_test_split(cnt_data_train_full, target_cat_train_full,
                                                                 test_size=0.3,
                                                                 stratify=target_cat_train_full)

        # store sample ids
        name_samps_train = cnt_data_train.index
        name_samps_test = cnt_data_test.index

        # fit/load models
        if i == 0:
            if mode_ == "create":
                # fit the chosen model
                search.fit(cnt_data_train, target_cat_train)
                dump(search, results_path + output_prefix + "_dump.joblib")
            else:
                if mode_ == "load":
                    search = load(results_path + output_prefix + "_dump.joblib")

        else:
            if mode_ == "create":
                # fit the chosen model
                search.fit(cnt_data_train, target_cat_train)
                dump(search, results_path + output_prefix + "_" + cv_id + "_dump.joblib")
            else:
                if mode_ == "load":
                    search = load(results_path + output_prefix + "_" + cv_id + "_dump.joblib")

        # extract predictors
        best_vars = name_vars[y][search.best_estimator_.named_steps["rfe"].support_]

        # add results to the list
        probs = search.predict_proba(cnt_data_train)[:, 1]
        cvs_aucroc_train.append({"labels": target_cat_train,
                                 "probs": probs,
                                 "classes": search.classes_[1],
                                 "n_preds": len(best_vars),
                                 "roc_auc": roc_auc_score(target_cat_train, probs)})

        # print output
        print(cv_id)
        print(search.best_params_)
        #print(search.best_score_)

        print(len(best_vars))
        print(best_vars)

        # evaluate on test data
        probs = search.predict_proba(cnt_data_test)
        probs = probs[:, 1]
        roc_auc = roc_auc_score(target_cat_test, probs)
        cvs_aucroc_test1.append({"labels": target_cat_test,
                                 "probs": probs,
                                 "classes": search.classes_[1],
                                 "roc_auc": roc_auc})
        print(roc_auc)

        # evaluate on full test data
        probs = search.predict_proba(cnt_data_test_full)
        probs = probs[:, 1]
        roc_auc = roc_auc_score(target_cat_test_full, probs)
        cvs_aucroc_test_full.append({"labels": target_cat_test_full,
                                     "probs": probs,
                                     "classes": search.classes_[1],
                                     "roc_auc": roc_auc})
        print(roc_auc)

        if data_from_ == "paxgene":
            # evaluate on low count paxgene samples
            probs = search.predict_proba(paxgene_cnt_data_low)
            probs = probs[:, 1]
            roc_auc = roc_auc_score(target_cat_low, probs)
            print(roc_auc)
            cvs_aucroc_test2.append({"labels": target_cat_low,
                                       "probs": probs,
                                       "classes": search.classes_[1],
                                       "roc_auc": roc_auc})

            if target_ == 'sepsis':
                # test on two external datasets with a good enough feature overlap
                # filter to keep only genes of interest
                gse28750_cnt_data_tmp = gse28750_cnt_data.loc[:, gse28750_cnt_data.columns.isin(best_vars)]
                gse9960_cnt_data_tmp = gse9960_cnt_data.loc[:, gse9960_cnt_data.columns.isin(best_vars)]

                # extract final classifier
                if algo_ == "xgb":
                    best_estimator = search.best_estimator_.named_steps["xgbc"]
                else:
                    if algo_ == "bsvm":
                        best_estimator = search.best_estimator_.named_steps["bsvmc"]

                # scale data
                ext_scaler = StandardScaler()
                gse28750_cnt_data_scaled_tmp = ext_scaler.fit_transform(gse28750_cnt_data_tmp)
                gse9960_cnt_data_scaled_tmp = ext_scaler.fit_transform(gse9960_cnt_data_tmp)

                # apply classifiers
                if gse28750_cnt_data_scaled_tmp.shape[1] == len(best_vars):
                    probs = best_estimator.predict_proba(gse28750_cnt_data_scaled_tmp)
                    probs = probs[:, 1]
                    roc_auc = roc_auc_score(gse28750_meta_data.sepsis_cat, probs)
                    print(roc_auc)

                if gse9960_cnt_data_scaled_tmp.shape[1] == len(best_vars):
                    probs = best_estimator.predict_proba(gse9960_cnt_data_scaled_tmp)
                    probs = probs[:, 1]
                    roc_auc = roc_auc_score(gse9960_meta_data.sepsis_cat, probs)
                    print(roc_auc)

        else:
            if data_from_ == "plasma":
                # evaluate on low count plasma samples
                probs = search.predict_proba(plasma_cnt_data_low)
                probs = probs[:, 1]
                roc_auc = roc_auc_score(target_cat_low, probs)
                print(roc_auc)
                cvs_aucroc_test2.append(
                    {"labels": target_cat_low,
                     "probs": probs,
                     "classes": search.classes_[1],
                     "roc_auc": roc_auc})

                if comp_ == '12':
                    # evaluate on G4 plasma samples
                    probs = search.predict_proba(G4_plasma_cnt_data)
                    probs = probs[:, 1]
                    csv_labels = pd.DataFrame(np.repeat("nonviral", G4_plasma_cnt_data.shape[0]), index=G4_plasma_cnt_data.index)
                    cvs_aucroc_G4.append(
                        {"labels": csv_labels,
                         "probs": probs,
                         "classes": search.classes_[1],
                         "roc_auc": np.nan})

                # evaluate on G3 and G5 plasma samples
                probs = search.predict_proba(G3_plasma_cnt_data)
                probs = probs[:, 1]
                csv_labels = pd.DataFrame(np.repeat("nonviral", G3_plasma_cnt_data.shape[0]),
                                          index=G3_plasma_cnt_data.index)
                cvs_aucroc_G3.append(
                    {"labels": csv_labels,
                     "probs": probs,
                     "classes": search.classes_[1],
                     "roc_auc": np.nan})

                probs = search.predict_proba(G5_plasma_cnt_data)
                probs = probs[:, 1]
                csv_labels = pd.DataFrame(np.repeat("nonviral", G5_plasma_cnt_data.shape[0]),
                                          index=G5_plasma_cnt_data.index)
                cvs_aucroc_G5.append(
                    {"labels": csv_labels,
                     "probs": probs,
                     "classes": search.classes_[1],
                     "roc_auc": np.nan})


    # rebuild the model on the full train set and test on held-out
    if mode_ == "create":
        search.fit(cnt_data_train_full, target_cat_train_full)
        dump(search, results_path + output_prefix + "_full_dump.joblib")
    else:
        if mode_ == "load":
            search = load(results_path + output_prefix + "_full_dump.joblib")

    # full train set
    #print(search.best_score_)

    # extract and save predictors
    best_vars = name_vars[y][search.best_estimator_.named_steps["rfe"].support_]
    print(len(best_vars))
    pd.DataFrame(best_vars).to_csv(results_path + output_prefix + "_full_best_vars.csv", header=False,
                                   index=False)

    # test on full test set
    probs = search.predict_proba(cnt_data_test_full)[:, 1]
    roc_auc_test_full = roc_auc_score(target_cat_test_full, probs)
    print(roc_auc_test_full)

    # test full model on low counts and external data
    if data_from_ == "paxgene":
        # evaluate on low count paxgene samples
        probs = search.predict_proba(paxgene_cnt_data_low)
        probs = probs[:, 1]
        roc_auc_low = roc_auc_score(target_cat_low, probs)
        print(roc_auc_low)

        if target_ == 'sepsis':
            # test on two external datasets with a good feature overlap
            # filter to keep only genes of interest
            gse28750_cnt_data_tmp = gse28750_cnt_data.loc[:, gse28750_cnt_data.columns.isin(best_vars)]
            gse9960_cnt_data_tmp = gse9960_cnt_data.loc[:, gse9960_cnt_data.columns.isin(best_vars)]

            if algo_ == "xgb":
                best_estimator = search.best_estimator_.named_steps["xgbc"]
            else:
                if algo_ == "bsvm":
                    best_estimator = search.best_estimator_.named_steps["bsvmc"]

            # scale data
            ext_scaler = StandardScaler()
            gse28750_cnt_data_scaled_tmp = ext_scaler.fit_transform(gse28750_cnt_data_tmp)
            gse9960_cnt_data_scaled_tmp = ext_scaler.fit_transform(gse9960_cnt_data_tmp)

            # apply classifiers
            if gse28750_cnt_data_scaled_tmp.shape[1] == len(best_vars):
                probs = best_estimator.predict_proba(gse28750_cnt_data_scaled_tmp)
                probs = probs[:, 1]
                roc_auc = roc_auc_score(gse28750_meta_data.sepsis_cat, probs)
                print(roc_auc)

            if gse9960_cnt_data_scaled_tmp.shape[1] == len(best_vars):
                probs = best_estimator.predict_proba(gse9960_cnt_data_scaled_tmp)
                probs = probs[:, 1]
                roc_auc = roc_auc_score(gse9960_meta_data.sepsis_cat, probs)
                print(roc_auc)

    else:
        if data_from_ == "plasma":
            # evaluate on low count plasma samples
            probs = search.predict_proba(plasma_cnt_data_low)
            probs = probs[:, 1]
            roc_auc_low = roc_auc_score(target_cat_low, probs)
            print(roc_auc_low)

    #########################
    # Performance summary
    #########################
    # performance summary
    res_dict = {'cv_id': [x + 1 for x in cvs_id],
                'n_preds': [x["n_preds"] for x in cvs_aucroc_train],
                'roc_auc_train': [x["roc_auc"].round(2) for x in cvs_aucroc_train],
                'roc_auc_test1': [x["roc_auc"].round(2) for x in cvs_aucroc_test1],
                'roc_auc_test2': [x["roc_auc"].round(2) for x in cvs_aucroc_test2],
                'roc_auc_test_full': [x["roc_auc"].round(2) for x in cvs_aucroc_test_full]}

    # add mean and std
    res_dict['cv_id'] = res_dict['cv_id'] + ["mean (std)"]
    res_dict['n_preds'] = res_dict['n_preds'] + [""]
    res_dict['roc_auc_train'] = res_dict['roc_auc_train'] + [
        str(np.mean(res_dict['roc_auc_train']).round(2)) + " (" + str(np.std(res_dict['roc_auc_train']).round(2)) + ")"]
    res_dict['roc_auc_test1'] = res_dict['roc_auc_test1'] + [
        str(np.mean(res_dict['roc_auc_test1']).round(2)) + " (" + str(np.std(res_dict['roc_auc_test1']).round(2)) + ")"]
    res_dict['roc_auc_test2'] = res_dict['roc_auc_test2'] + [
        str(np.mean(res_dict['roc_auc_test2']).round(2)) + " (" + str(np.std(res_dict['roc_auc_test2']).round(2)) + ")"]
    res_dict['roc_auc_test_full'] = res_dict['roc_auc_test_full'] + [
        str(np.mean(res_dict['roc_auc_test_full']).round(2)) + " (" + str(np.std(res_dict['roc_auc_test_full']).round(2)) + ")"]

    # add full model summary
    probs = search.predict_proba(cnt_data_train_full)[:, 1]
    res_dict['cv_id'] = res_dict['cv_id'] + ["full"]
    res_dict['n_preds'] = res_dict['n_preds'] + [len(best_vars)]
    res_dict['roc_auc_train'] = res_dict['roc_auc_train'] + [roc_auc_score(target_cat_train_full, probs).round(2)]
    res_dict['roc_auc_test1'] = res_dict['roc_auc_test1'] + ["NA"]
    res_dict['roc_auc_test2'] = res_dict['roc_auc_test2'] + [roc_auc_low.round(2)]
    res_dict['roc_auc_test_full'] = res_dict['roc_auc_test_full'] + [roc_auc_test_full.round(2)]

    pd.DataFrame(data=res_dict).to_csv(results_path + output_prefix + "_summary_table.csv")

    #########################
    # AUC-ROC curves
    #########################
    # plot ROC curve for the training set
    plt.figure()
    for i in cvs_id:
        cv_id = str(i + 1)
        fpr_train, tpr_train, _ = roc_curve(cvs_aucroc_train[i]["labels"],
                                            cvs_aucroc_train[i]["probs"],
                                            pos_label=cvs_aucroc_train[i]["classes"])

        if i == 0:
            plt.plot(fpr_train, tpr_train, label='Cross-validation splits, AUC=' +
                                                 str(np.mean([x["roc_auc"].round(2) for x in cvs_aucroc_train]).round(2)) + " (" +
                                                 str(np.std([x["roc_auc"].round(2) for x in cvs_aucroc_train]).round(2)) + ")",
                     color="red", linewidth=1, alpha=0.3)
        else:
            plt.plot(fpr_train, tpr_train, color="red", linewidth=1, alpha=0.3)

    # plot ROC curve for the full train set
    fpr_train, tpr_train, _ = roc_curve(target_cat_train_full,
                                        probs,
                                        pos_label=search.classes_[1])

    plt.plot(fpr_train, tpr_train, label='Full train set, AUC=' + str(roc_auc_score(target_cat_train_full, probs).round(2)), color='grey',
             linewidth=2)
    plt.plot([0, 1], [0, 1], color='grey', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Train sets')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_train_all_auc_roc.pdf")

    # plot ROC curve for the testing set
    plt.figure()
    for i in cvs_id:
        cv_id = str(i + 1)
        fpr_test, tpr_test, _ = roc_curve(cvs_aucroc_test1[i]["labels"],
                                          cvs_aucroc_test1[i]["probs"],
                                          pos_label=cvs_aucroc_test1[i]["classes"])

        if i == 0:
            plt.plot(fpr_test, tpr_test, label='Cross-validation splits, AUC=' +
                                               str(np.mean([x["roc_auc"].round(2) for x in cvs_aucroc_test1]).round(2)) + " (" +
                                               str(np.std([x["roc_auc"].round(2) for x in cvs_aucroc_test1]).round(2)) + ")",
                     color="red", linewidth=1, alpha=0.3)
        else:
            plt.plot(fpr_test, tpr_test, color="red", linewidth=1, alpha=0.3)

    # plot ROC curve for the full test set
    probs = search.predict_proba(cnt_data_test_full)
    probs = probs[:, 1]
    roc_auc = roc_auc_score(target_cat_test_full, probs)
    fpr_test, tpr_test, _ = roc_curve(target_cat_test_full,
                                      probs,
                                      pos_label=search.classes_[1])

    plt.plot(fpr_test, tpr_test, label='Full test set, AUC=' + str(roc_auc.round(2)), color='grey',
             linewidth=2)
    plt.plot([0, 1], [0, 1], color='grey', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Test sets')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_test_all_auc_roc.pdf")

    # plot confusion matrices
    plot_confusion_matrix(search, cnt_data_train_full, target_cat_train_full)
    plt.savefig(results_path + output_prefix + "_train_full_conf_mat.pdf")
    plot_confusion_matrix(search, cnt_data_test_full, target_cat_test_full)
    plt.savefig(results_path + output_prefix + "_test_full_conf_mat.pdf")

    # save predicted probabilities for the train set
    cvs_dict_probs = {}
    for i in cvs_id:
        cv_id = str(i + 1)
        for id_ in cvs_aucroc_test1[i]["labels"].index:
            if id_ not in cvs_dict_probs.keys():
                cvs_dict_probs[id_] = [
                    cvs_aucroc_test1[i]["probs"][np.where(cvs_aucroc_test1[i]["labels"].index == id_)][0]]
            else:
                cvs_dict_probs[id_] = cvs_dict_probs[id_] + [cvs_aucroc_test1[i]["probs"][np.where(cvs_aucroc_test1[i]["labels"].index == id_)][0]]

    for id_ in cvs_dict_probs.keys():
        cvs_dict_probs[id_] = np.mean(cvs_dict_probs[id_])

    if (search.classes_[1] != "1_Sepsis+BldCx+") & (search.classes_[1] != "viral"):
        for id_ in cvs_dict_probs.keys():
            cvs_dict_probs[id_] = 1 - cvs_dict_probs[id_]

    dict_df = pd.DataFrame.from_dict(data=cvs_dict_probs, orient='index')
    if target_ == 'sepsis':
        dict_df['true_label'] = meta_data.loc[dict_df.index, "sepsis_cat"]
    else:
        if target_ == 'virus':
            dict_df['true_label'] = meta_data.loc[dict_df.index, "viruspos"]
    dict_df.to_csv(results_path + output_prefix + "_mean_full_train_probs.csv")

    # save probabilities for full test samples
    cvs_dict_probs = {}
    for i in cvs_id:
        cv_id = str(i + 1)
        for id_ in cvs_aucroc_test_full[i]["labels"].index:
            if id_ not in cvs_dict_probs.keys():
                cvs_dict_probs[id_] = [
                    cvs_aucroc_test_full[i]["probs"][np.where(cvs_aucroc_test_full[i]["labels"].index == id_)][0]]
            else:
                cvs_dict_probs[id_] = cvs_dict_probs[id_] + [
                    cvs_aucroc_test_full[i]["probs"][np.where(cvs_aucroc_test_full[i]["labels"].index == id_)][0]]

    for id_ in cvs_dict_probs.keys():
        cvs_dict_probs[id_] = np.mean(cvs_dict_probs[id_])

    if (search.classes_[1] != "1_Sepsis+BldCx+") & (search.classes_[1] != "viral"):
        for id_ in cvs_dict_probs.keys():
            cvs_dict_probs[id_] = 1 - cvs_dict_probs[id_]

    dict_df = pd.DataFrame.from_dict(data=cvs_dict_probs, orient='index')
    dict_df['true_label'] = target_cat_test_full
    dict_df.to_csv(results_path + output_prefix + "_mean_full_test_probs.csv")

    # save probabilities for G4 samples
    if comp_ == '12' and data_from_ == "plasma":
        cvs_dict_probs = {}
        for i in cvs_id:
            cv_id = str(i + 1)
            for id_ in cvs_aucroc_G4[i]["labels"].index:
                if id_ not in cvs_dict_probs.keys():
                    cvs_dict_probs[id_] = [
                        cvs_aucroc_G4[i]["probs"][np.where(cvs_aucroc_G4[i]["labels"].index == id_)][0]]
                else:
                    cvs_dict_probs[id_] = cvs_dict_probs[id_] + [
                        cvs_aucroc_G4[i]["probs"][np.where(cvs_aucroc_G4[i]["labels"].index == id_)][0]]

        for id_ in cvs_dict_probs.keys():
            cvs_dict_probs[id_] = np.mean(cvs_dict_probs[id_])

        if search.classes_[1] != "viral":
            for id_ in cvs_dict_probs.keys():
                cvs_dict_probs[id_] = 1 - cvs_dict_probs[id_]

        dict_df = pd.DataFrame.from_dict(data=cvs_dict_probs, orient='index')
        dict_df['true_label'] = np.repeat('nonviral', len(cvs_dict_probs))
        dict_df.to_csv(results_path + output_prefix + "_mean_G4_probs.csv")

    if data_from_ == "plasma":
        cvs_dict_probs = {}
        for i in cvs_id:
            cv_id = str(i + 1)
            for id_ in cvs_aucroc_G3[i]["labels"].index:
                if id_ not in cvs_dict_probs.keys():
                    cvs_dict_probs[id_] = [
                        cvs_aucroc_G3[i]["probs"][np.where(cvs_aucroc_G3[i]["labels"].index == id_)][0]]
                else:
                    cvs_dict_probs[id_] = cvs_dict_probs[id_] + [
                        cvs_aucroc_G3[i]["probs"][np.where(cvs_aucroc_G3[i]["labels"].index == id_)][0]]

        for id_ in cvs_dict_probs.keys():
            cvs_dict_probs[id_] = np.mean(cvs_dict_probs[id_])

        if (search.classes_[1] != "viral") & (search.classes_[1] != "1_Sepsis+BldCx+"):
            for id_ in cvs_dict_probs.keys():
                cvs_dict_probs[id_] = 1 - cvs_dict_probs[id_]

        dict_df = pd.DataFrame.from_dict(data=cvs_dict_probs, orient='index')
        dict_df.to_csv(results_path + output_prefix + "_mean_G3_probs.csv")

        cvs_dict_probs = {}
        for i in cvs_id:
            cv_id = str(i + 1)
            for id_ in cvs_aucroc_G5[i]["labels"].index:
                if id_ not in cvs_dict_probs.keys():
                    cvs_dict_probs[id_] = [
                        cvs_aucroc_G5[i]["probs"][np.where(cvs_aucroc_G5[i]["labels"].index == id_)][0]]
                else:
                    cvs_dict_probs[id_] = cvs_dict_probs[id_] + [
                        cvs_aucroc_G5[i]["probs"][np.where(cvs_aucroc_G5[i]["labels"].index == id_)][0]]

        for id_ in cvs_dict_probs.keys():
            cvs_dict_probs[id_] = np.mean(cvs_dict_probs[id_])

        if (search.classes_[1] != "viral") & (search.classes_[1] != "1_Sepsis+BldCx+"):
            for id_ in cvs_dict_probs.keys():
                cvs_dict_probs[id_] = 1 - cvs_dict_probs[id_]

        dict_df = pd.DataFrame.from_dict(data=cvs_dict_probs, orient='index')
        dict_df.to_csv(results_path + output_prefix + "_mean_G5_probs.csv")

    # save predicted probabilities for the full train and test sets (+G4 if plasma 12 virus comparison)
    # train set
    probs_full_train = search.predict_proba(cnt_data_train_full)
    probs_full_train = probs_full_train[:, 1]
    if (search.classes_[1] == "1_Sepsis+BldCx+") | (search.classes_[1] == "viral"):
        target_probs = probs_full_train
    else:
        target_probs = [1 - x for x in probs_full_train]

    dict_probs = {'sample_ids': cnt_data_train_full.index.values,
                  'probs': target_probs, 'true_label': target_cat_train_full}
    pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_full_train_probs.csv")

    # test set
    if (search.classes_[1] == "1_Sepsis+BldCx+") | (search.classes_[1] == "viral"):
        target_probs = probs
    else:
        target_probs = [1 - x for x in probs]

    dict_probs = {'sample_ids': cnt_data_test_full.index.values,
                  'probs': target_probs, 'true_label': target_cat_test_full}
    pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_full_test_probs.csv")

    # G4 test set
    if comp_ == "12" and data_from_ == "plasma":
        probs_G4 = search.predict_proba(G4_plasma_cnt_data)
        probs_G4 = probs_G4[:, 1]
        if search.classes_[1] == "viral":
            target_probs = probs_G4
        else:
            target_probs = [1 - x for x in probs_G4]

        dict_probs = {'sample_ids': G4_plasma_cnt_data.index.values,
                      'probs': target_probs, 'true_label': np.repeat('nonviral', len(G4_plasma_cnt_data.index.values))}
        pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_G4_probs.csv")

    # G3 and G5 test sets
    if data_from_ == "plasma":
        probs_G3 = search.predict_proba(G3_plasma_cnt_data)
        probs_G3 = probs_G3[:, 1]
        if (search.classes_[1] == "1_Sepsis+BldCx+") | (search.classes_[1] == "viral"):
            target_probs = probs_G3
        else:
            target_probs = [1 - x for x in probs_G3]

        dict_probs = {'sample_ids': G3_plasma_cnt_data.index.values,
                      'probs': target_probs}
        pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_G3_probs.csv")

        probs_G5 = search.predict_proba(G5_plasma_cnt_data)
        probs_G5 = probs_G5[:, 1]
        if (search.classes_[1] == "1_Sepsis+BldCx+") | (search.classes_[1] == "viral"):
            target_probs = probs_G5
        else:
            target_probs = [1 - x for x in probs_G5]

        dict_probs = {'sample_ids': G5_plasma_cnt_data.index.values,
                      'probs': target_probs}
        pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_G5_probs.csv")