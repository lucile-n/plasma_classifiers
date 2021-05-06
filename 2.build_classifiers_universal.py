'''
2.build_classifiers_universal.py
created on April 23 2021
lucile.neyton@ucsf.edu

This script aims at building classifiers to allocate samples to one of the two
groups (sepsis vs non-sepsis) given gene expression values.
The final models are tested on a held-out set.

Input files (data folder):
    - CSV-formatted file containing gene counts (samples x genes)
    - CSV-formatted file containing raw plasma gene counts (samples x genes)
    - CSV-formatted file containing sepsis labels (samples x)
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
from sklearn.metrics import roc_auc_score, roc_curve
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

# list parameter values
# data type / sample filter / gene filter / DE FDR / DE genes list generated from / covariates / algo /cv train test
comb_list = [('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'xgb', True, "1vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'xgb', True, "1vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'bsvm', True, "1vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'bsvm', True, "1vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'xgb', True, "12vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'xgb', True, "12vs4"),
             ('plasma', '50000', '20', '0.1', 'plasma', 'TRUE', 'bsvm', True, "12vs4"),
             ('paxgene', '50000', '20', '0.1', 'paxgene', 'TRUE', 'bsvm', True, "12vs4")]

# read data
# counts data
paxgene_cnt_data = pd.read_csv(plasma_data_path + "processed/" + "paxgene_cnts.csv", index_col=0)

# read metadata
# group labels
meta_data = pd.read_csv(
    paxgene_data_path + "processed/" + "EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20.csv")

# external datasets and meta data files
gse28750_cnt_data = pd.read_csv(plasma_data_path + "processed/" + "GSE28750_geneLevel_data.csv", index_col=0)
gse9960_cnt_data = pd.read_csv(plasma_data_path + "processed/" + "GSE9960_geneLevel_data.csv", index_col=0)

gse28750_meta_data = pd.read_csv(plasma_data_path + "processed/" + "GSE28750_class_labels.csv", index_col=0)
gse9960_meta_data = pd.read_csv(plasma_data_path + "processed/" + "GSE9960_class_labels.csv", index_col=0)

#########################
# FOR EACH COMBINATION
#########################
for comb_ in comb_list:
    results_prefix = comb_[1] + "_" + comb_[2] + "_" + comb_[3] + "_" + comb_[5] + "_" + comb_[8]
    data_from_ = comb_[0]
    de_genes_from_ = comb_[4]
    algo_ = comb_[6]
    cv_train_test_ = comb_[7]
    comp_ = comb_[8]

    print("=====")
    print(comb_)
    print("=====")

    #########################
    # DATA LOADING
    #########################
    # counts data
    paxgene_cnt_data_tmp = paxgene_cnt_data
    plasma_cnt_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_plasma_cnts.csv", index_col=0)

    # plasma meta data (to order main metadata frame)
    plasma_meta_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_metadata.csv", index_col=0)

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
    # keep only samples of interest (groups 1 and 4, from the metadata file)
    plasma_cnt_data = plasma_cnt_data.drop("hgnc_symbol", axis=1)

    meta_data_tmp = meta_data

    if comp_ == "12vs4":
        meta_data_tmp.Group = ["1_Sepsis+BldCx+" if x=="2_Sepsis+OtherCx+" else x for x in meta_data_tmp.Group]

    meta_data_tmp["EARLI_Barcode"] = ["EARLI_" + str(x) for x in meta_data_tmp["Barcode"]]

    meta_data_tmp_low = meta_data_tmp

    meta_data_tmp = meta_data_tmp.loc[(meta_data_tmp.EARLI_Barcode.isin(plasma_cnt_data.columns.values)), :]
    meta_data_tmp = meta_data_tmp.rename(columns={"Group": "sepsis_cat"})
    paxgene_cnt_data_tmp = paxgene_cnt_data_tmp.loc[:, paxgene_cnt_data_tmp.columns.isin(meta_data_tmp.HOST_PAXgene_filename)]

    meta_data_tmp_low = meta_data_tmp_low.loc[(meta_data_tmp_low.EARLI_Barcode.isin(plasma_cnt_data_low.columns.values)), :]
    meta_data_tmp_low = meta_data_tmp_low.rename(columns={"Group": "sepsis_cat"})

    # order meta data frame
    meta_data_tmp = meta_data_tmp.set_index("EARLI_Barcode")
    meta_data_tmp = meta_data_tmp.loc[plasma_meta_data.SampleID, :]

    meta_data_tmp_low = meta_data_tmp_low.set_index("EARLI_Barcode")
    meta_data_tmp_low = meta_data_tmp_low.loc[plasma_cnt_data_low.columns.drop(["hgnc_symbol"]), :]

    # format count data columns
    paxgene_cnt_data_tmp.columns = [x.split("_")[0] + "_" + x.split("_")[1] for x in paxgene_cnt_data_tmp.columns]

    # keep only pre-filtered genes
    paxgene_cnt_data_tmp = paxgene_cnt_data_tmp.loc[paxgene_cnt_data_tmp.index.isin(plasma_cnt_data.index), :]

    # reindex paxgene data so that order is the same as for plasma data
    paxgene_cnt_data_tmp = paxgene_cnt_data_tmp.reindex(plasma_cnt_data.index)
    paxgene_cnt_data_tmp = paxgene_cnt_data_tmp.loc[:, plasma_cnt_data.columns.values]

    # reindex low count data frame
    paxgene_cnt_data_low = paxgene_cnt_data_low.reindex(plasma_cnt_data_low.index)
    paxgene_cnt_data_low = paxgene_cnt_data_low.loc[:, plasma_cnt_data_low.columns.values]

    # for target data
    if data_from_ == "paxgene":
        name_vars = paxgene_cnt_data_tmp.index.values

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
        rfecv = CustomRFECV(estimator=XGBClassifier(random_state=123, nthread=1), cv=5, n_jobs=1,
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
                                cv=5, n_jobs=1, scoring='roc_auc',
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
    search = GridSearchCV(estimator=pipe, cv=5, n_jobs=1, scoring='roc_auc', param_grid=param_grid,
                          verbose=True)

    # add parameters prefix
    output_prefix = comb_[0] + "_" + comb_[1] + "_" + comb_[2] + "_" + comb_[3] + "_" + comb_[4] + \
                    "_" + comb_[5] + "_" + comb_[6] + "_" + comb_[8]

    # transpose data for classifiers
    paxgene_cnt_data_tmp = paxgene_cnt_data_tmp.T
    plasma_cnt_data = plasma_cnt_data.T
    paxgene_cnt_data_low = paxgene_cnt_data_low.T
    plasma_cnt_data_low = plasma_cnt_data_low.T

    if data_from_ == "paxgene":
        target_data = paxgene_cnt_data_tmp
    else:
        if data_from_ == "plasma":
            target_data = plasma_cnt_data

    #########################
    # FOR EACH FOLD
    #########################
    if cv_train_test_:
        # 10-fold CV
        folds_id = range(0, 10)
    else:
        folds_id = 0

    # for each fold
    np.random.seed(123)
    folds_aucroc_train = []
    folds_aucroc_test1 = []
    folds_aucroc_test2 = []

    # prepare data
    paxgene_cnt_data_low = paxgene_cnt_data_low.drop("hgnc_symbol", axis=0)
    plasma_cnt_data_low = plasma_cnt_data_low.drop("hgnc_symbol", axis=0)

    for i in folds_id:
        fold_id = str(i+1)

        # split data into train and test sets
        cnt_data_train, cnt_data_test, \
            sepsis_cat_train, sepsis_cat_test = train_test_split(target_data, meta_data_tmp.sepsis_cat,
                                                                 test_size=0.3,
                                                                 stratify=meta_data_tmp.sepsis_cat)

        # store sample ids
        name_samps_train = cnt_data_train.index
        name_samps_test = cnt_data_test.index

        # fit/load models
        if i == 0:
            if mode_ == "create":
                # fit the chosen model
                search.fit(cnt_data_train, sepsis_cat_train)
                dump(search, results_path + output_prefix + "_dump.joblib")
            else:
                if mode_ == "load":
                    search = load(results_path + output_prefix + "_dump.joblib")

            # save list of predictors
            best_vars = name_vars[y][search.best_estimator_.named_steps["rfe"].support_]
            pd.DataFrame(best_vars).to_csv(results_path + output_prefix + "_best_vars.csv", header=False,
                                           index=False)

        else:
            if mode_ == "create":
                # fit the chosen model
                search.fit(cnt_data_train, sepsis_cat_train)
                dump(search, results_path + output_prefix + "_" + fold_id + "_dump.joblib")
            else:
                if mode_ == "load":
                    search = load(results_path + output_prefix + "_" + fold_id + "_dump.joblib")
        folds_aucroc_train.append({"labels": sepsis_cat_train,
                                   "probs": search.predict_proba(cnt_data_train)[:, 1],
                                   "classes": search.classes_[1]})

        print(fold_id)
        print(search.best_params_)
        print(search.best_score_)

        print(name_vars[y][search.best_estimator_.named_steps["rfe"].support_])

        # evaluate on test data
        probs = search.predict_proba(cnt_data_test)
        probs = probs[:, 1]
        roc_auc = roc_auc_score(sepsis_cat_test, probs)
        folds_aucroc_test1.append({"labels": sepsis_cat_test,
                                   "probs": probs,
                                   "classes": search.classes_[1]})
        print(roc_auc)

        if data_from_ == "paxgene":
            # evaluate on low count paxgene samples
            probs = search.predict_proba(paxgene_cnt_data_low)
            probs = probs[:, 1]
            roc_auc = roc_auc_score(meta_data_tmp_low.sepsis_cat, probs)
            print(roc_auc)
            folds_aucroc_test2.append({"labels": meta_data.loc[meta_data.EARLI_Barcode.isin(paxgene_cnt_data_low.index.values), :].Group,
                                       "probs": probs,
                                       "classes": search.classes_[1]})

        else:
            if data_from_ == "plasma":
                # evaluate on low count plasma samples
                probs = search.predict_proba(plasma_cnt_data_low)
                probs = probs[:, 1]
                roc_auc = roc_auc_score(meta_data_tmp_low.sepsis_cat, probs)
                print(roc_auc)
                folds_aucroc_test2.append(
                    {"labels": meta_data.loc[meta_data.EARLI_Barcode.isin(plasma_cnt_data_low.index.values), :].Group,
                     "probs": probs,
                     "classes": search.classes_[1]})

    # ROC curves
    plt.figure()
    for i in folds_id:
        # plot ROC curve for
        fpr_train, tpr_train, _ = roc_curve(folds_aucroc_train[i]["labels"],
                                            folds_aucroc_train[i]["probs"],
                                            pos_label=folds_aucroc_train[i]["classes"])

        plt.plot(fpr_train, tpr_train, label='fold ' + fold_id)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Train set')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_train_auc_roc.pdf")

    # ROC curves
    plt.figure()
    for i in folds_id:
        # plot ROC curve for
        fpr_train, tpr_train, _ = roc_curve(folds_aucroc_test1[i]["labels"],
                                            folds_aucroc_test1[i]["probs"],
                                            pos_label=folds_aucroc_test1[i]["classes"])

        plt.plot(fpr_train, tpr_train, label='fold ' + fold_id)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Test set')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_test_auc_roc.pdf")

    # ROC curves
    plt.figure()
    for i in folds_id:
        # plot ROC curve for
        fpr_train, tpr_train, _ = roc_curve(folds_aucroc_test2[i]["labels"],
                                            folds_aucroc_test2[i]["probs"],
                                            pos_label=folds_aucroc_test2[i]["classes"])

        plt.plot(fpr_train, tpr_train, label='fold ' + fold_id)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Alternative test set')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_alt_test_auc_roc.pdf")