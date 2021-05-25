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
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, plot_confusion_matrix
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
genes_to_use = "overlap"
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

    # split data to have a held-out set
    if target_ == 'sepsis':
        cnt_data_train_full, cnt_data_test_full, \
        target_cat_train_full, target_cat_test_full = train_test_split(target_data, meta_data.sepsis_cat,
                                                             test_size=0.3, random_state=123,
                                                             stratify=meta_data.sepsis_cat)
    else:
        if target_ == 'virus':
            cnt_data_train_full, cnt_data_test_full, \
            target_cat_train_full, target_cat_test_full = train_test_split(target_data, meta_data.viruspos,
                                                                           test_size=0.3, random_state=123,
                                                                           stratify=meta_data.viruspos)

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
        cvs_aucroc_train.append({"labels": target_cat_train,
                                 "probs": search.predict_proba(cnt_data_train)[:, 1],
                                 "classes": search.classes_[1],
                                 "n_preds": len(best_vars),
                                 "roc_auc": search.best_score_})

        # print output
        print(cv_id)
        print(search.best_params_)
        print(search.best_score_)

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

    # plot ROC curve for the training set
    plt.figure()
    for i in cvs_id:
        cv_id = str(i + 1)
        fpr_train, tpr_train, _ = roc_curve(cvs_aucroc_train[i]["labels"],
                                            cvs_aucroc_train[i]["probs"],
                                            pos_label=cvs_aucroc_train[i]["classes"])

        plt.plot(fpr_train, tpr_train, label='cv ' + cv_id)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Train set')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_train_auc_roc.pdf")

    # plot ROC curve for the testing set
    plt.figure()
    for i in cvs_id:
        cv_id = str(i + 1)

        # plot ROC curve for
        fpr_train, tpr_train, _ = roc_curve(cvs_aucroc_test1[i]["labels"],
                                            cvs_aucroc_test1[i]["probs"],
                                            pos_label=cvs_aucroc_test1[i]["classes"])

        plt.plot(fpr_train, tpr_train, label='cv ' + cv_id)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Test set')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_test_auc_roc.pdf")

    # plot ROC curve for the low counts set
    plt.figure()
    for i in cvs_id:
        cv_id = str(i + 1)

        # plot ROC curve for
        fpr_train, tpr_train, _ = roc_curve(cvs_aucroc_test2[i]["labels"],
                                            cvs_aucroc_test2[i]["probs"],
                                            pos_label=cvs_aucroc_test2[i]["classes"])

        plt.plot(fpr_train, tpr_train, label='cv ' + cv_id)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Alternative test set')
    plt.legend()

    plt.savefig(results_path + output_prefix + "_alt_test_auc_roc.pdf")

    # rebuild the model on the full train set and test on held-out
    if mode_ == "create":
        search.fit(cnt_data_train_full, target_cat_train_full)
        dump(search, results_path + output_prefix + "_full_dump.joblib")
    else:
        if mode_ == "load":
            search = load(results_path + output_prefix + "_full_dump.joblib")

    # extract and save predictors
    best_vars = name_vars[y][search.best_estimator_.named_steps["rfe"].support_]
    print(len(best_vars))
    pd.DataFrame(best_vars).to_csv(results_path + output_prefix + "_full_best_vars.csv", header=False,
                                   index=False)

    # plot ROC curve for the train set
    print(search.best_score_)
    fpr_train, tpr_train, _ = roc_curve(target_cat_train_full,
                                        search.predict_proba(cnt_data_train_full)[:, 1],
                                        pos_label=search.classes_[1])

    plt.figure()
    plt.plot(fpr_train, tpr_train, color='orange')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Full train set')

    plt.savefig(results_path + output_prefix + "_train_full_auc_roc.pdf")

    # plot ROC curve for the test set
    probs = search.predict_proba(cnt_data_test_full)[:, 1]
    roc_auc_test_full = roc_auc_score(target_cat_test_full, probs)
    print(roc_auc_test_full)
    fpr_test, tpr_test, _ = roc_curve(target_cat_test_full, probs, pos_label=search.classes_[1])

    plt.figure()
    plt.plot(fpr_test, tpr_test, color='orange')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Full test set')

    plt.savefig(results_path + output_prefix + "_test_full_auc_roc.pdf")

    # plot confusion matrix
    plot_confusion_matrix(search, cnt_data_test_full, target_cat_test_full)
    plt.savefig(results_path + output_prefix + "_test_full_conf_mat.pdf")

    # save predicted probabilities for the test set
    if (search.classes_[1] == "1_Sepsis+BldCx+") | (search.classes_[1] == "viral"):
        target_probs = probs
    else:
        target_probs = [1-x for x in probs]

    dict_probs = {'sample_ids': cnt_data_test_full.index.values,
                  'probs': target_probs}
    pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_test_probs.csv")

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

    # performance summary
    res_dict = {'cv_id': [x+1 for x in cvs_id],
                'n_preds': [x["n_preds"] for x in cvs_aucroc_train],
                'roc_auc_train': [x["roc_auc"].round(2) for x in cvs_aucroc_train],
                'roc_auc_test1': [x["roc_auc"].round(2) for x in cvs_aucroc_test1],
                'roc_auc_test2': [x["roc_auc"].round(2) for x in cvs_aucroc_test2]}

    # add meand and std
    res_dict['cv_id'] = res_dict['cv_id'] + ["mean (std)"]
    res_dict['n_preds'] = res_dict['n_preds'] + [""]
    res_dict['roc_auc_train'] = res_dict['roc_auc_train'] + [
        str(np.mean(res_dict['roc_auc_train']).round(2)) + " (" + str(np.std(res_dict['roc_auc_train']).round(2)) + ")"]
    res_dict['roc_auc_test1'] = res_dict['roc_auc_test1'] + [
        str(np.mean(res_dict['roc_auc_test1']).round(2)) + " (" + str(np.std(res_dict['roc_auc_test1']).round(2)) + ")"]
    res_dict['roc_auc_test2'] = res_dict['roc_auc_test2'] + [
        str(np.mean(res_dict['roc_auc_test2']).round(2)) + " (" + str(np.std(res_dict['roc_auc_test2']).round(2)) + ")"]

    # add full model summary
    res_dict['cv_id'] = res_dict['cv_id'] + ["full"]
    res_dict['n_preds'] = res_dict['n_preds'] + [len(best_vars)]
    res_dict['roc_auc_train'] = res_dict['roc_auc_train'] + [search.best_score_.round(2)]
    res_dict['roc_auc_test1'] = res_dict['roc_auc_test1'] + [roc_auc_test_full.round(2)]
    res_dict['roc_auc_test2'] = res_dict['roc_auc_test2'] + [roc_auc_low.round(2)]

    pd.DataFrame(data=res_dict).to_csv(results_path + output_prefix + "_summary_table.csv")