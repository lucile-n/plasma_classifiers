'''
2.build_classifiers_paxgene_complete.py
created on June 11 2021
lucile.neyton@ucsf.edu

This script aims at building classifiers to allocate samples to one of the two
groups (sepsis vs non-sepsis or viral vs non-viral) given gene expression values from PAXgene data.
The final models are tested on a held-out set.

Input files (data folder):
    - CSV-formatted files containing gene counts (samples x genes)
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
from numpy import interp
# 1.1.3
import pandas as pd
# 0.23.2
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# 0.17.0
from joblib import dump, load

# set paths
plasma_data_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_plasma/data/"
paxgene_data_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_VALID/data/"
results_path = "/Users/lucileneyton/OneDrive - University of California, San Francisco/UCSF/EARLI_plasma/results/"

# set params
mode_ = "load"
num_cv = 5
cv_train_test_ = True
test_prop = 0.25

# build the results prefix that will be used to load the relevant files
#results_prefix = '0.1_TRUE_12_virus'
results_prefix = '0.1_TRUE_12vs4_sepsis'

#########################
# DATA LOADING
#########################
# counts data
paxgene_cnt_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_paxgene_complete_cnts.csv", index_col=0)

# plasma meta data (to order main metadata frame)
meta_data = pd.read_csv(plasma_data_path + "processed/" + results_prefix + "_complete_metadata.csv", index_col=0)

# choose the list of DE genes (paxgene or plasma)
#dgea_results = pd.read_csv(results_path + results_prefix + "_paxgene_complete_DGEA_results.csv", index_col=0)
dgea_results = pd.read_csv(results_path + results_prefix + "_paxgene_complete_train_only_DGEA_results.csv", index_col=0)

#########################
# DATA PREPROCESSING
#########################
# drop the gene symbol column
paxgene_cnt_data = paxgene_cnt_data.drop("hgnc_symbol", axis=1)

# list genes used as input
name_vars = paxgene_cnt_data.index.values

# DE genes indexes
y = dgea_results.index.values
y = [name_vars.tolist().index(x) for x in y]

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
#output_prefix = 'paxgene_complete_' + results_prefix + '_bsvm'
output_prefix = 'paxgene_complete_train_only_' + results_prefix + '_bsvm'

# transpose data for classifiers
paxgene_cnt_data = paxgene_cnt_data.T

# identify target data
target_data = paxgene_cnt_data

# target classes for count data
if 'virus' in results_prefix:
    target_cat = meta_data.viruspos
else:
    if 'sepsis' in results_prefix:
        target_cat = meta_data.sepsis_cat

# split data to have a held-out set
cnt_data_train_full, cnt_data_test_full, \
target_cat_train_full, target_cat_test_full = train_test_split(target_data, target_cat,
                                                     test_size=test_prop, random_state=123,
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
cvs_aucroc_test = []

# for each CV set
for i in cvs_id:
    cv_id = str(i+1)

    # split data into train and test sets
    cnt_data_train, cnt_data_test, \
        target_cat_train, target_cat_test = train_test_split(cnt_data_train_full, target_cat_train_full,
                                                             test_size=test_prop,
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

    pd.DataFrame(best_vars).to_csv(results_path + output_prefix + "_" + cv_id + "_full_best_vars.csv", header=False,
                                   index=False)

    # add results to the list
    cvs_aucroc_train.append({"labels": target_cat_train,
                             "probs": search.predict_proba(cnt_data_train)[:, 1],
                             "classes": search.classes_[1],
                             "n_preds": len(best_vars),
                             "roc_auc": roc_auc_score(target_cat_train, search.predict_proba(cnt_data_train)[:, 1])})

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
    cvs_aucroc_test.append({"labels": target_cat_test,
                             "probs": probs,
                             "classes": search.classes_[1],
                             "roc_auc": roc_auc})
    print(roc_auc)

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

#########################
# Performance summary
#########################
# performance summary
res_dict = {'cv_id': [x+1 for x in cvs_id],
            'n_preds': [x["n_preds"] for x in cvs_aucroc_train],
            'roc_auc_train': [x["roc_auc"].round(2) for x in cvs_aucroc_train],
            'roc_auc_test1': [x["roc_auc"].round(2) for x in cvs_aucroc_test]}

# add mean and std
res_dict['cv_id'] = res_dict['cv_id'] + ["mean (std)"]
res_dict['n_preds'] = res_dict['n_preds'] + [""]
res_dict['roc_auc_train'] = res_dict['roc_auc_train'] + [
    str(np.mean(res_dict['roc_auc_train']).round(2)) + " (" + str(np.std(res_dict['roc_auc_train']).round(2)) + ")"]
res_dict['roc_auc_test1'] = res_dict['roc_auc_test1'] + [
    str(np.mean(res_dict['roc_auc_test1']).round(2)) + " (" + str(np.std(res_dict['roc_auc_test1']).round(2)) + ")"]

# add full model summary
res_dict['cv_id'] = res_dict['cv_id'] + ["full"]
res_dict['n_preds'] = res_dict['n_preds'] + [len(best_vars)]
res_dict['roc_auc_train'] = res_dict['roc_auc_train'] + [roc_auc_score(target_cat_train_full, search.predict_proba(cnt_data_train_full)[:, 1]).round(2)]
res_dict['roc_auc_test1'] = res_dict['roc_auc_test1'] + [roc_auc_test_full.round(2)]

pd.DataFrame(data=res_dict).to_csv(results_path + output_prefix + "_summary_table.csv")

#########################
# AUC-ROC curves
#########################
# plot ROC curve for the testing set
tprs_ = []
base_fpr = np.linspace(0, 1, 101)

plt.figure()
for i in cvs_id:
    cv_id = str(i + 1)
    fpr_test, tpr_test, _ = roc_curve(cvs_aucroc_test[i]["labels"],
                                      cvs_aucroc_test[i]["probs"],
                                      pos_label=cvs_aucroc_test[i]["classes"])

    #if i == 0:
        #plt.plot(fpr_test, tpr_test, label='Cross-validation splits, AUC=' +
                                           #str(np.mean([x["roc_auc"].round(2) for x in cvs_aucroc_test]).round(2)) + " (" +
                                           #str(np.std([x["roc_auc"].round(2) for x in cvs_aucroc_test]).round(2)) + ")",
                 #color="red", linewidth=1, alpha=0.3)
    #else:
        #plt.plot(fpr_test, tpr_test, color="red", linewidth=1, alpha=0.3)

    tpr_ = interp(base_fpr, fpr_test, tpr_test)
    tpr_[0] = 0.0
    tprs_.append(tpr_)

tprs_ = np.array(tprs_)
mean_tprs = tprs_.mean(axis=0)
std_tprs = tprs_.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
tprs_lower = mean_tprs - std_tprs

plt.plot(base_fpr, mean_tprs, 'red', label='Cross-validation splits, AUC=' +
                                           str(np.mean([x["roc_auc"].round(2) for x in cvs_aucroc_test]).round(2)) + " (" +
                                           str(np.std([x["roc_auc"].round(2) for x in cvs_aucroc_test]).round(2)) + ")")
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='red', alpha=0.2)

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

# save predicted probabilities for the full test set
if (search.classes_[1] == "1_Sepsis+BldCx+") | (search.classes_[1] == "viral"):
    target_probs = probs
else:
    target_probs = [1 - x for x in probs]

dict_probs = {'sample_ids': cnt_data_test_full.index.values,
              'probs': target_probs}
pd.DataFrame(data=dict_probs).to_csv(results_path + output_prefix + "_test_probs.csv")