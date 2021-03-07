import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit as sss

from imblearn.ensemble import BalancedRandomForestClassifier as bal_rf

"""## 0.1 Load data"""

## 0.1 Load data

df_train_res = pd.read_pickle("data_in_progress/df_train_res_nn.pkl")

cols_selected_boruta_resampled = pickle.load(open("data_in_progress/cols_selected_boruta_resampled_nn.pkl", "rb"))

resp = ['response']
cols_selected_boruta_resampled_full = cols_selected_boruta_resampled.copy()
cols_selected_boruta_resampled_full.extend(resp)

df7 = df_train_res[cols_selected_boruta_resampled_full].copy()

"""## 0.2 Helper functions"""

## 0.2 Helper Functions

def metric_scores(y_true, y_pred):
    return pd.DataFrame({'accuracy':accuracy_score(y_true, y_pred),
                        'balanced_accuracy':balanced_accuracy_score(y_true, y_pred),
                        'precision':precision_score(y_true, y_pred),
                        'precision_0':precision_score(y_true, y_pred, pos_label=0),
                        'recall':recall_score(y_true, y_pred),
                        'specificity':recall_score(y_true, y_pred, pos_label=0),
                        'F1':f1_score(y_true, y_pred),
                        'F1_weighted':f1_score(y_true, y_pred, average='weighted'),
                        'G_mean':np.sqrt(recall_score(y_true, y_pred)*recall_score(y_true, y_pred, pos_label=0))},
                        index=[0])


def cross_validation(training_data, kfolds, model, model_name, verbose=False):
    xtraining = training_data.drop(['response'], axis=1)
    ytraining = training_data.response

    cv = sss(n_splits=kfolds)
    acc_list = []
    bal_acc_list = []
    prec_list = []
    prec0_list = []
    rec_list = []
    spec_list = []
    f1_list = []
    f1w_list = []
    g_list = []
    for train_index, prim_val_index in cv.split(xtraining, ytraining):
        X_training, X_prim_val = xtraining.iloc[train_index], xtraining.iloc[prim_val_index]
        y_training, y_prim_val = ytraining.iloc[train_index], ytraining.iloc[prim_val_index]

        m = model.fit(X_training, y_training)
        yhat = m.predict(X_prim_val)

        score_table = metric_scores(y_prim_val, yhat)
        acc_list.append(score_table['accuracy'])
        bal_acc_list.append(score_table['balanced_accuracy'])
        prec_list.append(score_table['precision'])
        prec0_list.append(score_table['precision_0'])
        rec_list.append(score_table['recall'])
        spec_list.append(score_table['specificity'])
        f1_list.append(score_table['F1'])
        f1w_list.append(score_table['F1_weighted'])
        g_list.append(score_table['G_mean'])

    acc_pred = np.round(np.mean(acc_list), 4).astype(str) + '+/-' + np.round(np.std(acc_list), 4).astype(str)
    bal_acc_pred = np.round(np.mean(bal_acc_list), 4).astype(str) + '+/-' + np.round(np.std(bal_acc_list), 4).astype(str)
    prec_pred = np.round(np.mean(prec_list), 4).astype(str) + '+/-' + np.round(np.std(prec_list), 4).astype(str)
    prec0_pred = np.round(np.mean(prec0_list), 4).astype(str) + '+/-' + np.round(np.std(prec0_list), 4).astype(str)
    rec_pred = np.round(np.mean(rec_list), 4).astype(str) + '+/-' + np.round(np.std(rec_list), 4).astype(str)
    spec_pred = np.round(np.mean(spec_list), 4).astype(str) + '+/-' + np.round(np.std(spec_list), 4).astype(str)
    f1_pred = np.round(np.mean(f1_list), 4).astype(str) + '+/-' + np.round(np.std(f1_list), 4).astype(str)
    f1w_pred = np.round(np.mean(f1w_list), 4).astype(str) + '+/-' + np.round(np.std(f1w_list), 4).astype(str)
    g_pred = np.round(np.mean(g_list), 4).astype(str) + '+/-' + np.round(np.std(g_list), 4).astype(str)
    return pd.DataFrame({'Model name':model_name,
                         'accuracy':acc_pred,
                         'balanced_accuracy':bal_acc_pred,
                         'precision':prec_pred,
                         'precision_0':prec0_pred,
                         'recall':rec_pred,
                         'specificity':spec_pred,
                         'F1':f1_pred,
                         'F1_weighted':f1w_pred,
                         'G_mean':g_pred}, index=[0])

"""# 7.0 Machine Learning Models"""

# Balanced Random Forest Classifier
model = bal_rf(n_estimators = 100, max_depth=10, random_state=42, n_jobs=-1)
bal_rf_cv = cross_validation(df7, 5, model, 'Balanced Random Forest Classifier')
bal_rf_cv

bal_rf_cv.to_pickle('bal_rf_cv.pkl')
