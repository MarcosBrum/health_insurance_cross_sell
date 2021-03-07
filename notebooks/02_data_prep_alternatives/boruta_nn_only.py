"""
0.0 IMPORTS
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

from boruta import BorutaPy

from IPython.display import display

import pickle


"""
Feature Selection
"""
home_path = "/home/marcos/Documentos/comunidade_DS/pa004_health_insurance_cross_sell/"
df_train_res = pd.read_pickle(home_path + 'interim/df_train_res_nn.pkl')


"""
DataFrame contains only train data.
"""


X_train_res = df_train_res.drop(['id', 'response'], axis=1)
y_train_res = df_train_res.response


"""
6.2 Boruta as Feature selector
"""


rf = RandomForestRegressor(n_jobs=-1)


"""
The Boruta selector only takes arrays as input
"""


X_train_arr = X_train_res.to_numpy()
y_train_arr = y_train_res.ravel()

# instantiate Boruta
boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42).fit(X_train_arr, y_train_arr)

# list of indexes of selected features
cols_selected = boruta.support_.tolist()

# best features
cols_selected_boruta_resampled = X_train_res.iloc[:, cols_selected].columns.to_list()

# not selected
cols_not_selected_boruta_resampled = list(np.setdiff1d(X_train_res.columns, cols_selected_boruta_resampled))

# export list of selected features
pickle.dump(cols_selected_boruta_resampled, open(home_path + "interim/cols_selected_boruta_resampled_nn.pkl", "wb"))

display(cols_selected_boruta_resampled)
