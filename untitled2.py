# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:59:13 2020

@author: cjcho
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from BorutaShap import BorutaShap, load_data
Feature_Selector = BorutaShap(importance_measure='shap',
                              classification=False)
X, y = load_data(data_type='regression')
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
#Boruta Shap plot
Feature_Selector.fit(X=X, y=y, n_trials=100, random_state=0)
Feature_Selector.plot(which_features='all')


from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb
lgb_train=lgb.Dataset(X_train,Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
from catboost import CatBoostRegressor
model = CatBoostRegressor(iterations=2,learning_rate=1,depth=2)
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from xgboost import XGBRegressor
model=XGBRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from lightgbm import LGBMRegressor
model=LGBMRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from sklearn.ensemble import IsolationForest
model=IsolationForest()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

#안되는거 목록


from sklearn.ensemble import RandomTreesEmbedding



from ngboost import NGBRegressor
model=NGBRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
estimators = [('lr', RidgeCV()),('svr', LinearSVR(random_state=42))]
model= StackingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=10,random_state=42))
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=10, random_state=1)
model= VotingRegressor([('lr', r1), ('rf', r2)])
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')


from sklearn.ensemble import AdaBoostRegressor
model=AdaBoostRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=10, random_state=1)
Feature_Selector.plot(which_features='all')

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
model=HistGradientBoostingClasser()
Feature_Selector = BorutaShap(model=model ,importance_measure='shap',classification=False)
Feature_Selector.fit(X=X, y=y, n_trials=100, random_state=1)
Feature_Selector.plot(which_features='all')



model.fit(X_train, Y_train)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)



X, Y = load_boston(True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print('Test NLL', test_NLL)

import shap
shap.initjs()
explainer = shap.TreeExplainer(ngb, model_output=1)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=load_boston()['feature_names'])


test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
feature_importance_loc = ngb.feature_importances_[0]
feature_importance_scale = ngb.feature_importances_[1]

df_loc = pd.DataFrame({'feature':load_boston()['feature_names'], 
                       'importance':feature_importance_loc})\
    .sort_values('importance',ascending=False)
df_scale = pd.DataFrame({'feature':load_boston()['feature_names'], 
                       'importance':feature_importance_scale})\
    .sort_values('importance',ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))
fig.suptitle("Feature importance plot for distribution parameters", fontsize=17)
sns.barplot(x='importance',y='feature',ax=ax1,data=df_loc, color="skyblue").set_title('loc param')
sns.barplot(x='importance',y='feature',ax=ax2,data=df_scale, color="skyblue").set_title('scale param')

#