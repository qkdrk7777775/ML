import os
os.chdir(r'Y:\cj\발생소산\script')
import shutil
import time
import warnings
from collections import namedtuple
from functools import partial
from itertools import product
from operator import itemgetter
from pathlib import Path

import catboost
import joblib
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost
from sklearn.datasets import make_classification
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, OneClassSVM
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EpochScoring
from torch import nn
from tsfresh.transformers import FeatureAugmenter, FeatureSelector
from ml_get_train_test import get_train_test
import shap
from BorutaShap import BorutaShap, load_data
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from ngboost import NGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


seed= 0
torch.manual_seed(seed)
# %matplotlib auto
def calc_perf(obs_data, pred_data):
    pred_data = np.round(pred_data)
    tn, fp, fn, tp = confusion_matrix(obs_data, pred_data).flatten()
    performance = dict(     ACC=accuracy_score(obs_data, pred_data) * 100,
                            CSI=tp / (tp + fn + fp) * 100,
                            PAG=100 - fp / (tp + fp) * 100,
                            FAR=fp / (tp + fp) * 100,
                            POD=recall_score(obs_data, pred_data) * 100,
                            POFD=fp / (tn + fp) * 100,
                            f1_score=f1_score(obs_data, pred_data),
                            tn=tn,
                            fp=fp,
                            fn=fn,
                            tp=tp)
    return performance


def undersample_seafog_half(x, y):
    rs= np.random.RandomState(0)
    seafog_mask = y == 1
    seafog_idx  = np.flatnonzero(seafog_mask.values)
    seafog_size = seafog_mask.sum()
    seafog_sampler = rs.permutation(seafog_idx)[:seafog_size//2]
    x = x[~seafog_mask].append(x.iloc[seafog_sampler])
    y = y[~seafog_mask].append(y.iloc[seafog_sampler])
    return x,y


def get_model(name):
    if name == 'XT':
        return ExtraTreesClassifier(n_jobs=12, random_state=0)
    if name =='HGB':  
        return HistGradientBoostingClassifier(random_state=0)
    if name =='RF':
        return RandomForestClassifier(n_jobs=12, random_state=0)
    if name =='LGB':
        return lightgbm.LGBMClassifier(n_estimators=500, objective='binary', importance_type='split')


def calc_fi(model, columns):
    # edge case hb, lgb
    try:
        fi = model.feature_importances_
        fi = pd.Series(data=fi, index=columns)
    except:
        fi=model.feature_importances_[0]
        fi = pd.Series(data=fi, index=columns)
    if not np.allclose(fi.sum(), 1., atol=0.1):
        fi = fi / fi.sum()
    return fi

def fit_pred(x_train, y_train, x_test, y_test, task_type='CPU',model=False,model_name='cat'):
    if model_name=='cat':
        model = catboost.CatBoostClassifier(iterations=5000, eval_metric='F1', early_stopping_rounds=10, task_type=task_type, random_state=0)
    elif model_name=='xgb':
        model=XGBClassifier(random_state=0)
    elif model_name=='lgb':
        model=lightgbm.LGBMClassifier(n_estimators=500, objective='binary', importance_type='split',random_state=0)
    elif model_name=='ngb':
        model=NGBClassifier(random_state=0)
    elif model_name=='rf':
        model=RandomForestClassifier(random_state=0)
    elif model_name=='et':
        model=ExtraTreesClassifier(random_state=0)

    x_train, x_val, y_train, y_val = train_test_split(
                                                x_train, y_train, 
                                                test_size=0.2, 
                                                shuffle=True, 
                                                random_state=0
                                                )

    if model_name in ['cat','xgb','lgb','ngb']:
        if model_name!='ngb':
            model.fit(x_train, y_train,
                    eval_set=[(x_val,y_val)], 
                    early_stopping_rounds=50, 
                    )
        elif model_name=='ngb':
            model.fit(x_train, np.array(y_train.values,dtype=np.int),
                    X_val=x_val,Y_val=np.array(y_val.values,dtype=np.int), 
                    early_stopping_rounds=50, 
                    )
    else:
        model.fit(x_train, y_train)

    pred= model.predict_proba(x_val)[:,1]
#    fraction_of_positives, mean_predicted_value = calibration_curve(y_val, pred, n_bins=10)# 10개 구간 기준의 curve
#    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    
    model2=CalibratedClassifierCV(model,cv=2,method='sigmoid')
    model2=model2.fit(x_train,y_train)
    pred=model2.predict(x_test)
#    fraction_of_positives, mean_predicted_value = calibration_curve(y_val, pred, n_bins=10)# 10개 구간 기준의 curve
#    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    
    obs_pred = y_test.to_frame(name='obs').assign(pred=pred)
    
    perf = calc_perf(y_test, pred)
    fi = calc_fi(model, columns=x_train.columns)
    if model:
        return perf, obs_pred, fi, model
    else:
        return perf, obs_pred, fi

# pred_hour_list = [1,3,6]
# multiplier_list= [1,2,3]
# train_threshold_list=[1100,1500, 2000, 2500, 3000]
pred_hour_list = [1,3,6,9,12,24]
multiplier_list= [1]
train_threshold_list=[1000]
Dimension = namedtuple('Dimension', 'pred_hour, multiplier, train_threshold')
model_names= ['xgb', 'cat', 'lgb', 'rf', 'et']
dimension_result = {}
for d in product(pred_hour_list, multiplier_list, train_threshold_list):
    for ml_name in model_names:
        dimension = Dimension(*d)
        shift_hour = dimension.pred_hour
        multiplier = dimension.multiplier
        train_threshold = dimension.train_threshold
        print(shift_hour, 'started')
        x_train,y_train,x_test,y_test = get_train_test(shift_hour=shift_hour, multiplier=multiplier, train_threshold=train_threshold)
        # x_test, y_test = undersample_seafog_half(x_test,y_test)
        perf, obs_pred, fi, _ = fit_pred(x_train,y_train,x_test,y_test, task_type='GPU', model=True,model_name=ml_name)
        dimension_result[dimension] = {'perf':perf, 'obs_pred':obs_pred, 'fi':fi,'model':ml_name}
        
        obs_pred.to_pickle(f'../product/calibration/obs_pred/{ml_name}_{shift_hour}.pkl', protocol=4)
        fi.to_pickle(f'../product/calibration/fi/{ml_name}_{shift_hour}.pkl', protocol=4)

else:
    m = 'train ended\n'
    print(m*10)

dimension_result = {}
for d in product(pred_hour_list, multiplier_list, train_threshold_list):
    dimension = Dimension(*d)
    shift_hour = dimension.pred_hour
    multiplier = dimension.multiplier
    train_threshold = dimension.train_threshold
    os.chdir(r'Y:\cj\발생소산\input')
    x_train,y_train,x_test,y_test = get_train_test(shift_hour=shift_hour, multiplier=multiplier, train_threshold=train_threshold)
    perf = calc_perf(y_test, pd.read_pickle(rf'Y:/cj/발생소산/product/calibration/obs_pred/rf_{shift_hour}.pkl')['pred'])
    dimension_result[dimension]={'perf':perf}

dimension_perfs = {k:v['perf'] for k,v in dimension_result.items()}
dimension_perfs = pd.DataFrame(dimension_perfs).T
dimension_perfs.to_clipboard()


# for model in model_list:
#     pass
    # results.append(dict(model=model, perf=perf, fi=fi, pred=pred, score=score))
    # best_preds = [x['pred'] for x in sorted(results, key=lambda x: x['score'], reverse=True)[:4]]
    # ensemble = pd.concat(best_preds, axis=1).mean(axis=1).round()
    # ensemble_perf = get_performance(y_test, ensemble, 'ensemble')
    # result = {}
    # result['perf']  = ensemble_perf
    # dimension_result[dimension] = result
# else:


# perf_map = pd.DataFrame({k:v['perf'] for k,v in dimension_result.items()}).T
# perf_map.to_clipboard()
# raise
# perf_map = {tuple(k):v['perf'] for k,v in dimension_result.items()}
# perf_df = pd.DataFrame(perf_map).T
# names = dimension._asdict().keys()
# perf_df.index = perf_df.index.set_names(names)
# perf_df = perf_df.reset_index()
# perf_df.to_clipboard()
# test_pred_map = {k:v['test_pred'] for k,v in dimension_result.items()}
# %matplotlib auto
# cb = _
# loc_iloc_map = pd.Series(np.arange(y_train.size), index=y_train.index)
# shap.initjs()
# explainer = shap.TreeExplainer(cb)
# shap_values = explainer.shap_values(x_train, y_train)

# spot = (y_train==1).idxmax()
# ispot = loc_iloc_map[spot]
# # shap.force_plot(explainer.expected_value, shap_values[ispot,:], x_train.loc[spot,:], y_train.loc[spot])
# shap.summary_plot(shap_values, x_train, y_train, plot_type="bar")
# shap.force_plot(explainer.expected_value, shap_values, x_train, y_train)
# shap.summary_plot(shap_values, x_train, y_train)

