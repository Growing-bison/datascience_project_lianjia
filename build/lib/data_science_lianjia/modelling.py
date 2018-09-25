# coding: utf-8
# Copyright (c) 2018-present, Qikun Lu..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

'''Modelling function
Environment
    python 3.6
    matplotlib2.0.2
    numpy 1.12.1
    seaborn 0.7.1
    scipy 0.19.1
    scikit-learn 0.19.0
    lightgbm 2.1.2
    py-xgboost 0.60

This is for modelling.
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# 数据准备
def prepare_data(useful_dataset, pred_feature='smeter_price_edit1'):
    X = useful_dataset.drop(pred_feature, axis=1)
    y = useful_dataset[pred_feature]

    # 注意这一步！！数据结果与类型转换
    X = X.as_matrix().astype(np.float)
    y = y.as_matrix().astype(np.float)

    # 训练集与测试集划分
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

    return train_x, test_x, train_y, test_y

# 交叉验证函数
def rmsle_cv(model, train_x, train_y, n_folds = 5):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x)
    rmse = np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

# 单模型
class Model_build():
    def lasso(self):
        # ###### lasso模型
        lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.5, random_state=1))
        score = rmsle_cv(lasso)
        print("\nLasso 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    def ENet(self):
        # ###### ENet模型
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.5, l1_ratio=.9, random_state=3))
        score = rmsle_cv(ENet)
        print("\ENet 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    def KRR(self):
        # ###### KRR模型
        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        score = rmsle_cv(KRR)
        print("\nKernel Ridge 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    def GBoost(self):
        # ###### GBoost模型
        GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
                                           max_depth=6, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        score = rmsle_cv(GBoost)
        print("Gradient Boosting 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    def model_xgb(self):
        # ###### xgboost模型
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                     learning_rate=0.1, max_depth=6,
                                     min_child_weight=1.7817, n_estimators=1000,
                                     reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, silent=1,
                                     seed=7, nthread=-1)
        score = rmsle_cv(model_xgb)
        print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        return model_xgb
    def model_lgb(self):
        # ###### LightGBM模型
        model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin=55, bagging_fraction=0.8,
                                      bagging_freq=5, feature_fraction=0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
        score = rmsle_cv(model_lgb)
        print("lightgbm 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        return model_lgb

# 基模型融合
class AverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 遍历所有模型
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    # 预估，并对预估结果做average
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

    def apply_model(self):
        lasso = Model_build().lasso()
        ENet = Model_build().ENet()
        GBoost = Model_build().GBoost()

        average_models = AverageModels(models=(lasso, ENet, GBoost))
        score = rmsle_cv(average_models)
        print('对基模型进行集成之后的得分：{:.4f} ({:.4f})\n'.format(score.mean(), score.std()))

        return average_models

# 模型深层融合-构建stacking averagd models的类
class StackingAverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 遍历拟合原始模型
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=123)

        # 得到基模型之后，对out_of_fold的数据做预估，并为学习stacking的第二层做数据准备
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)), dtype=np.float64)

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 学习stacking模型
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 做stacking预估
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])

        return self.meta_model_.predict(meta_features)

    def apply_model(self):
        lasso = Model_build().lasso()
        ENet = Model_build().ENet()
        GBoost = Model_build().GBoost()

        stacked_averaged_model = StackingAverageModels(base_models=(ENet, GBoost), meta_model=lasso, n_folds=5)
        score = rmsle_cv(stacked_averaged_model)
        print('对基模型进行集成之后的得分：{:.4f} ({:.4f})\n'.format(score.mean(), score.std()))

        return stacked_averaged_model

