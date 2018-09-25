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

'''Main function
Environment
    python 3.6
    matplotlib2.0.2
    numpy 1.12.1
    seaborn 0.7.1
    scipy 0.19.1
    scikit-learn 0.19.0
    lightgbm 2.1.2
    py-xgboost 0.60

This is for finishing whole data mining program.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from data_processing import main_apply_func, main_check_func
from analysis import Plot_analysis
from feature_engineer import Abnormal_value_handle, Pred_feature_handle, Missing_value_handle, Other_feature_engineer
from modelling import prepare_data, rmsle_cv, Model_build, AverageModels, StackingAverageModels

### 0、数据导入与查看  ###
dataset = pd.read_csv('./houseInfo.csv')

dataset.info()
dataset.describe()
# 检查数据维度
print("训练集特征前的size：",dataset.shape)

### 1、数据处理  ###
dataset = main_apply_func(dataset)
main_check_func(dataset)

feature1 = ['community_house','unit_house','size_house_edit1','size_house_edit1_addcata','watch_time_edit1','watch_time_edit1_addcata','interests_house_edit1','interests_house_edit1_addcata','submit_period_edit1','submit_period_edit1_addcata','years_period_edit1','tax_free_edit1','total_price','smeter_price_edit1',
            'direction_edit1','decoration_edit1','elevator_edit1','type_house_edit1','years_house_type_edit1','years_house_year_edit1','region']
dataset_used1 = dataset[feature1]

### 2、绘图分析  ###
plot_analysit = Plot_analysis()

# ##### 面积、查看次数、收藏次数、发布时间
# 房屋面积：size_house_edit1_addcata
# 查看次数：watch_time_edit1_addcata
# 感兴趣人数：interests_house_edit1_addcata
# 多久前发布：submit_period_edit1_addcata

plot_analysit.plot_single_var1(dataset_used1)
plot_analysit.plot_single_var2(dataset_used1)
plot_analysit.plot_single_var3(dataset_used1)
plot_analysit.plot_single_var4(dataset_used1)
plot_analysit.plot_single_var5(dataset_used1)

# ##### 产权和查看次数、收藏次数
# 2年产权的查看次数、收藏次数
# 5年产权的产看次数、收藏次数
# 2室1厅户型的具有2年产权查看次数
# 2室1厅户型的具有5年产权查看次数
# 产权是满足5年：tax_free_edit1
# 产权是否满2年：years_period_edit1
# 房屋户型：unit_house
# 收藏次数watch_time_edit1 感兴趣人数：interests_house_edit1

plot_analysit.plot_multi_var1(dataset_used1)
plot_analysit.plot_multi_var2(dataset_used1)

# #### 关联分析
# 2年产权、5年产权vs装修程度、户型
plot_analysit.plot_rele_var1(dataset_used1)

# 电梯、楼层、楼型、建成时间单变量统计
plot_analysit.plot_rele_var2(dataset_used1)

# 连续变量的相关性
plot_analysit.plot_check_rele(dataset_used1)


### 3、特征工程  ###

# 异常值检查
abnormalvalue = Abnormal_value_handle()
abnormalvalue.plot_abnormalvalue(dataset_used1)
dataset_used1 = abnormalvalue.handle_abnormalvalue(dataset_used1)

# 目标变量处理
pred_feature_handle = Pred_feature_handle()
pred_feature_handle.goalvalue_check_normal(dataset_used1)
dataset_used1 = pred_feature_handle.goalvalue_handle_normal(dataset_used1)

# 缺失值处理
missingvalue = Missing_value_handle()
missingvalue.missingvalue_check(dataset_used1)
dataset_used1 = missingvalue.missingvalue_handle(dataset_used1)

# 其他特征工程
other_feature_engineer = Other_feature_engineer()
useful_dataset = other_feature_engineer.feature_eg_other(dataset_used1)



### 4、建模  ###
train_x, test_x, train_y, test_y = prepare_data(useful_dataset, pred_feature='smeter_price_edit1')

# 单个模型
model_build = Model_build()
model_build.lasso()
model_build.ENet()
model_build.GBoost()
model_xgb = model_build.model_xgb()
model_lgb = model_build.model_lgb()

# 单个模型融合
model_base_multi = AverageModels()
average_models = model_base_multi.apply_model()

# 单个模型融合改善1-stacking
model_base_multi_impov = StackingAverageModels()
stacked_averaged_model = model_base_multi_impov.apply_model()

### 5、测试模型融合  ###

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# stacking
stacked_averaged_model.fit(train_x, train_y)
stacked_train_pred = stacked_averaged_model.predict(train_x)
stacked_pred = np.expm1(stacked_averaged_model.predict(test_x))
print(rmsle(train_y, stacked_train_pred))
print(rmsle(test_y, stacked_pred))

# xgboost
model_xgb.fit(train_x, train_y)
xgb_train_pred = model_xgb.predict(train_x)
xgb_pred = np.expm1(model_xgb.predict(test_x))
print(rmsle(train_y, xgb_train_pred))
print(rmsle(test_y, xgb_pred))


# lightgbm
model_lgb.fit(train_x, train_y)
gbm_train_pred = model_lgb.predict(train_x)
gbm_pred = np.expm1(model_lgb.predict(test_x))
print(rmsle(train_y, gbm_train_pred))
print(rmsle(test_y, gbm_pred))
