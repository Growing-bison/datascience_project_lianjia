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

'''Featurn engineer function
Environment
    python 3.6
    matplotlib2.0.2
    numpy 1.12.1
    seaborn 0.7.1
    scipy 0.19.1
    scikit-learn 0.19.0
    lightgbm 2.1.2
    py-xgboost 0.60

This is for finishing feature engineer.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'  # 解决负号是方块
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder


class Abnormal_value_handle():
    '''
    异常值处理
    '''
    def plot_abnormalvalue(self, dataset_used1):
        fig, ax = plt.subplots()
        ax.scatter(x=dataset_used1.ix[:, 'size_house_edit1'], y=dataset_used1.ix[:, 'smeter_price_edit1'])
        plt.ylabel('SalePrice', fontsize=13)
        plt.xlabel('GrLivArea', fontsize=13)
        plt.show()

        return None

    def handle_abnormalvalue(self, dataset_used1):
        '''
        删除离群点
        :param dataset_used1: pd.DataFrame
        :return: pd.DataFrame
        '''
        dataset_used1 = dataset_used1.drop(dataset_used1[(dataset_used1['size_house_edit1'] > 1900) & (
        dataset_used1['smeter_price_edit1'] < 1250000)].index)
        fig, ax = plt.subplots()
        ax.scatter(x=dataset_used1.ix[:, 'size_house_edit1'], y=dataset_used1.ix[:, 'smeter_price_edit1'])
        plt.ylabel('SalePrice', fontsize=13)
        plt.xlabel('GrLivArea', fontsize=13)
        plt.show()

        return dataset_used1


class Pred_feature_handle():
    '''
    目标变量处理
    '''
    def goalvalue_check_normal(self, dataset_used1):
        '''
        目标变量处理——满足整体分布
        目标值处理：线性的模型需要正态分布的目标值才能发挥最大的作用。
        我们需要检测房价什么时候偏离正态分布。使用probplot函数，即正态概率图：
        :param dataset_used1:
        :return: None
        '''
        # 绘制正态分布图
        fig5 = plt.figure(figsize=(6, 6))
        sns.distplot(dataset_used1['smeter_price_edit1'], fit=norm)
        # 正态分布拟合
        (mu, sigma) = norm.fit(dataset_used1['smeter_price_edit1'])
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

        # 绘制QQ图  看是否与理论的一致
        fig5 = plt.figure(figsize=(6, 6))

        # 绘图
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        # 原始数据分布绘图
        res = stats.probplot(dataset_used1['smeter_price_edit1'], plot=plt)
        plt.show()

        return None

    def goalvalue_handle_normal(self, dataset_used1):
        '''
        目标变量不满足正态分布情况下的变换处理
        :param dataset_used1: pd.DataFrame
        :return dataset_used1: pd.DataFrame
        '''
        # 使用log1p函数完成log(1+x)变换
        dataset_used1['smeter_price_edit1'] = np.log1p(dataset_used1['smeter_price_edit1'])

        # 正态分布拟合
        (mu, sigma) = norm.fit(dataset_used1['smeter_price_edit1'])
        fig6 = plt.figure(figsize=(6, 6))

        # 绘图
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        # log变换之后的数据分布绘图
        res = stats.probplot(dataset_used1['smeter_price_edit1'], plot=plt)
        plt.show()

        return dataset_used1


class Missing_value_handle():
    '''
    缺失值处理
    '''
    def missingvalue_check(self, dataset_used1):
        '''
        检查各个列变量中的缺失值情况。
        由观察得知缺失变量
        unit_house
        years_house_year_edit1,num
        years_house_type_edit1
        type_house_edit1
        direction_edit1
        decoration_edit1
        elevator_edit1
        watch_time_edit1,num
        submit_period_edit1,num
        interests_house_edit1,num
        :param dataset_used1: pd.DataFrame
        :return: None
        '''
        dataset_used1.info()

        # Percent missing data by feature-连续(-1)
        all_data_na3 = (dataset_used1[dataset_used1 == -1].sum() / len(dataset_used1)) * 100
        all_data_na3 = all_data_na3.drop(all_data_na3[all_data_na3 == 0].index).sort_values(ascending=False)[:30]
        f, ax = plt.subplots(figsize=(8, 8))
        plt.xticks(rotation='90')
        sns.barplot(x=all_data_na3.index, y=all_data_na3)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)

        # Percent missing data by feature-类别
        temp1 = dataset_used1.dtypes
        temp2 = temp1[temp1 == 'object'].index
        temp3 = dataset_used1[temp2] == 'nodata'
        all_data_na2 = (temp3.sum() / len(dataset_used1[temp2])) * 100
        all_data_na2 = all_data_na2.drop(all_data_na2[all_data_na2 == 0].index).sort_values(ascending=False)[:30]
        f, ax = plt.subplots(figsize=(8, 8))
        plt.xticks(rotation='90')
        sns.barplot(x=all_data_na2.index, y=all_data_na2)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)

        # Percent missing data by feature-连续（0）
        all_data_na = (dataset_used1.isnull().sum() / len(dataset_used1)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        f, ax = plt.subplots(figsize=(8, 8))
        plt.xticks(rotation='90')
        sns.barplot(x=all_data_na.index, y=all_data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        return None

    def missingvalue_handle(self, dataset_used1):
        '''
        处理缺失值
        :param dataset_used1: pd.DataFrame
        :return dataset_used1: pd.DataFrame
        '''
        dataset_used1['unit_house'] = dataset_used1['unit_house'].fillna('nodata')
        dataset_used1['years_house_year_edit1'] = dataset_used1['years_house_year_edit1'].fillna(0)
        # 关于年份的处理，将其处理成距今2018的连续型年值
        dataset_used1['years_house_year_edit2'] = dataset_used1['years_house_year_edit1'].apply(
            lambda x: 0.0 if x == 0.0 else (2018.0 - x))

        return dataset_used1


class Other_feature_engineer():
    '''
    其他特征工程
    '''
    def feature_eg_other(self, dataset_used1, threshold=0.75):
        # 1、有许多特征实际上是类别型的特征，但给出来的是数字，所以需要将其转换成类别型。
        dataset_used1['years_house_year_edit1'] = dataset_used1['years_house_year_edit1'].astype(int).apply(str)

        # 2、接下来 LabelEncoder，对部分类别的特征进行编号。
        temp1 =dataset_used1.dtypes
        temp2 = temp1[temp1=='object'].index
        # 使用LabelEncoder做变换
        for c in temp2:
            lbl = LabelEncoder()
            lbl.fit(list(dataset_used1[c].unique()))
            dataset_used1[c] = lbl.transform(list(dataset_used1[c].values))

        # 3、检查变量的正态分布情况
        ###### 检查
        total_price = dataset_used1['total_price']
        dataset_used1.drop('total_price', axis=1, inplace=True)
        numeric_feats = ['size_house_edit1', 'watch_time_edit1', 'interests_house_edit1', 'submit_period_edit1']

        # 对所有数值型的特征都计算skew，即计算一下偏度
        skewed_feats = dataset_used1[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness.head()

        ###### 变换处理
        skewness = skewness[abs(skewness) > threshold]  # 关于临界值，如何定，不知？？
        print("总共有 {} 数值型的特征做变换".format(skewness.shape[0]))

        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            #all_data[feat] += 1
            dataset_used1[feat] = boxcox1p(dataset_used1[feat], lam)

        ###### 哑变量处理
        temp1 = dataset_used1.dtypes
        temp2 = temp1[temp1 == 'int64'].index
        for name in temp2:
            dataset_used1[name] = dataset_used1[name].astype(str)

        temp2_2 = temp1[temp1 == 'float64'].index
        for name in temp2_2:
            dataset_used1[name] = dataset_used1[name].astype(float)

        temp_ds_use1 = dataset_used1.drop(['community_house', 'years_house_year_edit1'], axis=1)
        all_usedata = pd.get_dummies(temp_ds_use1)
        useful_dataset = all_usedata.sample(frac=0.1, random_state=123)

        all_usedata = None
        del all_usedata, temp_ds_use1, dataset_used1

        return useful_dataset




