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

'''Data analysis function
Environment
    python 3.6
    matplotlib2.0.2
    numpy 1.12.1
    seaborn 0.7.1
    scipy 0.19.1
    scikit-learn 0.19.0
    lightgbm 2.1.2
    py-xgboost 0.60

This is for finishing data analysis.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'  # 解决负号是方块
color = sns.color_palette()
sns.set_style('darkgrid')
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

class Plot_analysis():

    # 单变量观察1
    def plot_single_var1(self, dataset_used1):
        '''
        面积、查看次数、收藏次数、发布时间的变量观察
        :param dataset_used1: pd.DataFrame
        :return: None
        '''
        fig1 = plt.figure(figsize=(10,10))
        fig1.add_subplot(221)
        fg1 = sns.distplot(dataset_used1[dataset_used1['size_house_edit1_addcata']=='0']['size_house_edit1'],bins=1000,kde=False,color='b')
        fg1.set(xlim=(0,500))
        plt.title('size')

        fig1.add_subplot(222)
        fg2 = sns.distplot(dataset_used1[dataset_used1['watch_time_edit1_addcata']=='0']['watch_time_edit1'],bins=500,kde=False,color='b')
        fg2.set(xlim=(0,50))
        plt.title('watch_time')

        fig1.add_subplot(223)
        fg3 = sns.distplot(dataset_used1[dataset_used1['interests_house_edit1_addcata']=='0']['interests_house_edit1'],bins=1000,kde=False,color='b')
        fg3.set(xlim=(0,500))
        plt.title('interests_house')

        fig1.add_subplot(224)
        fg4 = sns.distplot(dataset_used1[dataset_used1['submit_period_edit1_addcata']=='0']['submit_period_edit1'],bins=200,kde=False,color='b')
        fg4.set(xlim=(0,100))
        plt.title('submit_period_edit1')

    # 单变量观察2
    def plot_single_var2(self, dataset_used1):
        '''
        2年产权、5年产权、房屋户型单变量观察
        :param dataset_used1: pd.DataFrame
        :return: None
     '''
        fig2_1 = plt.figure(figsize=(10, 6))
        ax3 = fig2_1.add_subplot(221)

        dataset_used1['tax_free_edit1'].value_counts().plot(kind='bar')
        plt.title('tax_free')

        ax2 = fig2_1.add_subplot(222)
        dataset_used1['years_period_edit1'].value_counts().plot(kind='bar')
        plt.title('years_period')

        ax1 = fig2_1.add_subplot(212)
        ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit

        dataset_used1['unit_house'].value_counts().plot(kind='bar')
        plt.title('unit_house')

        return None

    # 单变量观察3
    def plot_single_var3(self, dataset_used1):
        '''
        总价钱total_price分布的观察
        :param dataset_used1:pd.DataFrame
        :return:None
        '''
        fig2 = plt.figure(figsize=(6, 6))
        fig2.add_subplot(111)
        fg1 = sns.distplot(dataset_used1['total_price'], bins=1500, kde=False, color='b')
        fg1.set(xlim=(0, 2000))
        plt.title('total_price')

        return None

    # 单变量观察4
    def plot_single_var4(self, dataset_used1):
        '''
        装修程度、电梯配备、楼层位置、楼型
        :param dataset_used1: pd.DataFrame
        :return: None
        '''

        fig3 = plt.figure(figsize=(10, 10))
        fig3.add_subplot(221)
        dataset_used1['decoration_edit1'].value_counts().plot(kind='bar')
        plt.title('decoration_edit1')

        fig3.add_subplot(222)
        dataset_used1['elevator_edit1'].value_counts().plot(kind='bar')
        plt.title('elevator_edit1')

        fig3.add_subplot(223)
        dataset_used1['type_house_edit1'].value_counts().plot(kind='bar')
        plt.title('type_house_edit1')

        fig3.add_subplot(224)
        dataset_used1['years_house_type_edit1'].value_counts().plot(kind='bar')
        plt.title('years_house_type_edit1')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        return None

    # 单变量观察5
    def plot_single_var5(self, dataset_used1):
        '''
        房屋建造年代years_house_year_edit1的观察
        :param dataset_used1: pd.DataFrame
        :return: None
        '''
        fig31 = plt.figure(figsize=(10, 6))
        fig31.add_subplot(111)
        dataset_used1['years_house_year_edit1'].value_counts().plot(kind='bar')
        plt.title('years_house_year_edit1')

        return None

    # 多变量观察1
    def plot_multi_var1(self, dataset_used1):
        '''
        产权VS查看次数、收藏次数
        :param dataset_used1: pd.DataFrame
        :return: None
        '''
        cond_dataset1 = dataset_used1[dataset_used1['years_period_edit1'] == '0']['watch_time_edit1'].value_counts()
        fig42 = plt.figure(figsize=(12, 12))
        fig42.add_subplot(221)
        fg1 = sns.distplot(cond_dataset1[1:], bins=200, kde=False, color='b')
        fg1.set(xlim=(0, 400))
        plt.title('产权是满2年的收藏次数')

        cond_dataset2 = dataset_used1[dataset_used1['years_period_edit1'] == '0']['interests_house_edit1'].value_counts()
        fig42.add_subplot(222)
        fg1 = sns.distplot(cond_dataset2[1:], bins=200, kde=False, color='b')
        fg1.set(xlim=(0, 200))
        plt.title('产权是满2年的感兴趣人数')

        cond_dataset3 = dataset_used1[dataset_used1['tax_free_edit1'] == '0']['watch_time_edit1'].value_counts()
        fig42.add_subplot(223)
        fg1 = sns.distplot(cond_dataset1[1:], bins=200, kde=False, color='b')
        fg1.set(xlim=(0, 400))
        plt.title('产权是满5年的收藏次数')

        cond_dataset4 = dataset_used1[dataset_used1['tax_free_edit1'] == '0']['interests_house_edit1'].value_counts()
        fig42.add_subplot(224)
        fg1 = sns.distplot(cond_dataset2[1:], bins=200, kde=False, color='b')
        fg1.set(xlim=(0, 200))
        plt.title('产权是满5年的感兴趣人数')
        return None

    # 多变量观察2
    def plot_multi_var2(self, dataset_used1):
        '''
        户型+产权vs查看次数、收藏次数
        :param dataset_used1:pd.DataFrame
        :return: None
        '''
        cond_dataset1 = dataset_used1[(dataset_used1['unit_house'] == '2室1厅') &
                                      (dataset_used1['years_period_edit1'] == '0')]['watch_time_edit1'].value_counts()
        fig42 = plt.figure(figsize=(16, 8))
        fig42.add_subplot(121)
        fg1 = sns.barplot(x=cond_dataset1[1:].index, y=cond_dataset1[1:].values)
        # fg1.set(xlim=(0,400))
        plt.title('产权是满2年的收藏次数')

        cond_dataset2 = \
        dataset_used1[(dataset_used1['unit_house'] == '2室1厅') & (dataset_used1['years_period_edit1'] == '0')][
            'interests_house_edit1'].value_counts()
        fig42.add_subplot(122)
        fg1 = sns.barplot(x=cond_dataset2[1:].index, y=cond_dataset2[1:].values)
        # fg1.set(xlim=(0,200))
        plt.title('产权是满2年的感兴趣人数')
        return None

    # 关联观察1
    def plot_rele_var1(self, dataset_used1):
        '''
        2年产权、5年产权vs装修程度、户型
        :param dataset_used1: pd.DataFrame
        :return: None
        '''
        fig5 = plt.figure(figsize=(10, 6))
        fig5.suptitle('2年产权、5年产权、装修程度、户型')
        fig5.add_subplot(231)
        dataset_used1.groupby('years_period_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('2年产权')

        fig5.add_subplot(232)
        dataset_used1.groupby('tax_free_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('5年产权')

        fig5.add_subplot(233)
        dataset_used1.groupby('decoration_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('装修程度')

        fig5.add_subplot(212)
        dataset_used1.groupby('unit_house')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('不同户型')
        return None

    # 关联观察2
    def plot_rele_var2(self, dataset_used1):
        '''
        电梯、楼层、楼型、建成时间单变量统计
        :param dataset_used1: pd.DataFrame
        :return: None
        '''
        fig5 = plt.figure(figsize=(10, 6))
        # plt.tight_layout(pad=1)
        fig5.suptitle('电梯、楼层、楼型、建成时间')

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        fig5.add_subplot(231)
        dataset_used1.groupby('elevator_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('电梯')

        fig5.add_subplot(232)
        dataset_used1.groupby('type_house_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('楼层')

        fig5.add_subplot(233)
        dataset_used1.groupby('years_house_type_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('楼型')

        fig5.add_subplot(212)
        dataset_used1.groupby('years_house_year_edit1')['smeter_price_edit1'].mean().plot(kind='bar')
        plt.title('建成时间')

    # 连续变量相关性处理
    def plot_check_rele(self, dataset_used1):
        '''
        连续变量相关性
        :param dataset_used1:pd.DataFrame
        :return: None
        '''
        plt.subplots(figsize=(8, 8))
        sns.heatmap(dataset_used1.corr(), vmax=1.0, square=True)
        return None
