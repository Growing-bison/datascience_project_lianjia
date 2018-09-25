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

'''Data processing function.
Environment
    python 3.6
    matplotlib2.0.2
    numpy 1.12.1
    seaborn 0.7.1
    scipy 0.19.1
    scikit-learn 0.19.0
    lightgbm 2.1.2
    py-xgboost 0.60

This is for finishing data processing.
'''
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output
pd.set_option('display.max_columns',40) # 显示隐藏的列，显示40列

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决plt绘图无法显示中文问题
plt.rcParams['font.family'] = 'sans-serif'  # 解决负号是方块
# get_ipython().run_line_magic('matplotlib', 'notebook')  在ipython上执行显示时候需要用到

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


def size_func(x, y):
    '''
    获取面积值
    :param x: string
    :param y: string
    :return: float
    '''
    def size_help_func(x):
        '''
        获取面积值-辅助1
        :param x: string
        :return: float, area
        '''
        x = str(x)
        x = x.replace('平方米', '')
        x = x.replace('平米', '')
        x = x.replace('米', '')

        if (('室' in x) | ('厅' in x) | (x == 'nan') | ('车位' in x) | ('房' in x) | ('墅' in x)):
            num = 0
        else:
            num = float(x)

        return num

    def info_func(x):
        '''
        获取面积值-辅助2规范信息
        :param x: string
        :return: float
        '''
        if '平米' in str(x):
            a = x.split('平米')[0].split('|')[-1].strip()
            if len(a) > 1:
                num = a
            else:
                num = 0.0
        else:
            num = 0.0
        return num

    a = size_help_func(x)
    b = info_func(y)

    if a == 0.0:
        if ('车位' not in str(b)):
            num = float(b)
        else:
            num = a
    else:
        num = a

    return num

def size_addcata_func(a):
    '''
    异常值辅助处理-面积信息类型
    :param a: float
    :return: string
    '''
    #     a = watch_time_func(x)
    if a <= 10:
        label = str(1)
    else:
        label = str(0)
    return label

def watch_time_func(x):
    '''
    watch_times列信息提取
    :param x: string
    :return: int
    '''
    if str(x) == 'nan':
        num = -1
    else:
        a = x.split('次')[0].strip()
        num = int(a)
    return num

def watch_time_addcata_func(x):
    '''
    缺失值处理-watch_times列信息类别
    :param x: int
    :return: string
    '''
    if x == -1:
        label = str(1)
    else:
        label = str(0)
    return label

def interests_house_func(x):
    '''
    interests_house列信息提取
    :param x: string
    :return: int
    '''
    if str(x) == 'nan':
        num = -1
    else:
        a = x.split('人')[0].strip()
        num = int(a)
    return num

def interests_house_addcata_func(x):
    '''
    缺失值处理-interests_house列信息类别
    :param x: int
    :return: string
    '''
    if x == -1:
        label = str(1)
    else:
        label = str(0)
    return label

def submit_period_func(x):
    '''
    submit_period列信息提取
    :param x: string
    :return: int
    '''
    if str(x) == 'nan':
        num = -1
    elif '刚刚' in str(x):
        num = 0
    elif '年' in str(x):
        a = x.split('年')[0].strip()
        if a == '一':
            num = 365
        elif a == '二':
            num = 730
        else:
            num = 1000
    elif '个月' in x:
        a = x.split('个月')[0].strip()
        num = int(a) * 30
    elif '天' in x:
        a = x.split('天')[0].strip()
        num = int(a)
    else:
        num = -2
    return num

def submit_period_addcata_func(x):
    '''
    缺失值处理-submit_period列信息类别
    :param x: int
    :return: string
    '''
    a = submit_period_func(x)
    if a == -2:
        label = 3
    elif a == -1:
        label = 2
    elif a == 1000:
        label = 1
    else:
        label = 0
    return str(label)

def years_period_func(x):
    '''
    years_period列信息提取
    :param x: string
    :return: string
    '''
    if str(x) == 'nan':
        label = str(0)
    else:
        label = str(1)
    return label

def smeter_price_func(x):
    '''
    smeter_price列信息提取
    :param x: string
    :return: int
    '''
    a = x.split('元')[0].replace('单价', '')
    if len(a) <= 3:
        num = -1
    else:
        num = int(a)
    return num

def direction_func(x, y, z):
    '''
    direction信息提取
    :param x: string, direction_house列
    :param y: string, decoration_house列
    :param z: string, info_cluster列
    :return: string
    '''
    x = str(x)
    y = str(y)
    z = str(z)
    dir_list = ['东', '西', '南', '北']
    if ((dir_list[0] in x) | (dir_list[1] in x) | (dir_list[2] in x) | (dir_list[3] in x)):
        label = x
    elif (dir_list[0] in y) | (dir_list[1] in y) | (dir_list[2] in y) | (dir_list[3] in y):
        label = y
    elif (dir_list[0] in z) | (dir_list[1] in z) | (dir_list[2] in z) | (dir_list[3] in z):
        a = z.split('|')
        for value in a:
            if (dir_list[0] in value) | (dir_list[1] in value) | (dir_list[2] in value) | (dir_list[3] in value):
                label = value
            else:
                label = 'nodata'
    else:
        label = 'nodata'

    return label

def decoration_func(x, y, z):
    '''
    decoration信息提取
    :param x: string, direction_house列
    :param y: string, decoration_house列
    :param z: string, info_cluster列
    :return: string
    '''
    x = str(x)
    y = str(y)
    z = str(z)
    dir_list = ['精装', '其他', '毛坯', '简装']
    if ((dir_list[0] in x) | (dir_list[1] in x) | (dir_list[2] in x) | (dir_list[3] in x)):
        label = x.strip()
    elif (dir_list[0] in y) | (dir_list[1] in y) | (dir_list[2] in y) | (dir_list[3] in y):
        label = y.strip()
    elif (dir_list[0] in z) | (dir_list[1] in z) | (dir_list[2] in z) | (dir_list[3] in z):
        a = z.split('|')
        for value in a:
            if (dir_list[0] in value) | (dir_list[1] in value) | (dir_list[2] in value) | (dir_list[3] in value):
                label = value.strip()
            else:
                label = 'nodata'
    else:
        label = 'nodata'

    return label

def elevator_func(x, y, z):
    '''
    elevator信息提取
    :param x: string, decoration_house列
    :param y: string, elevator_house列
    :param z: string, info_cluster列
    :return: string
    '''
    x = str(x)
    y = str(y)
    z = str(z)
    dir_list = ['有电梯', '无电梯']
    if (dir_list[0] in x) | (dir_list[1] in x):
        label = x.strip()
    elif (dir_list[0] in y) | (dir_list[1] in y):
        label = y.strip()
    elif (dir_list[0] in z) | (dir_list[1] in z):
        a = z.split('|')
        for value in a:
            if (dir_list[0] in value) | (dir_list[1] in value):
                label = value.strip()
            else:
                label = 'nodata'
    else:
        label = 'nodata'
    return label

def floor_type_func(x):
    '''
    type_house列信息提取
    :param x: string
    :return: string
    '''
    x = str(x)
    if '共' in x:
        a = x.split('(')[0]
        label = a
    elif '层' in x:
        a = x.split('层')[0]
        a = int(a)
        if a <= 1:
            label = '底层'
        elif (a > 1) | (a < 6):
            label = '低楼层'
        elif (a >= 6) | (a < 15):
            label = '中楼层'
        else:
            label = '高楼层'
    elif '平房' in x:
        label = '底层'
    elif x == 'nan':
        label = 'nodata'
    else:
        label = 'nodata'
    return label

def years_house_type_func(x, y):
    '''
    house type信息提取
    :param x: string, type_house
    :param y: string, years_house
    :return: string
    '''
    x = str(x)
    y = str(y)
    type_list = ['板塔', '板', '塔', '平房', '叠']
    if (type_list[0] in x) | (type_list[0] in y):
        label = '板塔'
    elif (type_list[1] in x) | (type_list[1] in y):
        label = '板'
    elif (type_list[2] in x) | (type_list[2] in y):
        label = '塔'
    elif (type_list[3] in x) | (type_list[3] in y):
        label = '平房'
    elif (type_list[4] in x) | (type_list[4] in y):
        label = '别墅'
    else:
        label = 'nodata'
    return label

def years_house_year_func(x, y):
    '''
    house year信息提取
    :param x: string, type_house
    :param y: string, years_house
    :return: int
    '''
    x = str(x)
    y = str(y)
    if ('年' in x):
        a = x.split('年')[0].replace('\'', '').strip()
        num = int(a)
    elif ('年' in y):
        a = y.split('年')[0].replace('\'', '').strip()
        num = int(a)
    else:
        num = None

    return num

def main_apply_func(dataset):
    '''
    所有列的处理，提取信息并新建立列
    :param dataset: pd.DataFrame
    :return dataset: pd.DataFrame
    '''

    dataset['size_house_edit1'] = list(
        map(lambda x, y: size_func(x, y), dataset['unit_house'], dataset['info_cluster']))
    dataset['size_house_edit1_addcata'] = dataset['size_house_edit1'].apply(size_addcata_func)

    dataset['watch_time_edit1'] = dataset['watch_times'].apply(watch_time_func)
    dataset['watch_time_edit1_addcata'] = dataset['watch_time_edit1'].apply(watch_time_addcata_func)

    dataset['interests_house_edit1'] = dataset['interests_house'].apply(interests_house_func)
    dataset['interests_house_edit1_addcata'] = dataset['interests_house_edit1'].apply(interests_house_addcata_func)

    dataset['submit_period_edit1'] = dataset['submit_period'].apply(submit_period_func)
    dataset['submit_period_edit1_addcata'] = dataset['submit_period'].apply(submit_period_addcata_func)

    dataset['years_period_edit1'] = dataset['years_period'].apply(years_period_func)
    dataset['tax_free_edit1'] = dataset['tax_free'].apply(years_period_func)
    dataset['smeter_price_edit1'] = dataset['smeter_price'].apply(smeter_price_func)

    dataset['direction_edit1'] = list(
        map(lambda x, y, z: direction_func(x, y, z), dataset['direction_house'], dataset['decoration_house'],
            dataset['info_cluster']))
    dataset['decoration_edit1'] = list(
        map(lambda x, y, z: decoration_func(x, y, z), dataset['direction_house'], dataset['decoration_house'],
            dataset['info_cluster']))
    dataset['elevator_edit1'] = list(
        map(lambda x, y, z: elevator_func(x, y, z), dataset['decoration_house'], dataset['elevator_house'],
            dataset['info_cluster']))

    dataset['type_house_edit1'] = dataset['type_house'].apply(floor_type_func)
    dataset['years_house_type_edit1'] = list(
        map(lambda x, y: years_house_type_func(x, y), dataset['type_house'], dataset['years_house']))
    dataset['years_house_year_edit1'] = list(
        map(lambda x, y: years_house_year_func(x, y), dataset['type_house'], dataset['years_house']))

    return dataset

def main_check_func(dataset):
    '''
    检查和统计功能
    :param dataset: pd.DataFrame
    :return: N
    '''
    print('房屋小区类型：', len(dataset['community_house'].unique()));

    print('房屋户型：', len(dataset['unit_house'].unique()))

    print('房屋面积：', 'max:', max(dataset['size_house_edit1'].unique()),
          'min:', min(dataset['size_house_edit1'].unique()),
          '空值：', len(dataset[dataset['size_house_edit1'] == 0.]))

    print('房屋朝向：', len(dataset['direction_house'].unique()))

    print('看房次数：', 'max:', max(dataset['watch_time_edit1'].unique()),
          'min:', min(dataset['watch_time_edit1'].unique()),
          '空值：', len(dataset[dataset['watch_time_edit1'] == -1]))

    print('收藏次数：', 'max:', max(dataset['interests_house_edit1'].unique()),
          'min:', min(dataset['interests_house_edit1'].unique()),
          '空值：', len(dataset[dataset['interests_house_edit1'] == -1]))

    print('多久前发布：', 'max:', max(dataset['submit_period_edit1'].unique()),
          'min:', min(dataset['submit_period_edit1'].unique()),
          '空值：', len(dataset[dataset['submit_period_edit1'] == -1]))

    print('多久前发布的类型：', len(dataset['submit_period_edit1_addcata'].unique()))
    print('2年产权类型：', len(dataset['years_period_edit1'].unique()))
    print('5年产权类型：', len(dataset['tax_free_edit1'].unique()))

    print('总价：', 'max:', max(dataset['total_price'].unique()),
          'min:', min(dataset['total_price'].unique()),
          '空值：', len(dataset[dataset['total_price'] == -1]))

    print('单位价钱：', 'max:', max(dataset['smeter_price_edit1'].unique()),
          'min:', min(dataset['smeter_price_edit1'].unique()),
          '空值：', len(dataset[dataset['smeter_price_edit1'] == -1]))
