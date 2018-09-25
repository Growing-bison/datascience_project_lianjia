from setuptools import setup, find_packages
setup(name = 'data_science_lianjia',version = '1.0',
py_modules = ['data_science_lianjia.analysis','data_science_lianjia.data_processing','data_science_lianjia.feature_engineer','data_science_lianjia.main','data_science_lianjia.modelling'],
author = 'Qikun Lu',
author_email = 'jack.lu@gmail.com',
url = '',
description = 'A simple exercise for machine learning about house price prediction',
packages=find_packages(exclude=['*.*']))