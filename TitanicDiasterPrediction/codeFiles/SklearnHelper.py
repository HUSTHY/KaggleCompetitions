#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   SklearnHelper.py
@Time    :   2019/7/8 10:46
@Desc    :

'''
class SklearnHelper(object):
    def __init__(self,clf,seed=0,params=None):
        params['random_state']=seed
        self.clf=clf(**params)

    def train(self,train_x,train_y):
        self.clf.fit(train_x,train_y)

    def predict(self,train_x,train_y):
        return self.clf.predict(train_x,train_y)

    def fit(self,x,y):
        return self.clf.fit(x,y)
