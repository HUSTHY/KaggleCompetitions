#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   rossmanPrediction.py
@Time    :   2019/7/14 10:35
@Desc    :

'''
from RossmanPredictionCompetition.code.functions import *



if __name__ == '__main__':
    train,test=dataLoadandExplore()
    # dataAnalysis(train)
    hold_out_train,hold_out_test,test=dataAndFeatureProcess( train,test)
    # model=modeling(hold_out_train,hold_out_test)
    # savemodel(model,'../model/xgbrModel.m')
    # prediction(hold_out_test,test)
    # model=modelingLGBM(hold_out_train,hold_out_test)
    # savemodel(model,'../model/lgbmModel.m')
    # predictionlgbm(hold_out_test, test)
    modelIntergrated(hold_out_train, hold_out_test, test)
