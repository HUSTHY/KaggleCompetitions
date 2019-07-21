#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   stackingPrediction.py
@Time    :   2019/7/8 10:25
@Desc    :

'''

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
import seaborn as sn
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
from xgboost.sklearn import XGBClassifier as XGB
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import os
from sklearn.model_selection import KFold
from TitanicDiasterPrediction.codeFiles.SklearnHelper import SklearnHelper





def loadData():
    trainData = pd.read_csv('../data/train.csv')
    testData = pd.read_csv('../data/test.csv')
    return trainData, testData


def dataPreprocess(trainData, testData):
    train=trainData.copy()
    test=testData.copy()
    train=switchData(train)
    test=switchData(test)
    train=fillAgeNa(train)
    test=fillAgeNa(test)
    train=featureProcess(train)
    test=featureProcess(test)
    return train,test


def switchData(data):
    data = data[(data['Embarked'].notna())]
    data['Fare']=data['Fare'].fillna(data['Fare'].mean())
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data=data.drop(drop_cols,1)
    data['Embarked'] = data['Embarked'].map(lambda x: 0 if x == 's' else 1 if x == 'C' else 2)
    data['Sex']=data['Sex'].map(lambda x: 0 if x=='male' else 1)
    return  data


def fillAgeNa(data):
    #使用随机森林算法进行预测来填充
    columns=data.columns
    if 'Survived' in columns:
        tr=data[data['Age'].notna()]
        te=data[data['Age'].isna()]
        tr_x=tr[['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']]
        tr_y=tr[['Age']]
        tr_y=tr_y.values.ravel()#把tr_y转化为一维向量输入
        te_x = te[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        rfModel=rfr(n_estimators=100)#采用默认参数的随机森林模型
        rfModel.fit(tr_x,tr_y)
        Age_predict=rfModel.predict(te_x)
    else:
        tr = data[data['Age'].notna()]
        te = data[data['Age'].isna()]
        tr_x = tr[[ 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        tr_y = tr[['Age']]
        tr_y = tr_y.values.ravel()  # 把tr_y转化为一维向量输入
        te_x = te[[ 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        rfModel = rfr(n_estimators=100)  # 采用默认参数的随机森林模型
        rfModel.fit(tr_x, tr_y)
        Age_predict = rfModel.predict(te_x)
    data.loc[data['Age'].isna(),'Age']=Age_predict
    data['Age']=data['Age'].map(lambda x: int(x)) #年龄取整数
    return data

def featureProcess(data):
    #年龄分成7段
    data['Age']=data['Age'].map(lambda x: 0 if x<15 else 1 if x<30 else 2  if x<50 else 3 )
    data['familySize']=data['SibSp']+data['Parch']+1
    #家庭大小分为3段
    data['familySize']=data['familySize'].map(lambda x: 0 if x<2 else 1 if x <5 else 0)
    data['Fare']=data['Fare'].map(lambda x: np.log(x+1))
    data['Fare'] = data['Fare'].map(lambda x: 0 if x <2.7 else 1)
    columns=['SibSp','Parch']
    data=data.drop(columns,1)
    return data


def get_oof(clf,x_train,y_train,x_test):
    n_train=x_train.shape[0]
    n_test=x_test.shape[0]
    oof_test_skf=np.empty((5,n_test))
    kf=KFold(n_splits=5,shuffle=True,random_state=0)
    oof_train=np.zeros((n_train,))
    oof_test=np.zeros((n_test,))

    for i,(train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr=x_train.iloc[train_index]
        y_tr=y_train[train_index]
        x_te=x_train.iloc[test_index]
        clf.fit(x_tr,y_tr)
        oof_train[test_index]=clf.predict(x_te)
        oof_test_skf[i,:]=clf.predict(x_test)
    oof_test=oof_test_skf.mean(axis=0)
    return oof_train.reshape((n_train,1)),oof_test.reshape((n_test,1))

def stackingPrediction(trainData, testData):
    x_train = trainData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    y_train = trainData['Survived'].values.ravel()
    x_test  = testData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]

    modelNames=os.listdir('../models/')
    oof_trainList=[]
    oof_testList=[]
    xgbclf=None
    for modelName in modelNames:
        strs=modelName.split('_')[0]
        modelpath = '../models/' + modelName
        if strs=='svm' or strs=='KNN' or strs=='ADA' or strs=='rfc':
            clf=joblib.load(modelpath)
            oof_train, oof_test=get_oof(clf,x_train,y_train,x_test)#第一层stacking层模型，得到新的训练集和测试集的特征变量
            oof_trainList.append(oof_train)
            oof_testList.append(oof_test)
        if strs=='XGB':
            xgbclf=joblib.load(modelpath)
    final_x_train=oof_trainList[0]
    print(final_x_train.shape)
    final_x_test=oof_testList[0]
    print(final_x_test.shape)
    for i in range(1,len(oof_trainList)):
        final_x_train=np.concatenate((final_x_train,oof_trainList[i]),axis=1)
        final_x_test=np.concatenate((final_x_test,oof_testList[i]),axis=1)

    tr_tr_x, tr_te_x, tr_tr_y, tr_te_y = train_test_split(final_x_train, y_train, test_size=0.32, random_state=1)
    # clf=XGB(reg_alpha=0.2, reg_lambda=0.5,n_estimators=67,gamma=0.2,max_depth=3)
    clfs=[]
    n_estimators=np.arange(10,20,1)
    best=0
    biggestScore=0
    tr_scores=[]
    te_scores=[]
    print(XGB())
    for n_estimator in n_estimators:
        clf=XGB(n_estimators=n_estimator,random_state=1)
        clfs.append(clf)
    for clf,n_estimator in zip(clfs,n_estimators):
        clf.fit(tr_tr_x, tr_tr_y)
        score1=clf.score(tr_tr_x, tr_tr_y)
        score2 = clf.score(tr_te_x, tr_te_y)
        tr_scores.append(score1)
        te_scores.append(score2)
        # print(score1,score2,n_estimator)
        if biggestScore<score2:
            biggestScore=score2
            best=n_estimator

    showlearninreate(tr_scores,te_scores,n_estimators)



    print('best:%s   biggestScore:%s'%(best,biggestScore))

    XGB(n_estimators=45, random_state=1)


    xgbclf.fit(final_x_train,y_train)
    print(xgbclf.score(final_x_train,y_train))
    result=xgbclf.predict(final_x_test)
    gender_data = pd.read_csv('../data/gender_submission.csv')
    subminssion = pd.DataFrame({'PassengerId': gender_data['PassengerId'], 'Survived': result})
    subminssion.to_csv('../result/stackingPrediction.csv', index=False)


def showlearninreate(y1,y2,x):
    plt.figure(figsize=(13,7))
    plt.plot(x,y1,C='b',label='train')
    plt.plot(x,y2,C='r',label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainData, testData = loadData()
    trainData, testData=dataPreprocess(trainData, testData)
    stackingPrediction(trainData, testData)
