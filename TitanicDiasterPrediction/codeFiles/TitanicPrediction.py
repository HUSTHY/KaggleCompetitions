'''
@Author  :    HY
@Software:   PyCharm
@File    :   TitanicPrediction.py
@Time    :   2019/7/3 15:51
@Desc    :

'''
#!/usr/bin/env python

# -*- encoding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
import seaborn as sn
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier as XGB
from sklearn.model_selection import cross_val_score as cvs
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import os
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def loadData():
    trainData = pd.read_csv('../data/train.csv')
    testData = pd.read_csv('../data/test.csv')
    return trainData, testData
def dataDescribe( trainData, testData):
    # print(trainData.describe())
    # print('*************************************')
    # print(testData.describe())
    # print('*************************************')
    print(trainData.isna().sum())
    print('*************************************')
    print(testData.isna().sum())
    print('*************************************')
    # print(trainData)
    # print('*************************************')
    # print(testData)
    # print('*************************************')


def dataAnalysis(trainData):

    showSurvived(trainData)
    showPclass(trainData)
    showSex(trainData)
    showAge(trainData)
    showAgePclassAndSurvived(trainData)
    showSibSpAndParch(trainData)
    showEmbarked(trainData)
    showFare(trainData)
    showcoff(trainData)

def showSurvived(trainData):
    #存活率
    f,ax=plt.subplots(1,2)
    trainData['Survived'].value_counts().plot.pie(explode=(0,0.1),ax=ax[0], autopct='%1.2f %%',shadow=True)#matplotlib中的饼状图explode离开轴心的距离
    ax[0].set_title('Survived')
    ax[0].set_ylabel('')
    sn.countplot('Survived',data=trainData,ax=ax[1])#seaborn中的柱形图——直接计数
    plt.show()


def showPclass(trainData):
    #Pclass与存活的关系
    f,ax=plt.subplots(1,3,figsize=(13,7))
    trainData['Pclass'].value_counts().plot.pie(ax=ax[0],autopct='%1.2f %%',shadow=True)
    sn.countplot('Pclass',data=trainData,ax=ax[1])
    sn.countplot('Pclass',hue='Survived',data=trainData,ax=ax[2])#Survived的状态来区分Pclass
    plt.show()
def showSex(trainData):
    #性别和存活的关系
    f, ax = plt.subplots(1, 3,figsize=(13,7))
    trainData['Sex'].value_counts().plot.pie(ax=ax[0],autopct='%1.2f %%',shadow=True)
    sn.countplot('Sex', data=trainData, ax=ax[1])
    sn.countplot('Sex', hue='Survived', data=trainData, ax=ax[2])
    plt.show()

def showAge(trainData):
    f, ax = plt.subplots(1, 2)
    data=trainData.copy()
    sn.violinplot('Survived', 'Age', data=data, ax=ax[0])
    my_pal = {0: "brown", 1: "blue"}
    sn.violinplot('Survived','Age',data=data,palette=my_pal,alpha=0.8,ax=ax[1])
    data['Survived']=data['Survived'].map(lambda x: 1 if x==0 else 0)
    my_pal = {1: "brown", 0: "blue"}
    sn.violinplot('Survived', 'Age', data=data, palette=my_pal,alpha=0.8,ax=ax[1])
    plt.show()


def showAgePclassAndSurvived(trainData):
    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    sn.violinplot('Pclass','Age',hue='Survived',data=trainData,split=True,ax=ax[0])
    sn.violinplot('Sex', 'Age', hue='Survived', data=trainData, split=True, ax=ax[1])
    ax[0].set_title('Pclass and Age  vs  Survived')
    ax[1].set_title('Sex and Age  vs  Survived')
    plt.show()

def showSibSpAndParch(trainData):
    #对familySize进行分段
    f, ax = plt.subplots(1, 3, figsize=(14, 7))
    trainData['familySize']=trainData['SibSp']+trainData['Parch']+1
    trainData['familySize'].value_counts().plot.pie(ax=ax[0],autopct='%1.2f%%',shadow=True)
    sn.countplot('familySize',data=trainData, ax=ax[1])
    sn.countplot('familySize',hue='Survived',data=trainData,ax=ax[2])
    plt.show()

def showFare(trainData):
    data=trainData.copy()
    f, ax = plt.subplots(1, 2)
    data['Fare']=data['Fare'].map(lambda x:np.log(x+1))
    sn.violinplot('Survived', 'Fare', data=data, split=True, ax=ax[0])
    my_pal = {1: "brown", 0: "blue"}
    sn.violinplot('Survived','Fare',data=data,split=True,palette=my_pal,ax=ax[1])
    data['Survived']=data['Survived'].map(lambda x: 1 if x==0 else 0)
    my_pal = {0: "brown",1: "blue"}
    sn.violinplot('Survived', 'Fare', data=data, split=True,palette=my_pal,ax=ax[1])
    plt.show()

def showEmbarked(trainData):
    f, ax = plt.subplots(2, 3, figsize=(14, 7))
    trainData['Embarked'].value_counts().plot.pie(ax=ax[0][0], autopct='%1.2f%%', shadow=True)
    sn.countplot('Embarked',data=trainData,ax=ax[0][1])
    sn.violinplot('Embarked','Age',hue='Survived',data=trainData,ax=ax[0][2])
    sn.countplot('Embarked',data=trainData,hue='Survived',ax=ax[1][0])
    sn.countplot('Embarked',data=trainData, hue='Pclass', ax=ax[1][1])
    sn.countplot('Embarked', data=trainData, hue='Sex', ax=ax[1][2])
    plt.show()

def showcoff(data):
    trainData=data.copy()
    trainData=trainData.drop(columns=['PassengerId'],axis=1)
    colormap=plt.cm.RdBu
    plt.figure(figsize=(13,7))
    plt.title('Person correlation of Features')
    sn.heatmap(trainData.corr(),cmap=colormap,vmax=1.0,square=True,annot=True)
    plt.show()
    print(trainData.corr())
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
    data['Age']=data['Age'].map(lambda x: 0 if x<15 else 1 if x<30 else 2 if x<50 else 3 )
    data['familySize']=data['SibSp']+data['Parch']+1
    #家庭大小分为3段
    data['familySize']=data['familySize'].map(lambda x: 0 if x<2 else 1 if x <5 else 0)
    data['Fare']=data['Fare'].map(lambda x: np.log(x+1))
    data['Fare'] = data['Fare'].map(lambda x: 0 if x <2.7 else 1)
    columns=['SibSp','Parch']
    data=data.drop(columns,1)
    return data


def delNoiseData(data):
    columns=['Age','SibSp','Parch','Fare']
    for col in columns:
        Q1=np.percentile(data[col],25)
        print(Q1)
        Q3=np.percentile(data[col],75)
        print(Q3)
        IQR=Q3-Q1
        outlier=1.5*IQR
        print(outlier)
        outlier_col=data[(data[col]<Q1-outlier)| (data[col]>Q3+outlier)].index
        print(col)
        print(outlier_col)
    return data


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()



def simpleAlgorithm(trainData):
    trainx=trainData[[ 'Pclass',  'Sex',  'Age',  'Fare',  'Embarked',  'familySize']]
    trainy=trainData['Survived'].values.ravel()
    tr_tr_x,tr_te_x,tr_tr_y,tr_te_y=train_test_split(trainx,trainy,test_size=0.3,random_state=0)

    clf_svms=[]
    clf_KNNs = []
    clf_LRs = []
    clf_MNBs = []
    clf_MLPs = []
    clf_rfcs = []
    clf_XGBs = []
    clf_ADAs = []

    # C=np.arange(0.1,2,0.1)
    # gamma=np.arange(0.01,0.2,0.01)
    # for p1 in C:
    #     for p2 in gamma:
    #         clf_svm = SVC(C=p1, gamma=p2, kernel='rbf', random_state=1)
    #         clf_svms.append(clf_svm)
    # tols=[1.0,0.01,0.001,0.0001,0.00001,0.000001]
    # for ele in tols:
    #     clf_svm=SVC(C=4.7,gamma=0.07,kernel='rbf',random_state=0,tol=ele)
    #     clf_svms.append(clf_svm)
    # for p2 in coef0s:
    #     clf_svm = SVC(C=0.9, gamma=0.0999999999, kernel='rbf', random_state=0,probability=True,coef0=p2)
    #     clf_svms.append(clf_svm)
    # parameters={
    #     'C':[0.01,0.1,1,2,5],
    #     'kernel':['rbf','linear','poly']
    # }
    # clf=SVC()
    # gsearch=GridSearchCV(clf,param_grid=parameters,scoring='roc_auc',cv=5)
    # gsearch.fit(trainx,trainy)
    # print(gsearch.best_params_,gsearch.best_score_)


    clf_svm = SVC(C=0.5, gamma=0.19, kernel='rbf', random_state=0,probability=True)
    clf_svms.append(clf_svm)


    # clf = SVC(C=2,kernel='poly')
    # cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=1)
    # plot_learning_curve(clf,tr_tr_x,tr_tr_y,ylim=(0.7,0.95),cv=cv)

    # n_neighbors=range(1,20)
    # for n_neighbor in n_neighbors:
    #     clf_KNN = KNN(n_neighbors=n_neighbor)
    #     clf_KNNs.append(clf_KNN)
    # sizes=range(20,100,10)
    # for size in sizes:
    #     clf_KNN = KNN(n_neighbors=4,leaf_size=size)
    #     clf_KNNs.append(clf_KNN)
    clf_KNN = KNN(n_neighbors=4,leaf_size=80)
    clf_KNNs.append(clf_KNN)

    C=np.arange(0.018,0.021,0.001)
    for c in C:
        clf_LR = LR(tol=0.0001, random_state=0, solver='lbfgs', penalty='l2', C=c)
        clf_LRs.append(clf_LR)

    clf_MNB=MNB()
    clf_MNBs.append(clf_MNB)
    hidden_layer_sizes=[]
    # for i in range(6,15,6):
    #     tuple=(i,)
    #     hidden_layer_sizes.append(tuple)
    # for  i in range(3,8,1):
    #     for j in range(3, 8, 1):
    #         for k in range(3,8,1):
    #             tuple = (i,j,k)
    #             hidden_layer_sizes.append(tuple)
    # for  i in range(7,15,1):
    #     tuple = (i,i,i)
    #     hidden_layer_sizes.append(tuple)
    # hidden_layer_sizes=[(6,6),(7, 3),(6,6,6),(7,3,7),(7,7,7),(70,70,70)]
    hidden_layer_sizes = [(12,12,12)]
    for hidden_layer_size in hidden_layer_sizes:
        clf_MLP = MLP(hidden_layer_sizes=hidden_layer_size, max_iter=10000,tol=0.00001,solver='lbfgs')
        clf_MLPs.append(clf_MLP)


    # n_estimators=range(10,50)
    # max_depths=range(3,5)
    # max_features=range(1,7)
    # min_samples_splits=range(2,23,1)
    # min_samples_leafs=range(1,101,10)
    n_estimators = np.arange(17,18,1)
    for n_estimator in n_estimators:
        clf_rfc = rfc(n_estimators=n_estimator, max_depth=4, random_state=0,max_features=2,oob_score=False,min_samples_split=18)
        clf_rfcs.append(clf_rfc)
    #     for depth in max_depths:
    #         clf_rfc=rfc(n_estimators=n_estimator,max_depth=depth,random_state=0)
    # for min_samples_leaf in min_samples_leafs:
    #     clf_rfc = rfc(n_estimators=30, max_depth=4, random_state=0,max_features=1,oob_score=True,min_samples_split=18)
    #     clf_rfcs.append(clf_rfc)
    # min_samples_leafs=np.arange(0.1,0.6,0.1)
    # min_samples_leafs.append(1)
    # for leaf in min_samples_leafs:
    clf_rfc = rfc(n_estimators=26, max_depth=4, random_state=0,max_features=2,oob_score=False,min_samples_split=18)
    clf_rfcs.append(clf_rfc)


    # n_estimators=range(62,63,1)
    # max_depths=range(3,4)
    # for n_estimator in n_estimators:
    #     for depth in  max_depths:
    #         clf_XGB=XGB(n_estimators=n_estimator,max_depth=depth)
    #         clf_XGBs.append(clf_XGB)
    # min_child_weights=range(1,101,20)
    # subsamples=np.arange(0.5,1.0,0.2)
    # gammas=np.arange(0,1.0,0.2)
    # for min_child_weight in min_child_weights:
    #     for subsample in subsamples:
    #         for gamma in gammas:
    #             clf_XGB = XGB(n_estimators=62, max_depth=3,min_child_weight=min_child_weight,subsample=subsample,gamma=gamma)
    #             clf_XGBs.append(clf_XGB)

    # n_estimators=np.arange(60,80,1)
    # for n_estimator in n_estimators:
    #     clf_XGB = XGB(random_state=0,n_estimators=n_estimator, max_depth=3, min_child_weight=1, subsample=0.9, gamma=0.2,reg_alpha=0.2,learning_rate=0.30000000000000004,reg_lambda=0.5)
    #     clf_XGBs.append(clf_XGB)

    clf_XGB = XGB(random_state=0, n_estimators=67, max_depth=3, min_child_weight=1, subsample=0.9, gamma=0.2,
                  reg_alpha=0.2, learning_rate=0.30000000000000004, reg_lambda=0.5)
    clf_XGBs.append(clf_XGB)

    # n_estimators=range(1,20,1)
    # learning_rates=np.arange(0.8,1.0,0.01)
    # for learning_rate in learning_rates:
    #     for n_estimator in n_estimators:
    #         clf_ada = ADA(n_estimators=n_estimator,learning_rate=learning_rate)
    #         clf_ADAs.append(clf_ada)
    # n_estimators=np.arange(1,13,1)
    # for n_estimator in n_estimators:
    #     clf_ada = ADA(n_estimators=n_estimator, learning_rate=0.9800000000000002,random_state=0)
    #     clf_ADAs.append(clf_ada)
    clf_ada = ADA(n_estimators=9, learning_rate=0.9800000000000002, random_state=0)
    clf_ADAs.append(clf_ada)

    clfs=[clf_svms,clf_KNNs,clf_LRs,clf_MNBs,clf_MLPs,clf_rfcs,clf_XGBs,clf_ADAs]
    Algorithms = ['svm', 'KNN', 'LR', 'MNB', 'MLP', 'rfc', 'XGB','ADA']
    savemodels = []
    savemodelpaths=[]

    for clf,Algorithm in zip(clfs,Algorithms):
        bigges_tr_score = 0
        best_tr_clf = ''
        bigges_te_score = 0
        best_te_clf = ''
        corr_te_score=0
        corr_tr_score=0
        for ele in clf:
            if Algorithm!='MNB':
                tr_x_scale=scale(tr_tr_x)
                tr_te_x_scale=scale(tr_te_x)
                tr_score,te_score,tr_mean_score = trainModel(ele, Algorithm, tr_x_scale,tr_tr_y,tr_te_x_scale,tr_te_y)
            else:
                tr_score,te_score,tr_mean_score = trainModel(ele, Algorithm, tr_tr_x,tr_tr_y,tr_te_x,tr_te_y)
            if tr_score>bigges_tr_score:
                bigges_tr_score=tr_score
                corr_te_score=te_score
                best_tr_clf=ele
            if te_score > bigges_te_score:
                bigges_te_score = te_score
                corr_tr_score=tr_score
                best_te_clf = ele
        print('训练集切分训练数据最佳得分：%s score is:%.4f    对应的测试数据：%s score is:%.4f'%(Algorithm,bigges_tr_score,Algorithm,corr_te_score))
        print('训练集切分测试数据最佳得分：%s score is:%.4f    对应的训练数据：%s score is:%.4f' % (Algorithm, bigges_te_score,Algorithm,corr_tr_score))
        print('best Algorithm is:',best_tr_clf)
        print('best Algorithm is:', best_te_clf)
        # cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=1)
        # plot_learning_curve(best_te_clf,Algorithm,trainx,trainy,ylim=(0.7,0.95),cv=cv)
        savemodels.append(best_te_clf)

    for Algorithm in Algorithms:
        str='../models/'+Algorithm+'_model.m'
        savemodelpaths.append(str)

    return savemodels,savemodelpaths


def trainModel(clf,Algorithm, tr_x, tr_y,tr_te_x,tr_te_y):
    clf.fit(tr_x, tr_y)
    tr_score = clf.score(tr_x, tr_y)
    print('训练集中切分的训练数据_%s score is %.4f' % (Algorithm, tr_score),end=' ')

    te_score = clf.score(tr_te_x, tr_te_y)
    print('训练集中切分的测试数据_%s  score is %.4f' % (Algorithm, te_score),end=' ')

    scores=cvs(clf,tr_x, tr_y,cv=5)
    np_tr_scores=np.array(scores)
    tr_mean_score=np_tr_scores.mean()
    print('训练集中切分的训练数据交叉验证：_%s mena score is %.4f'%(Algorithm,tr_mean_score))

    # te_scores = cvs(clf, tr_te_x,tr_te_y, cv=5)
    # np_te_scores = np.array(te_scores)
    # te_mean_score = np_te_scores.mean()
    # print('训练集中切分的测试数据交叉验证：_%s mena score is %.4f' % (Algorithm, te_mean_score), end='  ')
    return tr_score,te_score,tr_mean_score

def savemodel(savemodels,savemodelpaths):
    for model,path in zip(savemodels,savemodelpaths):
        joblib.dump(model,path)


def prediction(trainData, testData):
    gender_data=pd.read_csv('../data/gender_submission.csv')
    tr_x = trainData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    tr_y = trainData['Survived'].values.ravel()
    te_x=  testData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    modelsNames=os.listdir('../models/')
    for modelName in modelsNames:
        modelpath='../models/'+modelName
        clf=joblib.load(modelpath)
        strs=modelName.split('_')
        algor=strs[0]
        if algor!='MNB':
            tr_x_scale=scale(tr_x)
            te_x_scale=scale(te_x)
            getResult(clf, tr_x_scale, tr_y, te_x_scale, gender_data, algor)
        else:
            getResult(clf,tr_x, tr_y,te_x,gender_data,algor)

def predictionWithVoting(trainData, testData):
    gender_data = pd.read_csv('../data/gender_submission.csv')
    tr_x = trainData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    tr_y = trainData['Survived'].values.ravel()
    te_x = testData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    modelsNames = os.listdir('../models/')
    estimators=[]
    for modelName in modelsNames:
        modelpath = '../models/' + modelName
        strs = modelName.split('_')
        algor = strs[0]
        clf = joblib.load(modelpath)
        tuple=(algor,clf)
        estimators.append(tuple)
    voting_clf=VotingClassifier(estimators=estimators,voting='soft')
    voting_clf.fit(tr_x,tr_y)
    predict=voting_clf.predict(te_x)
    result = pd.DataFrame({'PassengerId': gender_data['PassengerId'], 'Survived': predict})
    result.to_csv('../result/votingWithAllBasePrediction.csv',index=False)

def predictionWithVotingASXRK(trainData, testData):
    gender_data = pd.read_csv('../data/gender_submission.csv')
    tr_x = trainData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    tr_y = trainData['Survived'].values.ravel()
    te_x = testData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'familySize']]
    modelsNames = os.listdir('../models/')
    estimators=[]
    for modelName in modelsNames:
        modelpath = '../models/' + modelName
        strs = modelName.split('_')
        algor = strs[0]
        clf = joblib.load(modelpath)
        if algor=='ADA' or algor=='svm' or algor=='XGB' or algor=='rfc' or algor=='KNN':
            tuple=(algor,clf)
            estimators.append(tuple)
    soft_voting_clf=VotingClassifier(estimators=estimators,voting='soft')
    soft_voting_clf.fit(tr_x,tr_y)
    soft_voting_score=soft_voting_clf.score(tr_x,tr_y)
    soft_Predict=soft_voting_clf.predict(te_x)
    print('soft_Predict：',soft_voting_score)

    hard_voting_clf = VotingClassifier(estimators=estimators, voting='hard')
    hard_voting_clf.fit(tr_x, tr_y)
    hard_voting_score = hard_voting_clf.score(tr_x, tr_y)
    hard_Predict = hard_voting_clf.predict(te_x)
    print('hard_Predict:',hard_voting_score)



    soft_result = pd.DataFrame({'PassengerId': gender_data['PassengerId'], 'Survived': soft_Predict})
    soft_result.to_csv('../result/VotingASXRK_soft.csv',index=False)

    soft_result = pd.DataFrame({'PassengerId': gender_data['PassengerId'], 'Survived': hard_Predict})
    soft_result.to_csv('../result/VotingASXRK_hard.csv', index=False)



def getResult(clf,tr_x, tr_y,te_x,gender_data,str):
    clf.fit(tr_x, tr_y)
    predict = clf.predict(te_x)
    result = pd.DataFrame({'PassengerId': gender_data['PassengerId'], 'Survived': predict})
    resultpath = '../result/' + str+ '_prediction.csv'
    result.to_csv(resultpath,index=False)





if __name__ == '__main__':
    trainData, testData = loadData()
    dataDescribe(trainData, testData )
    dataAnalysis(trainData)
    trainData, testData=dataPreprocess(trainData, testData)
    savemodels,savemodelpaths=simpleAlgorithm(trainData)
    savemodel(savemodels,savemodelpaths)
    # prediction(trainData, testData)
    # # predictionWithVoting(trainData, testData)
    predictionWithVotingASXRK(trainData, testData)