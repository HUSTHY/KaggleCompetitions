
#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   functions.py
@Time    :   2019/7/14 10:56
@Desc    :

'''
import pandas as pd
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
import random
import datetime
import numpy as np
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from lightgbm.sklearn import LGBMRegressor as LGBM


def dataLoadandExplore( ):
    trainData=pd.read_csv('../data/train.csv')
    testData=pd.read_csv('../data/test.csv')
    storeData=pd.read_csv('../data/store.csv')
    # print(trainData.info())
    # print(testData.info())
    # print(storeData.info())
    # print(trainData.head())
    # print(trainData.tail())
    # print(trainData.describe())
    # print(storeData.head())
    # print(storeData.tail())
    # print(storeData.describe())
    # print(trainData.isnull().sum())
    # print(testData.isnull().sum())
    # print(testData[testData['Open'].isnull()])#这里的控制应该填充为1--正常上班


    testData.fillna(1,inplace=True)#这里的空值应该填充为1--正常上班


    # print(storeData.isnull().sum())
    storeData.fillna(0,inplace=True)
    # print(storeData.isnull().sum())


    train=pd.merge(trainData,storeData,how='left',on='Store')#合并数据集--相当于数据库的内外连接
    test=pd.merge(testData,storeData,how='left',on='Store')#合并数据集--相当于数据库的内外连接


    return train,test

def dataAnalysis(trainData):
    import matplotlib.pyplot as plt
    from itertools import cycle
    import matplotlib.dates as mdates
    colors=cycle(['red','blue','green','yellow','black','purple'])
    for color,i in zip(colors,range(5)):
        plt.figure(figsize=(20, 10))
        store=random.randint(1,1115)
        train=trainData.copy()
        train['Date']=pd.to_datetime(train['Date'],format='%Y/%m/%d')
        # train=trainData[(train['Store']==store) & (train['Open']==1)&(train['Sales']>0)&
        #                 (train['Date']>=datetime.datetime(2014, 6, 1))&(train['Date']<=datetime.datetime(2014, 7, 30))]
        # train = trainData[(train['Store'] == store) & (train['Open'] == 1) & (train['Sales'] > 0)
        #                   & (train['Date'] >= datetime.datetime(2014, 8, 1)) & (train['Date'] <= datetime.datetime(2014, 9, 30))]
        train = trainData[(train['Store'] == store) & (train['Open'] == 1) & (train['Sales'] > 0)
                          & (train['Date'] >= datetime.datetime(2014, 6, 1)) & (
                                      train['Date'] <= datetime.datetime(2014, 9, 30))]
        # #配置横坐标的日期
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        # plt.xticks(pd.date_range('2014-6-1', '2014-9-30', freq='1d'))  # 横坐标日期范围及间隔
        label='Store'+str(store)
        plt.plot(train['Date'],train['Sales'],label=label,color=color)
        plt.legend()
    plt.show()

def dataAndFeatureProcess(traindata,testdata):
    train=traindata.copy().sort_values(['Date'],ascending=False)
    test = testdata.copy().sort_values(['Date'],ascending=False)
    hold_out_test=train[0:6*7*1115]#1115家店，8周，每周7天的数据，代表的是6/7月的数据
    hold_out_train = train[6
                           * 7 * 1115:]
    hold_out_train=hold_out_train[(hold_out_train['Open']!=0) & (hold_out_train['Sales']>0)]
    hold_out_test = hold_out_test[(hold_out_test['Open'] != 0) & (hold_out_test['Sales'] > 0)]
    hold_out_train=featureEngineering(hold_out_train)
    hold_out_test=featureEngineering(hold_out_test)
    test=featureEngineering(test)

    # print(hold_out_test.groupby('Store').count())

    return hold_out_train,hold_out_test,test

    return test
def featureEngineering(data):
    data.loc[:,'Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    mappings = {
        '0': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4
    }
    #分类量多值化
    data['StoreType']=data['StoreType'].map(lambda x: 0 if x=='0' else 1 if x=='a' else 2 if x=='b' else 3 if x=='c'else 4)
    data['Assortment']=data.Assortment.map(lambda x: 0 if x=='0' else 1 if x=='a' else 2 if x=='b' else 3 if x=='c'else 4)
    data['StateHoliday'] = data.StateHoliday.map(
        lambda x: 0 if x == '0' else 1 if x == 'a' else 2 if x == 'b' else 3 if x == 'c' else 4)
    # data.StoreType.replace(mappings,inplace=True)
    # data.Assortment.replace(mappings,inplace=True)
    # data.StateHoliday.replace(mappings,inplace=True)

    #日期拆分
    data.loc[:,'Year']=data.Date.dt.year
    data.loc[:,'month']=data.Date.dt.month
    data.loc[:,'Day']=data.Date.dt.day
    data.loc[:,'DayOfWeek']=data.Date.dt.dayofweek
    data.loc[:,'weekOfYear']=data.Date.dt.weekofyear

    #构建新的特征：竞争对手店面已经营业时间、店铺促销时间
    data.loc[:,'CompetitionOpenMonths']=12*(data.Year-data.CompetitionOpenSinceYear)+(data.month-data.CompetitionOpenSinceMonth)
    data.loc[:,'CompetitionPromoOpenMonths']=12*(data.Year-data.Promo2SinceYear)+(data.weekOfYear-data.Promo2SinceWeek)/4.0
    # data.loc[:,'CompetitionPromoOpenMonths'] =data.apply(lambda row: compare(row['CompetitionOpenMonths'],row['CompetitionPromoOpenMonths']),axis=1)#店铺促销时间小于等于营业时间

    monthnumtostr={
        1:'Jan',
        2:'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sept',
        10:'Oct',
        11: 'Nov',
        12: 'Dec'
    }

    data.loc[:,'monthstr']=data.month.map(monthnumtostr)
    data.loc[data.PromoInterval==0,'PromoInterval']=''
    data.loc[:,'isPromoMonth']=0
    for interval in data.PromoInterval.unique():
        if interval!='':
            for month in interval.split(','):
                    data.loc[(data.monthstr==month)&(data.PromoInterval==interval),'isPromoMonth']=1
    return data


def compare(a,b):
    if a<b:
        return a
    else:
        return b

def RMSPE(y_true,y_predict):

    return np.sqrt(np.mean((y_predict/y_true-1)**2))

def modeling(hold_out_train,hold_out_test):
    from sklearn.linear_model import LassoCV as LaCV
    from sklearn.ensemble import RandomForestRegressor as RFR

    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RANSACRegressor
    from sklearn.neural_network import MLPRegressor as MLP
    from xgboost.sklearn import XGBRegressor as XGBR
    from xgboost.sklearn import DMatrix
    from lightgbm.sklearn import LGBMRegressor as LGBM

    traindata=hold_out_train.copy()
    testdata=hold_out_test.copy()
    traindata=traindata.drop(['Store','Customers','Date','Open','PromoInterval','monthstr'],axis=1)
    testdata=testdata.drop(['Store','Customers','Date','Open','PromoInterval','monthstr'],axis=1)
    train_x=traindata.drop(['Sales'],axis=1)
    train_y=np.log1p(traindata['Sales'])
    test_x=testdata.drop(['Sales'],axis=1)

    # lacvmodel=LaCV(cv=4)
    # print(lacvmodel)
    # lacvmodel.fit(train_x,train_y)
    # print(lacvmodel.coef_)

    # time1=time.time()
    # rfrmodel=RFR(n_estimators=500,max_depth=4,oob_score=True,random_state=10)
    # rfrmodel.fit(train_x,train_y)
    # print(rfrmodel.oob_score_)
    # print(rfrmodel.feature_importances_)
    # sales_predict=rfrmodel.predict(test_x)
    # rmspe=RMSPE(testdata['Sales'],np.expm1(sales_predict))
    # print(rmspe)
    # time2 = time.time()
    # print('耗费时间：', (time2 - time1))



    # for alpha in [0.01,0.1,1.0,10,100]:
    #     time1 = time.time()
    #     ridgeModel=Ridge(alpha=alpha)
    #     ridgeModel.fit(train_x, train_y)
    #     print(ridgeModel)
    #     print(ridgeModel.coef_)
    #     sales_predict = ridgeModel.predict(test_x)
    #     rmspe = RMSPE(testdata['Sales'], np.expm1(sales_predict))
    #     print(rmspe)
    #     time2 = time.time()
    #     print('耗费时间：', (time2 - time1))

    # time1 = time.time()
    # RANSACModel = RANSACRegressor()
    # RANSACModel.fit(train_x, train_y)
    # print(RANSACModel)
    # sales_predict = RANSACModel.predict(test_x)
    # rmspe = RMSPE(testdata['Sales'], np.expm1(sales_predict))
    # print(rmspe)
    # time2 = time.time()
    # print('耗费时间：', (time2 - time1))



    # time1 = time.time()
    # mlpModel = MLP(hidden_layer_sizes=(19,38,19),max_iter=100000)
    # print(mlpModel)
    # mlpModel.fit(train_x, train_y)
    # sales_predict = mlpModel.predict(test_x)
    # rmspe = RMSPE(testdata['Sales'], np.expm1(sales_predict))
    # print(rmspe)
    # time2 = time.time()
    # print('耗费时间：', (time2 - time1))


    n_estimators=range(500,501)#表示多少个基模型

    best_sales_predict=0
    smalles_rmspe=1000
    depths=range(9,11)
    # for n_estimator in n_estimators:
    time1 = time.time()
    xgbrModel = XGBR(max_depth=10,n_estimators=500,colsample_bytree=0.7,subsample=0.8,seed=10,gamma=0.1)
    print(xgbrModel)
    xgbrModel.fit(train_x, train_y)
    sales_predict = xgbrModel.predict(test_x)
    rmspe = RMSPE(testdata['Sales'], np.expm1(sales_predict))
    print(rmspe)
    time2 = time.time()
    print('耗费时间：', (time2 - time1))

    return xgbrModel

def modelingLGBM(hold_out_train,hold_out_test):
    from sklearn.linear_model import LassoCV as LaCV
    from sklearn.ensemble import RandomForestRegressor as RFR

    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RANSACRegressor
    from sklearn.neural_network import MLPRegressor as MLP
    from xgboost.sklearn import XGBRegressor as XGBR
    from xgboost.sklearn import DMatrix
    from lightgbm.sklearn import LGBMRegressor as LGBM

    traindata=hold_out_train.copy()
    testdata=hold_out_test.copy()
    traindata=traindata.drop(['Store','Customers','Date','Open','PromoInterval','monthstr'],axis=1)
    testdata=testdata.drop(['Store','Customers','Date','Open','PromoInterval','monthstr'],axis=1)
    train_x=traindata.drop(['Sales'],axis=1)
    train_y=np.log1p(traindata['Sales'])
    test_x=testdata.drop(['Sales'],axis=1)

    # #归一化
    # min_max_scaler = MinMaxScaler()
    # train_x = min_max_scaler.fit_transform(train_x)
    # test_x = min_max_scaler.fit_transform(test_x)

    smalest_rmspe=1000
    subsamples=np.arange(0.5,0.6,0.1)
    for subsample in subsamples:
        time1 = time.time()
        lgbmModel = LGBM(n_estimators=8000,subsample=0.8)
        print(lgbmModel)
        lgbmModel.fit(train_x, train_y)
        sales_predict = lgbmModel.predict(test_x)
        rmspe = RMSPE(testdata['Sales'], np.expm1(sales_predict))
        print(rmspe)
        time2 = time.time()
        print('耗费时间：', (time2 - time1))
        if smalest_rmspe>rmspe:
            smalest_rmspe=rmspe
            best_model=lgbmModel
    return best_model


def predictionlgbm(hold_out_test,test):
    testdata = hold_out_test.copy()
    testdata = testdata.drop(['Store', 'Customers', 'Date', 'Open', 'PromoInterval', 'monthstr'], axis=1)
    finaltest_x=test.copy().drop(['Id','Store', 'Date', 'Open', 'PromoInterval', 'monthstr'], axis=1)
    ho_test_x = testdata.drop(['Sales'], axis=1)

    # #归一化
    # min_max_scaler=MinMaxScaler()
    # ho_test_x=min_max_scaler.fit_transform(ho_test_x)

    xgbrModel=joblib.load('../model/lgbmModel.m')




    sales_predict = xgbrModel.predict(ho_test_x)
    final_predict=xgbrModel.predict(finaltest_x)
    test['predict_sales']=np.expm1(final_predict)



    smallest_rmspe=RMSPE(testdata['Sales'], np.expm1(sales_predict))
    print(smallest_rmspe)
    hold_out_test['sales_predict'] = np.expm1(sales_predict)
    res = hold_out_test[['Store', 'Date', 'Sales', 'sales_predict']]
    res2 = hold_out_test[['Store', 'Date', 'Sales', 'sales_predict']]

    showFigure(res)

    res.loc[:,'errorabs'] = abs((res['sales_predict'] - res['Sales'])/res['Sales'])
    res.loc[:,'error'] = ((res['sales_predict'] - res['Sales'])/res['Sales'])
    res.sort_values(['errorabs'], ascending=False, inplace=True)
    # print(res[res['error']>=0].count())
    # print(res[res['error'] <= 0].count())
    start=time.time()
    b_w=0.900
    for i in range(1,101):
        predict=sales_predict*(0.900+i/1000)
        rmspe=RMSPE(testdata['Sales'], np.expm1(predict))
        if rmspe<smallest_rmspe:
            b_w=0.900+i/1000
            smallest_rmspe=rmspe
            res2.loc[:,'sales_predict']= np.expm1(predict)
    end=time.time()
    print('模型校准耗时：%.4f'%(end-start))
    print(smallest_rmspe)
    print(b_w)
    showFigure(res2)

    stores=range(1,1116)
    hold_out_test['w']=1
    start = time.time()
    for store in stores:
        s1=pd.DataFrame(hold_out_test[hold_out_test['Store']==store],columns=['Store', 'Date', 'Sales', 'sales_predict'])
        s=[]
        for i in range(1, 401):
            error=RMSPE(s1.Sales,s1.sales_predict*(0.800+i/1000))
            s.append(error)
        score=pd.Series(s,index=[(0.800+i/1000) for i in range(1, 401)])
        BS=score[score.values==score.values.min()]
        a=np.array(BS.index.values)
        hold_out_test.loc[hold_out_test['Store']==store,'w']=a
        test.loc[test['Store']==store,'w']=a

    res3=hold_out_test[['Store', 'Date', 'Sales','sales_predict','w']]
    res3['sales_predict']=hold_out_test['sales_predict']*hold_out_test['w']
    end = time.time()
    print('模型精确校准耗时：%.4f' % (end - start))
    print(RMSPE(res3['Sales'],res3['sales_predict']))
    showFigure(res3)

    finalres=test[['Id']]
    finalres['Sales']=test['predict_sales']*test['w']
    finalres.to_csv('../submissionResult/submissionResult_lightGBM.csv',index=False)








def savemodel(model,path):
    joblib.dump(model,path)



def showFigure(df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,15))
    store=355
    plt.plot(df[df['Store']==store]['Date'],df[df['Store']==store]['Sales'],label='True')
    plt.plot(df[df['Store'] == store]['Date'], df[df['Store']==store]['sales_predict'], label='predict_Sales')
    plt.legend()
    plt.show()


def modelIntergrated( hold_out_train,hold_out_test,test):

    traindata = hold_out_train.copy().drop(['Store', 'Customers', 'Date', 'Open', 'PromoInterval', 'monthstr'], axis=1)
    train_x = traindata.drop(['Sales'], axis=1)
    train_y = np.log1p(traindata['Sales'])

    testdata = hold_out_test.copy()
    testdata = testdata.drop(['Store', 'Customers', 'Date', 'Open', 'PromoInterval', 'monthstr'], axis=1)
    ho_test_x = testdata.drop(['Sales'], axis=1)
    ho_test_y=testdata['Sales']

    finaltest_x = test.copy().drop(['Id', 'Store', 'Date', 'Open', 'PromoInterval', 'monthstr'], axis=1)

    predictions=[]
    RMSPES=[]
    start = time.time()
    k=5
    for i in range(0,k):
        lgbmModel = LGBM(n_estimators=8000, subsample=0.8,random_state=i)
        lgbmModel.fit(train_x,train_y)
        sales_predict = lgbmModel.predict(ho_test_x)
        final_predict = lgbmModel.predict(finaltest_x)
        test['sales_predict'] = np.expm1(final_predict)

        smallest_rmspe = RMSPE(testdata['Sales'], np.expm1(sales_predict))
        print(smallest_rmspe)
        hold_out_test['sales_predict'] = np.expm1(sales_predict)
        res = hold_out_test[['Store', 'Date', 'Sales', 'sales_predict']]
        # res2 = hold_out_test[['Store', 'Date', 'Sales', 'sales_predict']]

        # showFigure(res)

        res.loc[:, 'errorabs'] = abs((res['sales_predict'] - res['Sales']) / res['Sales'])
        res.loc[:, 'error'] = ((res['sales_predict'] - res['Sales']) / res['Sales'])
        res.sort_values(['errorabs'], ascending=False, inplace=True)
        # print(res[res['error']>=0].count())
        # print(res[res['error'] <= 0].count())
        # b_w = 0.900
        # for i in range(1, 101):
        #     predict = sales_predict * (0.900 + i / 1000)
        #     rmspe = RMSPE(testdata['Sales'], np.expm1(predict))
        #     if rmspe < smallest_rmspe:
        #         b_w = 0.900 + i / 1000
        #         smallest_rmspe = rmspe
        #         res2.loc[:, 'sales_predict'] = np.expm1(predict)
        # print(smallest_rmspe)
        # print(b_w)
        # showFigure(res2)

        stores = range(1, 1116)
        hold_out_test['w'] = 1
        for store in stores:
            s1 = pd.DataFrame(hold_out_test[hold_out_test['Store'] == store],columns=['Store', 'Date', 'Sales', 'sales_predict'])
            s = []
            for i in range(1, 201):
                error = RMSPE(s1.Sales, s1.sales_predict * (0.800 + i / 1000))
                s.append(error)
            score = pd.Series(s, index=[(0.800 + i / 1000) for i in range(1, 201)])
            BS = score[score.values == score.values.min()]
            a = np.array(BS.index.values)
            hold_out_test.loc[hold_out_test['Store'] == store, 'w'] = a
            test.loc[test['Store'] == store, 'w'] = a

        res3 = hold_out_test[['Store', 'Date', 'Sales', 'sales_predict', 'w']]
        res3['sales_predict'] = hold_out_test['sales_predict'] * hold_out_test['w']
        RMSPES.append(RMSPE(res3['Sales'], res3['sales_predict']))

        finalres = test[['Id']]
        finalres['Sales'] = test['predict_sales'] * test['w']
        predictions.append(finalres['Sales'])

    print(RMSPES)
    finalres = test[['Id']]
    finalres['Sales']=0
    for i in range(0,k):
        finalres['Sales']+=predictions[i]
    finalres['Sales']=finalres['Sales']/k
    end = time.time()
    print((end-start))
    finalres.to_csv('../submissionResult/submissionResult_lightGBM_mean.csv', index=False)




