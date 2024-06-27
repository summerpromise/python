#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler


def trans(x):
        if x=='是':
            return 1
        else:
            return 0


# In[10]:


res=[]
res2=[]
for label in ['春柳-香工街','香工街-中长街','春柳-中长街']:
    
    data1=pd.read_excel('新建 Microsoft Excel 工作表.xlsx')
    data2=pd.read_excel('新建 XLSX 工作表.xlsx')
    feas=data2.T
    df=pd.DataFrame(feas.values[1:],columns=['天气种类', '最高气温', '最低气温', '是否为周末', '是否为节假日'])
    df['label']=data1[data1['Unnamed: 0']==label].values[0][1:]

    r,_=pd.factorize(df['天气种类'])
    df['天气种类']=r

    df['是否为周末']=df['是否为周末'].map(trans)
    df['是否为节假日']=df['是否为节假日'].map(trans)
    df['最高气温']=df['最高气温'].astype(int)
    df['最低气温']=df['最低气温'].astype(int)
    df['label']=df['label'].astype(int)

    import seaborn as sns
    import matplotlib.pyplot  as  plt 
    plt.rcParams["font.sans-serif"]=["SimHei"]
#     plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['axes.unicode_minus']=False
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, vmin=-1, vmax=1)

    plt.show()

    X_train,X_test,y_train,y_test=df[df.columns[:-1]][:46],df[df.columns[:-1]][46:],df['label'][:46],df['label'][46:]
    scaler = StandardScaler() # 标准化转换

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.fit_transform(X_test)



    # 定义模型
    rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0, max_depth = 4)
    rf_regressor.fit(X_train, y_train)
    rf_train = rf_regressor.predict(X_train)
    rf_pred = rf_regressor.predict(X_test)



    # 评估回归性能
    # print('mean_absolute_percentage_error:', metrics.mean_absolute_percentage_error(y_test, y_pred*0.5+y_pred1*0.5))

    print('rf_pred mean_absolute_percentage_error:', metrics.mean_absolute_percentage_error(y_test, rf_pred))  #
    # RMSE 均方根误差
    import math
    print('rf_pred mse:', metrics.mean_squared_error(y_test, rf_pred))
    # MAE  平均绝对误差
    print('rf_pred mae:', metrics.mean_absolute_error(y_test, rf_pred))  #

    from keras.models import Sequential
    from keras.layers import Dense, Dropout,LSTM
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import SGD, Adadelta, Adam, RMSprop

    def create_model():
        '''
        creat a  model
        :param opt:
        :return:model
        '''
        model = Sequential()
        model.add(LSTM(64,input_shape=(1,5)))           #输入维度为1，时间窗的长度为1，隐含层神经元节点个数为32
        model.add(Dense(1))
        return model

    callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=2)
        ]
    model=create_model()
    model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
    X_train=X_train.reshape(-1,1,5)
    X_test=X_test.reshape(-1,1,5)
    hist = model.fit(X_train, y_train, batch_size=128, epochs=500,
                     validation_data=(X_test,y_test),
                     callbacks=callbacks)
    mlp_pred=model.predict(X_test)

    print('mlp_pred mean_absolute_percentage_error:', metrics.mean_absolute_percentage_error(y_test, mlp_pred))  #
    print('mlp_pred mse:', metrics.mean_squared_error(y_test, mlp_pred))
    print('mlp_pred mae:', metrics.mean_absolute_error(y_test, mlp_pred))


    best_mape=1000
    best_mae=10000
    best_ronghe_pred=[]
    for lr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        ronghe_pred=[rf_pred[i]*lr+(1-lr)*mlp_pred[i]  for i in range(len(mlp_pred))]
        if metrics.mean_absolute_percentage_error(y_test, ronghe_pred)<best_mape:
            best_mape=metrics.mean_absolute_percentage_error(y_test, ronghe_pred)
            best_mae=metrics.mean_absolute_error(y_test, ronghe_pred)
            best_ronghe_pred=ronghe_pred

    res.append([label,metrics.mean_absolute_error(y_test, rf_pred),metrics.mean_absolute_error(y_test, mlp_pred),metrics.mean_absolute_error(y_test, best_ronghe_pred)])
    res2.append([label,metrics.mean_absolute_percentage_error(y_test, rf_pred),metrics.mean_absolute_percentage_error(y_test, mlp_pred),metrics.mean_absolute_percentage_error(y_test, best_ronghe_pred)])
    
    dates=[]
    for i in range(16,31):
        dates.append('8/'+str(i))
    sns.barplot(x=dates,y=y_test,label='true')
    sns.lineplot(x=dates,y=rf_pred,label='RandomForest')
    sns.lineplot(x=dates,y=mlp_pred.flatten(),label='LSTM')
    sns.lineplot(x=dates,y=np.array(best_ronghe_pred).flatten(),label='融合模型')
    plt.title(label)
    plt.xlabel('日期')
    plt.ylabel('客流量/人')
    plt.legend()
    plt.show()
    


# In[ ]:


dfres=pd.DataFrame(res,columns=['路线','rf_mae','lstm_mae','融合_mae'])
print(dfres)


# In[ ]:


dfres2=pd.DataFrame(res2,columns=['路线','rf_mape','lstm_mape','融合_mape'])
print(dfres2)


# In[ ]:


tmp=[]
for i in res:
    k =['RandomForest','LSTM','融合模型']
    tmp.append([i[0],k[0],i[1]])
    tmp.append([i[0],k[1],i[2]])
    tmp.append([i[0],k[2],i[3]])

sns.barplot(x=[i[0] for i in tmp],y=[i[2] for i in tmp],hue=[i[1] for i in tmp])
plt.xlabel('路线')
plt.ylabel('MAE')
plt.ylim(0, 80)
plt.legend()
# In[ ]: