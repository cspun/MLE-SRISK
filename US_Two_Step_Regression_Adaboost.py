import tensorflow as tf
import numpy as np
import pandas as pd
import math
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tensorflow.contrib.layers import fully_connected
import tensorflow.contrib.tensor_forest as tforest
from multiprocessing import Pool
from tensorflow.python.framework import ops
import functools
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn import preprocessing
import xgboost
from sklearn.ensemble import RandomForestRegressor
import gc
from sklearn.svm import SVR

tf.reset_default_graph()
tf.set_random_seed(0)
#random.seed(0)
np.random.seed(0)



#os.chdir(r'C:\Users\Administrator\Google 云端硬盘\SRISKwork\Comparison\Multi-level Regression')
os.chdir(r'H:\Google Drive\SRISKwork\Comparison\US Two-step regression')

market_raw = pd.read_csv('US Company Features.csv')
market_raw.index = pd.to_datetime(market_raw.date)
market_raw=market_raw.drop('date',1)

for fvariable in list(market_raw.columns.values):
        new_var=[]
        for j in market_raw[fvariable]:
            a=np.nan
            try:
                a=float(j)
            except:
                pass
            new_var.append(a)
        market_raw.loc[:,fvariable] = new_var

def cal_VaR(start_date_dt):
    h=5
    beforeYears=3
    n_steps = 5 # the length of X data 
    n_inputs = 7   # the number of variables: 宏观数据sum6.csv文件中的C到I列的7个变量
    n_neurons = 5
    n_outputs = 1
    learning_rate = 0.001
    VaR_alpha = 0.05
    n_epochs = 200
    
    back_date = (start_date_dt-relativedelta(years=beforeYears,months=0,days=0)).strftime('%Y-%m-%d')
    start_date=start_date_dt.strftime('%Y-%m-%d')
    macro_hist_t = market_raw[back_date:start_date]
    # calculate 未来一周的sp500
#    ret_sp = macro_hist_t[['Date','sp500']]
#    ret_sp.index = pd.to_datetime(ret_sp.Date)
#    #weekret = pd.rolling_apply(1+ret_sp.sp500, 5, lambda x: np.prod(x)) - 1
#    weekret = (1+ret_sp.sp500).rolling(h).apply(lambda x: np.prod(x))-1
#    weekret_future = weekret.shift(-h)
    xydata=macro_hist_t
    #xydata = pd.concat([macro_hist_t.iloc[:,-7:],weekret_future],axis=1).dropna()
    xydata=xydata.dropna()
    
    xydata_train =xydata
    xdata = xydata_train[['Rhsi_t','X3T_change', 'change_slope', 'ted', 'cre_spread',
                          're_excess','equ_vol']].iloc[:-5,:]
    
    
    
    ydata = xydata_train[['Rhsih']].iloc[:-5,:]
    # xdata in the last dates of this period:xdata[t]
    xdata_predict = xydata[['Rhsi_t','X3T_change', 'change_slope', 'ted', 'cre_spread',
           're_excess','equ_vol']].iloc[xydata.shape[0]-1:xydata.shape[0],:]
    #xdata_predict=xdata_predict.drop(labels='Varh', axis=1)
    
    std_scale_x = preprocessing.StandardScaler().fit(xdata)
    xdata = std_scale_x.transform(xdata)
    xdata_predict  = std_scale_x.transform(xdata_predict)

    
    std_scale_y = preprocessing.StandardScaler().fit(ydata)
    ydata = std_scale_y.transform(ydata)

    xdata_startdate = xdata_predict
    
    
    
    
    tf.reset_default_graph()    
    X = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.float32, [None,1])
    
    weights_0 = tf.Variable(tf.random_normal([n_inputs,n_outputs], stddev=(1/tf.sqrt(float(n_inputs)))))
    bias_0 = tf.Variable(tf.random_normal([n_outputs]))
    out = tf.matmul(X,weights_0)+bias_0
    
    loss = tf.reduce_mean((y-out)*(VaR_alpha-1*tf.cast((y-out)<=0,tf.float32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    #### 3. Prepare the Xdata and Ydata 
    
    #### 4. Execute the computational graph
    #g = tf.Graph()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):        
            _na,fitted_values,loss_total=sess.run([training_op,out,loss], 
                feed_dict={X: xdata, y: ydata})    
        #### 5. predict the missing y values using the Xdata in last periods
        VaR_startdate = sess.run(out, feed_dict={X: xdata_startdate})
        #print (sess.run(weights_0[0,0]))
    tf.reset_default_graph()  
    VaR_startdate = std_scale_y.inverse_transform(VaR_startdate)
    print(VaR_startdate[0,0])
    
    return(VaR_startdate[0,0])

dates = list(market_raw['2003-01-03':].index)
p = Pool(7)
#calculate VaR using RNN model
#VaR_predict = list(p.map(cal_VaR,dates))
VaR_predict = list(map(cal_VaR,dates))
#VaR_predict=np.squeeze(VaR_predict,axis=2)

VaR_p_df = pd.DataFrame(np.array(VaR_predict),index = dates)
VaR_p_df.columns = ['VaR_hsi']
market_raw2 = pd.concat([market_raw,VaR_p_df],axis=1)

market_raw2.to_csv('2step Hong Kong Market Data with VaRs RandomForest.csv')


######### Step1: construc RNN to predict VaR of sp500 future reture ##################
### Read the external data files 
firmdata_raw = pd.read_csv('US Company Features.csv')
firmdata_raw.date = pd.to_datetime(firmdata_raw.date.astype(str))
firmdata_raw.index = firmdata_raw.date
firmcodes = np.unique(firmdata_raw.PERMNO) # the list of firm codes
for fvariable in ['VOL','RET']:
    new_var=[]
    for j in firmdata_raw[fvariable]:
        a=np.nan
        try:
            a=float(j)
        except:
            print(j)
            pass
        new_var.append(a)
    firmdata_raw[fvariable] = new_var





n_inputs_1 = 9   # the number of variables
n_inputs_2 = 7
num_layers_0=30
keep_prob=0.5
#n_neurons = 10
n_outputs = 1

numpy_array_1_1 = np.float32(np.random.normal(scale=1/math.sqrt(float(n_inputs_1)),size=(n_inputs_1,num_layers_0)))
numpy_array_bias_1_1 = np.float32(np.random.normal(scale=1/math.sqrt(float(n_inputs_1)),size=n_outputs))
numpy_array_1_2 = np.float32(np.random.normal(scale=1/math.sqrt(float(num_layers_0)),size=(num_layers_0,n_outputs)))
numpy_array_bias_1_2 = np.float32(np.random.normal(scale=1/math.sqrt(float(num_layers_0)),size=n_outputs))

numpy_array_2_1 = np.float32(np.random.normal(scale=1/math.sqrt(float(n_inputs_2)),size=(n_inputs_2,n_outputs)))
numpy_array_bias_2_1 = np.float32(np.random.normal(scale=1/math.sqrt(float(n_inputs_2)),size=num_layers_0))


# linear and ANN1 (30) hidden nodes and use last date result weight as initialization
def cal_MSE_firm_onedate_svr(start_date_dt,firm_macro_2,firstdate):
    h=5
    c = 1
    beforeYears=3
    n_steps = 5 # the length of X data 
    n_inputs_1 = 10   # the number of variables
    n_inputs_2 = 7
    num_layers_0=30
    keep_prob=0.5
    #n_neurons = 10
    n_outputs = 1
    learning_rate = 0.0001
    n_epochs = 400
#    global numpy_array_1_1
#    global numpy_array_bias_1_1
#    global numpy_array_1_2
#    global numpy_array_bias_1_2
#    global numpy_array_2_1
#    global numpy_array_bias_2_1
#    global numpy_array_2_2
#    global numpy_array_bias_2_2
        
    
    back_date =  (start_date_dt-relativedelta(years=beforeYears,months=0,days=0)).strftime('%Y-%m-%d')
    start_date=start_date_dt.strftime('%Y-%m-%d')
    firm_hist_t = firm_macro_2[back_date:start_date]

    xydata=firm_hist_t
    #xydata_train =xydata.dropna()
    xydata=xydata.dropna(subset=filter(lambda x: x not in ['Varh','VaR_sp500','VaR_sse','VaR_hsi',
                                                           'VaR_sti'],xydata.columns))
    
    xydata_train =xydata
#    xdata = xydata_train[['VOL', 'RET', 'X3T_change', 'change_slope', 'ted', 'cre_spread',
#           'STI','re_excess','equ_vol', 'VOL_2','RET_2','X3T_change_2','change_slope_2','ted_2','cre_spread_2','STI_2',
#           're_excess_2','equ_vol_2','Rsysh']].iloc[:-5,:]
    xdata = xydata_train[['VOL', 'RET', 'X3T_change', 'change_slope', 'ted', 'cre_spread',
           're_excess','equ_vol', 'VOL_2','RET_2','X3T_change_2','change_slope_2','ted_2','cre_spread_2','sp500',
           're_excess_2','equ_vol_2','Rsysh']].iloc[:-5,:]

    
    ydata = xydata_train[['Rjh']].iloc[:-5,:]

    
#    xdata_predict = xydata[['VOL', 'RET', 'X3T_change', 'change_slope', 'ted', 'cre_spread',
#           'STI','re_excess','equ_vol', 'VOL_2','RET_2','X3T_change_2','change_slope_2','ted_2','cre_spread_2','STI_2',
#           're_excess_2','equ_vol_2','Rsysh','Varh']].iloc[xydata.shape[0]-1:xydata.shape[0],:]
    xdata_predict = xydata[['VOL', 'RET', 'X3T_change', 'change_slope', 'ted', 'cre_spread',
           're_excess','equ_vol', 'VOL_2','RET_2','X3T_change_2','change_slope_2','ted_2','cre_spread_2','sp500',
           're_excess_2','equ_vol_2','Rsysh','Varh']].iloc[xydata.shape[0]-1:xydata.shape[0],:]


    
    if not math.isnan(xdata_predict.Varh):
        xdata_predict.Rsysh=xdata_predict.Varh
    
    #drop column Varh
    xdata_predict=xdata_predict.drop(labels=['Varh'], axis=1)
    
    std_scale_x = preprocessing.StandardScaler().fit(xdata)
    xdata = std_scale_x.transform(xdata)
    xdata_predict  = std_scale_x.transform(xdata_predict)

    
    std_scale_y = preprocessing.StandardScaler().fit(ydata)
    ydata = std_scale_y.transform(ydata)

    

    
    xdata_startdate = xdata_predict

    np.random.seed(0)
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    
    regr.fit(xdata,ydata.ravel())
    MES_startdate=regr.predict(xdata_startdate)
    
    
    MES_startdate = std_scale_y.inverse_transform(MES_startdate)
    print(MES_startdate)

    
    if MES_startdate[0]<-1:
        MES_startdate[0]=-1
    #if MES_startdate[0,0]>0:
    #    MES_startdate[0,0]=0
    return (-MES_startdate[0])
    
    



def cal_MSE_firm(firmcode,beforeYears=3,firmdata_raw= firmdata_raw):
    #### get the firm data
    beforeYears=3
    firmdata_raw= firmdata_raw
    firm_raw = firmdata_raw.loc[firmdata_raw.PERMNO==firmcode] 
    #firm_raw = firm_raw[['VOL','RET']]
    firm_raw = firm_raw.drop(["PERMNO", "TICKER","date_number", "date",'COMNAM','PRC'], axis=1)
    for fvariable in list(firm_raw.columns.values):
        new_var=[]
        for j in firm_raw[fvariable]:
            a=np.nan
            try:
                a=float(j)
            except:
                pass
            new_var.append(a)
        firm_raw.loc[:,fvariable] = new_var
#    firm_raw.loc[:,'date_dt'] = firm_raw.index
#    macro_raw2.loc[:,'date_dt'] = macro_raw2.index
#    firm_macro = pd.merge(firm_raw.dropna(),macro_raw2.iloc[:,-2:],how='left',on='date_dt')    
    firm_macro_2 = firm_raw
    firm_macro_Friday=firm_macro_2[(firm_macro_2['weekdaynum'] == 6)]
    #firm_macro.index = pd.to_datetime( firm_macro['date_dt'])
    #firm_macro = firm_macro.dropna().astype(np.float32)
    #### extrect the dates of month end
    #del firm_macro['date_dt']
    data_startdates = firm_macro_2.index[0]
    
    if data_startdates<datetime.strptime('20000101', "%Y%m%d"):
        data_startdates=datetime.strptime('20000101', "%Y%m%d")
    #beforeYears=3
    mes_Startdates = (data_startdates+relativedelta(years=beforeYears,months=0,days=0)).strftime('%Y-%m-%d')
    
    dayli_dates = list(firm_macro_Friday[mes_Startdates:].index)
#    date_df = pd.DataFrame(index=dayli_dates)
#    date_df['year'] = date_df.index.year
#    date_df['month'] = date_df.index.month
#    date_df = date_df.drop_duplicates(['year','month'],keep='last')
    dates = dayli_dates
    firstdate = dates [0]
    #### extrect the dates of month end
#    dayli_dates = list(firm_macro['2002-01-04':].index)
#    date_df = pd.DataFrame(index=dayli_dates)
#    date_df['year'] = date_df.index.year
#    date_df['month'] = date_df.index.month
#    date_df = date_df.drop_duplicates(['year','month'],keep='last')
#    dates = list(date_df.index) 
    #start_date_dt=dates[0]
    #### calculate the MES using RNN model for each date
    #p = Pool(7)
    MSE_predict=[]
    dates_str=[]
    for i in range(len(dates)):
        
        new=cal_MSE_firm_onedate_svr(firm_macro_2=firm_macro_2, firstdate=firstdate,start_date_dt=dates[i])
        MSE_predict.append(new)
        dates_str.append(np.int(dates[i].strftime('%Y%m%d')))
        firm_MES = pd.DataFrame()
        firm_MES['companyID'] =firmcode
        firm_MES['YearMonth'] = dates_str
        firm_MES['companyID'] =firmcode
        firm_MES['MES']=MSE_predict
        path=r'H:\Google Drive\SRISKwork\Comparison\US Two-step regression\MES'
        firm_MES.to_csv(os.path.join(path,r'US_2Step_SVR_singlefirm_%d.csv'%firmcode))
        gc.collect()
    #MSE_predict = list(map(functools.partial(cal_MSE_firm_onedate_linear_lstm3,firm_macro_2=firm_macro_2, firstdate=firstdate),dates))
    #dates_str = [np.int(d.strftime('%Y%m%d')[:6]) for d in dates]
    dates_str = [np.int(d.strftime('%Y%m%d')) for d in dates]
    #### output the MES data of this firm
    firm_MES = pd.DataFrame()
    firm_MES['companyID'] =firmcode
    firm_MES['YearMonth'] = dates_str
    firm_MES['companyID'] =firmcode
    firm_MES['MES']=MSE_predict
    return(firm_MES)


#firmcode = firmcodes[0]
#mes=cal_MSE_firm(firmcode)
path=r'H:\Google Drive\SRISKwork\Comparison\US Two-step regression\MES'
all_mes=[]
for firmcode in firmcodes:
    try:
        mes = cal_MSE_firm(firmcode)
        mes.to_csv(os.path.join(path,r'US_2Step_SVR_singlefirm_%d.csv'%firmcode))
        all_mes.append(mes)
        firm_MES_all = pd.concat(all_mes)
        firm_MES_all.to_csv('US_2Step_SVR_test1.csv')
    except:
        pass
 
firm_MES_all = pd.concat(all_mes)
firm_MES_all.to_csv('US_2Step_SVR_test1.csv')
       
#firm_MES_all = pd.concat([cal_MSE_firm(firmcode) for firmcode in firmcodes])
#firm_MES_all.to_csv('firm_MES_all.csv')

