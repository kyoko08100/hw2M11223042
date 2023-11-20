#-----------------資料前處理----------------------------------------#
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import time

df_X = pd.read_csv ('adult.train.txt', header=None)
df_Y = pd.read_csv ('adult.test.txt', header=None)

df_X.columns = ['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hr_per_week','country','income']
df_Y.columns = ['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hr_per_week','country','income']

df_X = df_X.drop(columns=['fnlwgt','education_num'])
df_Y = df_Y.drop(columns=['fnlwgt','education_num'])


df_X =df_X.drop(df_X[df_X['workclass']==" ?"].index)
df_X =df_X.drop(df_X[df_X['country']==" ?"].index)
df_X =df_X.drop(df_X[df_X['occupation']==" ?"].index)
df_X =df_X.drop(df_X[df_X['marital']==" ?"].index)

df_Y =df_Y.drop(df_Y[df_Y['workclass']==" ?"].index)
df_Y =df_Y.drop(df_Y[df_Y['country']==" ?"].index)
df_Y =df_Y.drop(df_Y[df_Y['occupation']==" ?"].index)
df_Y =df_Y.drop(df_Y[df_Y['marital']==" ?"].index)

# 做Label encoding
labelencoder = LabelEncoder()

df_X["workclass"] = labelencoder.fit_transform(df_X["workclass"].astype('string')) # 將"gender"裡的字串自動轉換成數值
df_Y["workclass"] = labelencoder.fit_transform(df_Y["workclass"].astype('string'))

df_X["education"] = labelencoder.fit_transform(df_X["education"].astype('string'))
df_Y["education"] = labelencoder.fit_transform(df_Y["education"].astype('string'))


df_X["marital"] = labelencoder.fit_transform(df_X["marital"].astype('string'))
df_Y["marital"] = labelencoder.fit_transform(df_Y["marital"].astype('string'))


df_X["occupation"] = labelencoder.fit_transform(df_X["occupation"].astype('string')) 
df_Y["occupation"] = labelencoder.fit_transform(df_Y["occupation"].astype('string')) 


df_X["relationship"] = labelencoder.fit_transform(df_X["relationship"].astype('string'))
df_Y["relationship"] = labelencoder.fit_transform(df_Y["relationship"].astype('string'))


df_X["race"] = labelencoder.fit_transform(df_X["race"].astype('string'))
df_Y["race"] = labelencoder.fit_transform(df_Y["race"].astype('string'))


df_X["sex"] = labelencoder.fit_transform(df_X["sex"].astype('string'))
df_Y["sex"] = labelencoder.fit_transform(df_Y["sex"].astype('string'))

df_X["country"] = labelencoder.fit_transform(df_X["country"].astype('string'))
df_Y["country"] = labelencoder.fit_transform(df_Y["country"].astype('string'))

df_X["income"] = labelencoder.fit_transform(df_X["income"].astype('string'))
df_Y["income"] = labelencoder.fit_transform(df_Y["income"].astype('string'))

y_train = df_X['hr_per_week']
X_train = df_X.drop(columns=['hr_per_week'])

y_test = df_Y['hr_per_week']
X_test = df_Y.drop(columns=['hr_per_week'])



from sklearn.preprocessing import StandardScaler
#標準化
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)

#---------------------隨機森林-----------------------------------#


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

start_time = time.time()
forest = RandomForestRegressor()
ff = forest.fit(X_train_std, y_train)
f = forest.predict(X_test_std)

RMSE = metrics.mean_squared_error(f, y_test)**0.5
MAE = metrics.mean_absolute_error(f, y_test)
MAPE = metrics.mean_absolute_percentage_error(f, y_test)

print('隨機森林: ')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))
end_time = time.time()
print("執行時間：" + str(end_time - start_time) + "秒")

#---------------------XGBoost-------------------------------#

from xgboost import XGBRegressor

start_time = time.time()
xgb = XGBRegressor()
xx = xgb.fit(X_train, y_train)
x = xgb.predict(X_test)

RMSE = metrics.mean_squared_error(x, y_test)**0.5
MAE = metrics.mean_absolute_error(x, y_test)
MAPE = metrics.mean_absolute_percentage_error(x, y_test)

print('XGBoost:')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))
end_time = time.time()
print("執行時間：" + str(end_time - start_time) + "秒")


#---------------------SVR-----------------------------------#

from sklearn.svm import SVR

start_time = time.time()
svr = SVR()
svr.fit(X_train_std, y_train)
s = svr.predict(X_test_std)

RMSE = metrics.mean_squared_error(s, y_test)**0.5
MAE = metrics.mean_absolute_error(s, y_test)
MAPE = metrics.mean_absolute_percentage_error(s, y_test)
print('SVR:')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))
end_time = time.time()
print("執行時間：" + str(end_time - start_time) + "秒")

#---------------------KNN-----------------------------------#

from sklearn.neighbors import KNeighborsRegressor

start_time = time.time()
knn = KNeighborsRegressor()
knn.fit(X_train_std, y_train)
k = knn.predict(X_test_std)

RMSE = metrics.mean_squared_error(k, y_test)**0.5
MAE = metrics.mean_absolute_error(k, y_test)
MAPE = metrics.mean_absolute_percentage_error(k, y_test)

print('KNN: ')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))
end_time = time.time()
print("執行時間：" + str(end_time - start_time) + "秒")

#------------------------------------------------------------#