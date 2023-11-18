#-----------------資料前處理----------------------------------------#
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder

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



from sklearn.preprocessing import OneHotEncoder

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
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)

#---------------------隨機森林-----------------------------------#
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

names = ['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','country','income']

forest = RandomForestRegressor(n_estimators=25, 
                                random_state=1,
                                n_jobs=2,
                                min_samples_leaf=2)
forest.fit(X_train, y_train)
ff = forest.fit(X_train_std, y_train)
s = sorted(zip(map(lambda x: round(x,4), ff.feature_importances_), names),reverse=True)
f = forest.predict(X_test_std)
#print('score: '+str(forest.score(X_test_std, y_test)))

# X_combined = np.vstack((X_train, X_test))
# y_combined = np.hstack((y_train, y_test))

RMSE = metrics.mean_squared_error(f, y_test)**0.5
MAE = metrics.mean_absolute_error(f, y_test)
MAPE = metrics.mean_absolute_percentage_error(f, y_test)
# MAPE = np.mean(np.abs((f-y_test)/f)) * 100
print('隨機森林: ')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))

#print(s)
#---------------------XGBoost-------------------------------#

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
names = ['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','country','income']

xgb = XGBRegressor(learning_rate=0.1,
                    n_estimators=500, 
                    gamma=0,
                    reg_alpha=0,
                    reg_lambda=1,
                    max_depth=5,
                    min_child_weight=1)


xx = xgb.fit(X_train, y_train)
s = sorted(zip(map(lambda x: round(x,4), xx.feature_importances_), names),reverse=True)
x = xgb.predict(X_test)
print('score: '+str(xgb.score(X_test, y_test)))


RMSE = metrics.mean_squared_error(x, y_test)**0.5
MAE = metrics.mean_absolute_error(x, y_test)
MAPE = metrics.mean_absolute_percentage_error(x, y_test)
print('XGBoost:')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))



#---------------------SVR-----------------------------------#

from sklearn.svm import SVR
from sklearn import svm
from sklearn.inspection import permutation_importance
from matplotlib import *
import matplotlib.pyplot as pyplot

svr = SVR(epsilon=0.2, C=1.0)
#因為反覆计算都計算不出最優超平面，所以將模型中的參數max_iter設置一下，默认为-1，也就是計算出直到收斂的平面時，改成較小的次數就可以解決。
svr = svm.SVR(epsilon=0.2, C=1.0, max_iter=1000)

svr.fit(X_train_std, y_train)
s = svr.predict(X_test_std)
print('SVR score:')
print(svr.score(X_test_std, y_test))


RMSE = metrics.mean_squared_error(s, y_test)**0.5
MAE = metrics.mean_absolute_error(s, y_test)
MAPE = metrics.mean_absolute_percentage_error(s, y_test)

print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))

model = SVR(max_iter=100)
# fit the model
model.fit(X_train_std, y_train)
# perform permutation importance
results = permutation_importance(model, X_train_std, y_train, scoring='neg_mean_squared_error')
# get importance
importance = results.importances_mean
# summarize feature importance
##for i,v in enumerate(importance):
##    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#---------------------KNN-----------------------------------#

from sklearn.preprocessing import StandardScaler
#標準化
ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics


#p=2歐式距離

knn = KNeighborsRegressor(n_neighbors=10,
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)
k = knn.predict(X_test_std)
print('knn score:')
print(knn.score(X_test_std, y_test))


RMSE = metrics.mean_squared_error(k, y_test)**0.5
MAE = metrics.mean_absolute_error(k, y_test)
MAPE = metrics.mean_absolute_percentage_error(k, y_test)
print('KNN: ')
print('RMSE: '+str(RMSE))
print('MAE: '+str(MAE))
print('MAPE: '+str(MAPE))

#------------------------------------------------------------#