import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn import metrics

title = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
df = pd.read_csv(r'D:\Source\python\hw2M11223042\housing.csv',delimiter=r"\s+", names = title, header=None)

# 分割dataframe成訓練集跟測試集
df_train = df.loc[0:333]
# print(df_train)
X_train = df_train.drop(columns='medv')
Y_train = df_train['medv']
df_test = df.loc[334:]
X_test = df_test.drop(columns='medv')
Y_test = df_test['medv']



X_train_np = X_train.to_numpy()
Y_train_np = Y_train.to_numpy()
X_test_np = X_test.to_numpy()
kfold_score = []
i = 1

# Kfold k = 5
kf = KFold(5, shuffle=True)
for train_index,val_index in kf.split(X_train):
    xgb = XGBRegressor()
    x_fit = xgb.fit(X_train_np[train_index][:],Y_train_np[train_index][:])
    x = xgb.predict(X_train_np[val_index][:])
    score = xgb.score(X_train_np[val_index][:],Y_train_np[val_index][:])
    kfold_score.append((score))
    print('score'+ str(i) + ': ' + str(score))
    i += 1
print("avg:" + str(sum(kfold_score) / 5))

# 特徵刪除前
xgb = XGBRegressor()
x_fit = xgb.fit(X_train, Y_train)
x_pre = xgb.predict(X_test)
score = xgb.score(X_test,Y_test)
MAPE = metrics.mean_absolute_percentage_error(x_pre, Y_test)
RMSE = metrics.mean_squared_error(x_pre, Y_test)**0.5
R_square = metrics.r2_score(Y_test,x_pre)
print('score: ' + str(score)) # 特徵刪除前score
print('MAPE: ' + str(MAPE)) # 特徵刪除前MAPE
print('RMSE: ' + str(RMSE)) # 特徵刪除前RMSE
print('R_square: ' + str(R_square)) # 特徵刪除前R_square


# feature importances
x_fea_imp = x_fit.feature_importances_
x_title = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']
feature_importances = []
for i in range(len(x_fea_imp)):
    feature_importances.append([x_fea_imp[i], x_title[i]])
feature_importances.sort(key=lambda x:x[0])
print("特徵重要性排名(由小到大):")
print(feature_importances)

# 特徵刪除後(刪除zn、black)
X_train_del = X_train.drop(columns=['zn', 'black'])
X_test_del = X_test.drop(columns=['zn', 'black'])
xgb = XGBRegressor()
x_fit = xgb.fit(X_train_del, Y_train)
x_pre = xgb.predict(X_test_del)
score = xgb.score(X_test_del,Y_test)
MAPE = metrics.mean_absolute_percentage_error(x_pre, Y_test)
RMSE = metrics.mean_squared_error(x_pre, Y_test)**0.5
R_square = metrics.r2_score(Y_test,x_pre)
print('score: ' + str(score)) # 特徵刪除後score
print('MAPE: ' + str(MAPE)) # 特徵刪除後MAPE
print('RMSE: ' + str(RMSE)) # 特徵刪除後RMSE
print('R_square: ' + str(R_square)) # 特徵刪除後R_square