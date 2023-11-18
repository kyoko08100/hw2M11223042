import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

train_title = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
test_title = ['ID','crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']
df = pd.read_csv(r'D:\Source\python\hw2M11223042\housing.csv',delimiter=r"\s+", names = train_title, header=None)

# 分割dataframe成訓練集跟測試集
df_train = df.loc[0:333]
# print(df_train)
X_train = df_train.drop(columns='medv')
Y_train = df_train['medv']
df_test = df.loc[334:]
X_test = df_test.drop(columns='medv')
Y_test = df_test['medv']



X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()

kf = KFold(5, shuffle=True)
for train_index,val_index in kf.split(X_train):
    xgb = XGBRegressor()
    x_fit = xgb.fit(X_train[train_index][:],Y_train[train_index][:])
    x = xgb.predict(X_train[val_index][:])
    print('score: '+str(xgb.score(X_train[val_index][:],Y_train[val_index][:])))