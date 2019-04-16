import pandas as pd
import copy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import Lasso

train_pay_7 = pd.read_csv("/Users/cy_ariel/Desktop/data/train_pay_7.csv", index_col=0, parse_dates=True)
train_pay_7_pay_45 = copy.copy(train_pay_7[train_pay_7['pay_price']<train_pay_7['prediction_pay_price']])

#删掉user_id
train_pay_7_pay_45 = train_pay_7_pay_45.drop([ 'user_id'],axis=1)
label = 'prediction_pay_price'


x = train_pay_7_pay_45.loc[:, train_pay_7_pay_45.columns != label]
y = train_pay_7_pay_45.loc[:, train_pay_7_pay_45.columns == label]

#lasso解决共线性问题
model = Lasso()  # 分析了训练数据，存在大量共线，可使用L1正则化消除共线
model.fit(x, y)
print(model.coef_)
print(len(model.coef_))

tem = []
for i in range(len(model.coef_)):
    if abs(model.coef_[i]) < 1e-06:
        tem.append(x.columns[i])

print(tem)
print(len(tem))
x_final = x.drop(tem, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_final,y,test_size = 0.2, random_state = 0)


import time
start = time.time()
print('start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


gradient_boosting_regression = GradientBoostingRegressor()
gradient_boosting_regression.fit(x_train,y_train.values.ravel())
y_pred = gradient_boosting_regression.predict(x_test.values)

#The mean squared error
print("Root Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred) ** 0.5)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

end = time.time()
print(end-start,'s')
#保存模型
from sklearn.externals import joblib

print(gradient_boosting_regression)
joblib.dump(gradient_boosting_regression, '/Users/cy_ariel/Desktop/data/gradient_boosting_regression.model')
