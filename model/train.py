import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

#数据读取
train = pd.read_csv("/Users/cy_ariel/Desktop/data/tap_fun_train.csv", parse_dates=True)

#处理register_time （原始格式2018-02-02 19:47:15）
train['register_time_month'] = train.register_time.str[5:7]
train['register_time_day'] = train.register_time.str[8:10]
train = train.drop(['register_time'],axis=1)
train['register_time_count'] = train['register_time_month'] * 31 + train['register_time_day']

#前7天付费用户
train_pay_7 = copy.copy(train[train['pay_price']>0])
train_pay_7.to_csv ("/Users/cy_ariel/Desktop/data/train_pay_7.csv")

#分类前数据处理
#打标签
train_pay_7['7_45_same'] = (train_pay_7['pay_price'] == train_pay_7['prediction_pay_price'])
train_pay_7['7_45_same'] = train_pay_7['7_45_same'].map({True:1,False:0})

# 删掉不需要的字段，prediction_pay_price、user_id
train_pay_7 = train_pay_7.drop(['prediction_pay_price', 'user_id'],axis=1)

#训练分类模型
label = '7_45_same'
x = train_pay_7.loc[:,train_pay_7._columns != label]
y = train_pay_7.loc[:,train_pay_7._columns == label]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
data_train = pd.concat([x_train, y_train], axis=1)

#创建一个dataframe，然后对模型的效果进行记录。最后小结。
thresholds = [0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
thresholds_2 = thresholds[:]  #= thresholds,如果这样复杂是，浅复制，映射同一块内存
thresholds_2.append('time')

print(thresholds_2)
result_model_f1 = pd.DataFrame(index=thresholds_2)

print(result_model_f1)

import time
start = time.time()
print('start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_classifier.fit(X_train,y_train.values.ravel())

y_pred = gradient_boosting_classifier.predict(X_test.values)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(y_test,y_pred,title='Confusion matrix')

end = time.time()
print(end-start,'s')