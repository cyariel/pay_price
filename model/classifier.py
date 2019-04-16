import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

#数据读取
train = pd.read_csv("/Users/cy_ariel/Desktop/data/tap_fun_train.csv", parse_dates=True)

#处理register_time （原始格式2018-02-02 19:47:15）
train['register_time_month'] = train.register_time.str[5:7]
train['register_time_day'] = train.register_time.str[8:10]
train = train.drop(['register_time'],axis=1)
train[['register_time_month','register_time_day']] = train[['register_time_month','register_time_day']].apply(pd.to_numeric)
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
x = train_pay_7.loc[:,train_pay_7.columns != label]
y = train_pay_7.loc[:,train_pay_7.columns == label]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
data_train = pd.concat([x_train, y_train], axis=1)

#记录模型效果
thresholds = [0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
thresholds_2 = thresholds[:]
thresholds_2.append('time')
result_model_f1 = pd.DataFrame(index=thresholds_2)

import time
start = time.time()
print('start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_classifier.fit(x_train,y_train.values.ravel())

y_pred = gradient_boosting_classifier.predict(x_test.values)

# Plot non-normalized confusion matrix
plt.figure()

end = time.time()
print('end time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

print(end-start,'s')

#预测结果--概率
y_pred_proba = gradient_boosting_classifier.predict_proba(
    x_test.values)

#记录各阈值下的结果
result_model_f1['GradientBoostingClassifier'] = 0

for i in thresholds:
    y_test_predictions_high_recall = y_pred_proba[:, 1] > i
    f1_score(y_test, y_test_predictions_high_recall)

for i in thresholds:
    y_test_predictions_high_recall = y_pred_proba[:, 1] > i
    plt.figure(figsize=(4, 4))
    result_model_f1.loc[i, 'GradientBoostingClassifier'] = f1_score(y_test.values,
                                                                    y_test_predictions_high_recall)

result_model_f1.loc['time', 'GradientBoostingClassifier'] = end - start

print(result_model_f1)

#保存模型
from sklearn.externals import joblib

print(gradient_boosting_classifier)
joblib.dump(gradient_boosting_classifier, '/Users/cy_ariel/Desktop/data/gradient_boosting_classifier.model')
