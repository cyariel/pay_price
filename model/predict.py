#处理数据，保证和训练集一样
import pandas as pd
import copy
from sklearn.externals import joblib


test = pd.read_csv('/Users/cy_ariel/Desktop/data/tap_fun_test.csv')
test['register_time_month'] = test.register_time.str[5:7]
test['register_time_day'] = test.register_time.str[8:10]
test = test.drop(['register_time'],axis=1)
test[['register_time_month','register_time_day']] = test[['register_time_month','register_time_day']].apply(pd.to_numeric)
test['register_time_count'] = test['register_time_month'] * 31 + test['register_time_day']

#保存前7天付款的用户
test_pay_7 = test[test['pay_price']>0]
test_pay_7.to_csv('/Users/cy_ariel/Desktop/data/test_pay_7.csv')

#保存前7天未付款的用户
test_nopay_7 = test[test['pay_price']==0]
test_nopay_7.to_csv('/Users/cy_ariel/Desktop/data/test_nopay_7.csv')

#test_nopay_7部分结果
sub_test_nopay_7 = test_nopay_7[['user_id','pay_price']]
sub_test_nopay_7 = sub_test_nopay_7.rename(columns={'pay_price':'prediction_pay_price'})
sub_test_nopay_7.to_csv('/Users/cy_ariel/Desktop/data/sub_test_nopay_7.csv')

#对test_pay_7分类
x_test_pay_7 = test_pay_7.drop('user_id',axis=1)
gradient_boosting_classifier = joblib.load('/Users/cy_ariel/Desktop/data/gradient_boosting_classifier.model')
y_test_pay_7 = gradient_boosting_classifier.predict(x_test_pay_7)

#将预测结果y_test_pay_7转为DataFrame
y_test_pay_7 = pd.DataFrame(y_test_pay_7,columns={'test_label'})

#将预测结果和test_pay_7合并，需要先将test_pay_7的index从0开始
columns_test = test_pay_7.columns
test_pay_7 = test_pay_7.values
test_pay_7 = pd.DataFrame(test_pay_7,columns=columns_test)
#合并
test_pay_7_pre = pd.concat([test_pay_7,y_test_pay_7],axis=1)

#test_pay_7_pay_45为继续付费的，test_pay_7_nopay_45为不再付费的
test_pay_7_pay_45 = copy.copy(test_pay_7_pre[test_pay_7_pre['test_label']==0])
test_pay_7_nopay_45 = copy.copy(test_pay_7_pre[test_pay_7_pre['test_label']==1])

#test_pay_7_nopay_45部分结果
sub_test_pay_7_nopay_45 = test_pay_7_nopay_45[['user_id','pay_price']]
sub_test_pay_7_nopay_45 = sub_test_pay_7_nopay_45.rename(columns={'pay_price':'prediction_pay_price'})
sub_test_pay_7_nopay_45.to_csv('/Users/cy_ariel/Desktop/data/sub_test_pay_7_nopay_45.csv')

#处理数据
tem = ['stone_add_value', 'magic_add_value', 'sr_infantry_tier_2_level', 'sr_cavalry_tier_2_level', 'sr_cavalry_tier_4_level', 'sr_hide_storage_level', 'sr_rss_c_prod_level', 'sr_outpost_tier_3_level', 'sr_outpost_tier_4_level', 'sr_rss_help_bonus_level', 'register_time_month']
test_pay_7_pay_45_tem = test_pay_7_pay_45.drop(['test_label','user_id'],axis=1)
test_pay_7_pay_45_tem = test_pay_7_pay_45_tem.drop(tem,axis=1)
gradient_boosting_regression = joblib.load('/Users/cy_ariel/Desktop/data/gradient_boosting_regression.model')

#test_pay_7_pay_45部分结果
y_test_pay_7_pay_45 = gradient_boosting_regression.predict(test_pay_7_pay_45_tem)




