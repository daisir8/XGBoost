#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# 数据加载
train_data = pd.read_csv('./used_car_train_20200313.csv', sep=' ')
train_data


# In[3]:


test = pd.read_csv("./used_car_testB_20200421.csv", sep = ' ')
test


# In[16]:


# 查看数据缺失值
temp = train_data.isnull().sum()
#temp = pd.DataFrame(temp)
#temp = temp.reset_index()
#temp[temp[0]>0]
temp = temp[temp>0]
temp


# In[17]:


temp.plot.bar()


# In[11]:


# model, bodyType, fuelType, gearbox 存在缺失值
train_data['SaleID'].nunique()
train_data.shape


# In[19]:


# null可视化
import matplotlib.pyplot as plt
import missingno as msno

plt.figure(figsize=(12, 8))
sample = train_data.sample(1000)
msno.matrix(sample)


# In[20]:


msno.bar(sample)


# In[21]:


msno.heatmap(sample)


# In[22]:


# 查看数据集的大小
print('训练集大小:', train_data.shape)
print('测试集大小:', test.shape)


# In[24]:


# 查看price的分布
import seaborn as sns
import scipy.stats as st
y = train_data['price']
plt.title('johnsonsu')
sns.distplot(y, kde=False, fit=st.johnsonsu)


# In[25]:


plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)


# In[26]:


plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)


# In[30]:


import numpy as np
plt.hist(train_data['price'], color ='red')
plt.show()
plt.hist(np.log(train_data['price']), color ='red')
plt.show()


# In[31]:


sns.distplot(train_data['price'])
print("Skewness: %f" % train_data['price'].skew())
print("Kurtosis: %f" % train_data['price'].kurt())


# In[35]:


plt.figure(figsize=(20, 16))
sns.heatmap(train_data.corr(), annot=True)


# In[39]:


temp = train_data.corr()
temp[np.abs(temp['price'])>=0.5]['price']


# In[42]:


# 可以去掉 offerType
#train_data.info()
train_data['offerType'].value_counts()


# In[43]:


# 一行代码生成数据报告
import pandas_profiling as pp
report = pp.ProfileReport(train_data)


# In[44]:


get_ipython().run_cell_magic('time', '', "# 导出为html\nreport.to_file('report.html')")


# # 缺失值补全

# In[45]:


train_data['notRepairedDamage'].value_counts()
#train_data['notRepairedDamage'].describe()


# In[48]:


train_data['notRepairedDamage'].replace('-', '0.0', inplace=True)
#train_data['notRepairedDamage'].mode()[0]


# In[50]:


test['notRepairedDamage'].value_counts()
test['notRepairedDamage'].replace('-', '0.0', inplace=True)


# In[ ]:


# 对power异常值进行处理
# 缺失值用-1补全
train_data = train_data.fillna(-1)
test = test.fillna(-1)


# In[53]:


# 查看数值类型
#train_data.info()
numerical_cols = train_data.select_dtypes(exclude='object').columns
numerical_cols


# In[54]:


# 查看分类类型
categorical_cols = train_data.select_dtypes(include='object').columns
categorical_cols


# In[55]:


#train_data['regDate'].value_counts()


# In[56]:


# 特征选择
drop_cols = ['SaleID', 'regDate', 'creatDate', 'offerType', 'price']
feature_cols = [col for col in train_data.columns if col not in drop_cols]
feature_cols


# In[58]:


# 提取特征列
X_data = train_data[feature_cols]
Y_data = train_data['price']
X_test = test[feature_cols]


# In[61]:


# 定一个统计函数，用于统计某字段的特征
def show_stats(data):
    print('min: ', np.min(data))
    print('max: ', np.max(data))
    # ptp = max - min
    print('ptp: ', np.ptp(data))
    print('mean: ', np.mean(data))
    print('std: ', np.std(data))
    print('var: ', np.var(data))
# 查看price
show_stats(Y_data)


# In[68]:


import warnings
warnings.filterwarnings('ignore')
#X_data.info()
X_data['notRepairedDamage'] = X_data['notRepairedDamage'].astype('float64')
X_test['notRepairedDamage'] = X_test['notRepairedDamage'].astype('float64')


# In[75]:


import xgboost as xgb
# 创建模型
model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=7, random_state=2021)
model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=2000, objective='reg:linear', tree_method='gpu_hist', subsample=0.8, colsample_bytree=0.8, min_child_samples=3, eval_metric='auc', reg_lambda=0.5)
model.fit(X_data, Y_data)


# In[76]:


y_pred = model.predict(X_test)
y_pred


# In[77]:


# 训练 欠拟合 => n_estimators太小，或者 learning_rate太小
show_stats(y_pred)


# In[79]:


# 因为XGBoost是集成学习，多棵树组成
# 有些树的叶子节点 有可能为负
result = pd.DataFrame()
result['SaleID'] = test['SaleID']
result['price'] = y_pred
result[result['price'] < 0] = 11
result


# In[ ]:


result.to_csv('./baseline_xgb.csv', index=False)

