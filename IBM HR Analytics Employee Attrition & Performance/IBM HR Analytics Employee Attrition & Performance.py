#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Dataset

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv(r"F:\Github\IBM HR Analytics Employee Attrition & Performance\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head(5)


# # Data Pre-Processing

# In[11]:


print ("Rows    : " ,df.shape[0])
print ("Columns : " ,df.shape[1])


# In[4]:


df.info()


# In[8]:


df.isnull().sum()


# In[12]:


print(df.EmployeeCount.unique())
print(df.EmployeeNumber.unique())
print(df.Over18.unique())
print(df.StandardHours.unique())


# In[13]:


df = df.drop(['EmployeeNumber','EmployeeCount','Over18','StandardHours'],axis=1)
df.columns


# In[15]:


df_num = df.select_dtypes(include=[np.number])
df_num.head(3)


# In[16]:


df_dummies = df.select_dtypes(include=['object'])
df_dummies.head(3)


# In[17]:


from sklearn.preprocessing import LabelEncoder
df_dummies=df_dummies.apply(LabelEncoder().fit_transform)
df_dummies.head()


# In[18]:


df_combined=pd.concat([df_num,df_dummies],axis=1)
df_combined.head()


# # Data Partition

# In[20]:


from sklearn.model_selection import train_test_split

x = df_combined.drop('Attrition',axis=1)
y = df_combined[['Attrition']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)


# In[21]:


print('The number of samples into the Train data is {}.'.format(x_train.shape[0]))
print('The number of samples into the Test data is {}.'.format(x_test.shape[0]))


# # Ada Boosting

# ### Grid Search Model

# In[22]:


model_parameters = {'n_estimators' : [30,50,100,150],
                   'learning_rate': [0.1,0.5,0.4,1]}


# In[26]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV


# In[28]:


abc = AdaBoostClassifier()
gscv_ADA = GridSearchCV(estimator=abc,param_grid=model_parameters,cv=10,verbose=1,n_jobs=-1,scoring='accuracy')
gscv_ADA.fit(x_train,y_train)


# In[31]:


print('The best parameters are - ', gscv_ADA.best_params_)


# In[32]:


final_mod_ADA = AdaBoostClassifier(**gscv_ADA.best_params_)
final_mod_ADA.fit(x_train,y_train)


# In[33]:


imp = pd.Series(data=final_mod_ADA.feature_importances_,index=final_mod_ADA.feature_names_in_).sort_values(ascending = False)
plt.figure(figsize=(10,12))
plt.title("Feature Importance")
ax= sns.barplot(y=imp.head().index,x=imp.head().values,palette="Blues_r",orient='h')


# # Prediction

# In[34]:


train_pred = final_mod_ADA.predict(x_train)


# # Confusion Matrix on Train Data

# In[36]:


from sklearn.metrics import confusion_matrix, classification_report


# In[37]:


print('Classification report for train data is : \n',
     classification_report(y_train,train_pred))


# #  Confusion Matrix on Train Data

# In[38]:


test_pred = final_mod_ADA.predict(x_test)


# In[40]:


print('Classification report for test data is : \n',
     classification_report(y_test,test_pred))


# # Decision Tree

# ### Using Pruning Method

# In[51]:


from sklearn.tree import DecisionTreeClassifier

dt = tree.DecisionTreeClassifier(criterion='gini',
                                min_samples_leaf=50,
                                min_samples_split=150,
                                max_depth=3)
dt.fit(x_train,y_train)


# In[52]:


train=pd.concat([y_train,x_train],axis=1)
train.head()


# In[53]:


independent_variable = list(train.columns[1:])
independent_variable


# In[54]:


from sklearn import tree

Attrition = ['No','Yes']
fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(5,4),dpi=300)
tree.plot_tree(dt,
              feature_names= independent_variable,
              class_names=Attrition,
              filled = True,
              node_ids=2);


# In[55]:


train.head()


# In[56]:


train['Predicted'] = dt.predict(x_train)
train.head()


# # Model Performance Metrics

# ### On Train Data

# In[57]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(train['Predicted'],train['Attrition'])
print(matrix)


# In[58]:


x_train.shape


# In[59]:


Accuracy_train=((942+58)/(1176)*100)
print(Accuracy_train)


# In[61]:


print('Classification report for test data is : \n',
     classification_report(train['Attrition'],train['Predicted']))


# In[64]:


test=pd.concat([y_test,x_test],axis=1)
test.head()


# In[65]:


test['Predicted'] = dt.predict(x_test)
test.head()


# ### On Test Data

# In[66]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test['Predicted'],test['Attrition'])
print(matrix)


# In[68]:


x_test.shape


# In[69]:


Accuracy_train=((226+38)/(294)*100)
print(Accuracy_train)


# In[70]:


print('Classification report for test data is : \n',
     classification_report(test['Attrition'],test['Predicted']))


# # Using Grid Search Method

# In[72]:


params = {'min_samples_leaf': [60,70,80,120,150],
          'min_samples_split': [150,300,250,450], 
          'max_depth': [3,4,6]}


# In[73]:


gscv_DT = GridSearchCV(DecisionTreeClassifier(random_state=45),
                       params,
                       cv=10,
                       verbose=1)

gscv_DT.fit(x_train,y_train)


# In[74]:


gscv_DT.best_estimator_


# In[75]:


print('Classification report for test data is : \n',
     classification_report(train['Attrition'],train['Predicted']))


# # For Business

# In[77]:


import os
os.chdir(r"F:\Github\IBM HR Analytics Employee Attrition & Performance")


# In[79]:


import pickle
pickle.dump(dt, open(r"F:\Github\IBM HR Analytics Employee Attrition & Performance\build.pkl",'wb'))


# # Live Data

# In[81]:


LD= x_test.iloc[0:4,:]
LD


# In[82]:


FM= pickle.load(open('build.pkl','rb'))


# In[83]:


LD['Predicted']=FM.predict(LD)
LD


# In[ ]:




