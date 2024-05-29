#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"F:\Github\IBM HR Analytics Employee Attrition & Performance\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head(5)


# # Data Pre-Processing

# In[3]:


print ("Rows    : " ,df.shape[0])
print ("Columns : " ,df.shape[1])


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


print(df.EmployeeCount.unique())
print(df.EmployeeNumber.unique())
print(df.Over18.unique())
print(df.StandardHours.unique())


# In[7]:


df = df.drop(['EmployeeNumber','EmployeeCount','Over18','StandardHours'],axis=1)
df.columns


# In[8]:


df1 = df


# In[9]:


df_num = df.select_dtypes(include=[np.number])
df_num.head(3)


# In[10]:


df_dummies = df.select_dtypes(include=['object'])
df_dummies.head(3)


# In[11]:


from sklearn.preprocessing import LabelEncoder
df_dummies=df_dummies.apply(LabelEncoder().fit_transform)
df_dummies.head()


# In[12]:


df_combined=pd.concat([df_num,df_dummies],axis=1)
df_combined.head()


# # Data Visualization

# In[13]:


import plotly.express as px

fig = px.pie(df,names='Attrition',color='Attrition',color_discrete_map={'Yes':'red','No':'green'})
fig.show()


# # Data Partition

# In[14]:


from sklearn.model_selection import train_test_split

x = df_combined.drop('Attrition',axis=1)
y = df_combined[['Attrition']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)


# In[15]:


print('The number of samples into the Train data is {}.'.format(x_train.shape[0]))
print('The number of samples into the Test data is {}.'.format(x_test.shape[0]))


# # Ada Boosting

# ### Grid Search Model

# In[16]:


model_parameters = {'n_estimators' : [30,50,100,150],
                   'learning_rate': [0.1,0.5,0.4,1]}


# In[17]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV


# In[18]:


abc = AdaBoostClassifier()
gscv_ADA = GridSearchCV(estimator=abc,param_grid=model_parameters,cv=10,verbose=1,n_jobs=-1,scoring='accuracy')
gscv_ADA.fit(x_train,y_train)


# In[19]:


print('The best parameters are - ', gscv_ADA.best_params_)


# In[20]:


final_mod_ADA = AdaBoostClassifier(**gscv_ADA.best_params_)
final_mod_ADA.fit(x_train,y_train)


# In[21]:


imp = pd.Series(data=final_mod_ADA.feature_importances_,index=final_mod_ADA.feature_names_in_).sort_values(ascending = False)
plt.figure(figsize=(10,12))
plt.title("Feature Importance")
ax= sns.barplot(y=imp.head().index,x=imp.head().values,palette="Blues_r",orient='h')


# # Prediction

# In[22]:


train_pred = final_mod_ADA.predict(x_train)


# # Confusion Matrix on Train Data

# In[23]:


from sklearn.metrics import confusion_matrix, classification_report


# In[24]:


print('Classification report for train data is : \n',
     classification_report(y_train,train_pred))


# #  Confusion Matrix on Train Data

# In[25]:


test_pred = final_mod_ADA.predict(x_test)


# In[26]:


print('Classification report for test data is : \n',
     classification_report(y_test,test_pred))


# # Decision Tree

# ### Using Pruning Method

# In[27]:


from sklearn import tree 


# In[28]:


from sklearn.tree import DecisionTreeClassifier

dt = tree.DecisionTreeClassifier(criterion='gini',
                                min_samples_leaf=50,
                                min_samples_split=150,
                                max_depth=3,
                                class_weight='balanced')
dt.fit(x_train,y_train)


# In[29]:


train=pd.concat([y_train,x_train],axis=1)
train.head()


# In[30]:


independent_variable = list(train.columns[1:])
independent_variable


# In[31]:


from sklearn import tree

Attrition = ['No','Yes']
fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(5,4),dpi=300)
tree.plot_tree(dt,
              feature_names= independent_variable,
              class_names=Attrition,
              filled = True,
              node_ids=2);


# In[32]:


train.head()


# In[33]:


train['Predicted'] = dt.predict(x_train)
train.head()


# # Model Performance Metrics

# ### On Train Data

# In[34]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(train['Predicted'],train['Attrition'])
print(matrix)


# In[35]:


x_train.shape


# In[36]:


Accuracy_train=((704+134)/(1176)*100)
print(Accuracy_train)


# In[37]:


print('Classification report for train data is : \n',
     classification_report(train['Attrition'],train['Predicted']))


# In[38]:


test=pd.concat([y_test,x_test],axis=1)
test.head()


# In[39]:


test['Predicted'] = dt.predict(x_test)
test.head()


# ### On Test Data

# In[40]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test['Predicted'],test['Attrition'])
print(matrix)


# In[41]:


x_test.shape


# In[42]:


Accuracy_train=((156+39)/(294)*100)
print(Accuracy_train)


# In[43]:


print('Classification report for test data is : \n',
     classification_report(test['Attrition'],test['Predicted']))


# # Using Grid Search Method

# In[44]:


params = {'min_samples_leaf': [60,70,80,120,150],
          'min_samples_split': [150,300,250,450], 
          'max_depth': [3,4,6]}


# In[45]:


gscv_DT = GridSearchCV(DecisionTreeClassifier(random_state=45),
                       params,
                       cv=10,
                       verbose=1)

gscv_DT.fit(x_train,y_train)


# In[46]:


gscv_DT.best_estimator_


# In[106]:


print('Classification report for train data is : \n',
     classification_report(train['Attrition'],train['Predicted']))


# In[ ]:





# # Random Forest

# In[48]:


df1_num = df1.select_dtypes(include=[np.number])
df1_num.head(3)


# In[49]:


df1_dummies = df1.select_dtypes(include=['object'])
df1_dummies.head(3)


# In[50]:


from sklearn.preprocessing import LabelEncoder
df1_dummies=df1_dummies.apply(LabelEncoder().fit_transform)
df1_dummies.head()


# In[51]:


df1_combined=pd.concat([df1_num,df1_dummies],axis=1)
df1_combined.head()


# In[ ]:





# # Data Partition 

# In[52]:


from sklearn.model_selection import train_test_split

x = df1_combined.drop('Attrition',axis=1)
y = df1_combined[['Attrition']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)


# In[53]:


x_train.shape


# In[54]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=26,
                                  criterion='gini',
                                min_samples_leaf=50,
                                min_samples_split=150,
                                max_depth=5,
                                 class_weight='balanced')


rf_model.fit(x_train, y_train)


# ### Important Features

# In[55]:


imp = pd.Series(data=final_mod_ADA.feature_importances_,index=rf_model.feature_names_in_).sort_values(ascending = False)
plt.figure(figsize=(10,12))
plt.title("Feature Importance")
ax= sns.barplot(y=imp.head().index,x=imp.head().values,palette="BrBG",orient='h')


# In[56]:


from sklearn.tree import export_graphviz
import pydot


# In[57]:


list(x.columns)


# In[58]:


feature_list = list(x.columns)
Attrition = ['No','Yes']

tree_rf = rf_model.estimators_[10]

export_graphviz(tree_rf, out_file= 'rfabc.dot',
               feature_names=feature_list,
               class_names= Attrition,
               rounded= True,
               filled=True)

(graph, )=pydot.graph_from_dot_file('rfabc.dot')
graph.write_png('tree_rf.png')

from IPython.display import Image
Image(filename='tree_rf.png')


# # Train Data

# In[59]:


train_rf = pd.concat([x_train,y_train],axis=1)
train_rf.head()


# In[60]:


train_rf['Predicted'] = rf_model.predict(x_train)
train_rf.head()


# # Model Performance

# In[61]:


print(y_train.value_counts())


# In[62]:


x_train.shape


# In[63]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(train_rf['Predicted'],train_rf['Attrition'])
print(matrix)


# In[64]:


Accuracy_train_rf = ((803+135)/1176*100)
print(Accuracy_train_rf)


# In[65]:


print('Classification report for train data is : \n',
     classification_report(train_rf['Attrition'],train_rf['Predicted']))


# # Prediction On Test Dataset

# In[66]:


test_rf =pd.concat([y_test,x_test],axis=1)
test_rf.head()


# In[67]:


test_rf['Predicted'] = rf_model.predict(x_test)
test_rf.head()


# # Model Performance Metrics on Test Data

# In[68]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_rf['Predicted'],test_rf['Attrition'])
print(matrix)


# In[69]:


x_test.shape


# In[70]:


Accuracy_test_rf = ((192+36)/294*100)
print (Accuracy_test_rf)


# In[71]:


print('Classification report for test data is : \n',
     classification_report(test_rf['Attrition'],test_rf['Predicted']))


# # Gradient Boosting 

# In[72]:


df2 = df


# In[73]:


df2_num = df2.select_dtypes(include=[np.number])
df2_num.head(3)


# In[74]:


df2_dummies = df2.select_dtypes(include=['object'])
df2_dummies.head(3)


# In[75]:


from sklearn.preprocessing import LabelEncoder
df2_dummies=df2_dummies.apply(LabelEncoder().fit_transform)
df2_dummies.head()


# In[76]:


df2_combined=pd.concat([df2_num,df2_dummies],axis=1)
df2_combined.head()


# In[77]:


df2.info()


# In[78]:


from sklearn.model_selection import train_test_split

x = df2_combined.drop('Attrition',axis=1)
y = df2_combined[['Attrition']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)


# In[79]:


gb_model_parameters = {'n_estimators':[2,6,8],'learning_rate':[0.6,0.7],
                                'min_samples_leaf':[300,240,500],
                                'min_samples_split':[60,80,100],
                                'max_depth':[3,5]}


# In[80]:


from sklearn.ensemble import GradientBoostingClassifier 


# In[81]:


gb_model = GradientBoostingClassifier(random_state = 15, loss='log_loss',criterion = 'squared_error')

gscv_gbm = GridSearchCV(estimator=gb_model,
                       param_grid=gb_model_parameters,
                       cv= 15,
                       verbose=1,
                       n_jobs=-1,
                       scoring='accuracy')

gscv_gbm.fit(x_train, y_train)


# In[82]:


print('The best parameters are -', gscv_gbm.best_params_)


# In[83]:


final_gb_model = GradientBoostingClassifier(**gscv_gbm.best_params_)
final_gb_model.fit(x_train, y_train)


# In[84]:


train_pred_gb = final_gb_model.predict(x_train)
test_pred_gb = final_gb_model.predict(x_test)


# # Model Performance

# In[85]:


print('Classification report for train data is : \n',
     classification_report(y_train, train_pred_gb))


# In[86]:


print('Classification report for test data is : \n',
     classification_report(y_test, test_pred_gb))


# In[87]:


feature_list = list(x.columns)
Attrition = ['No','Yes']

tree_gb = final_gb_model.estimators_[0,0]

export_graphviz(tree_gb, out_file= 'gbabc.dot',
               feature_names=feature_list,
               class_names= Attrition,
               rounded= True,
               filled=True)

(graph, )=pydot.graph_from_dot_file('gbabc.dot')
graph.write_png('tree_gb.png')

from IPython.display import Image
Image(filename='tree_gb.png')


# In[88]:


print('Classification report for train data is : \n',
     classification_report(y_train, train_pred_gb))


# # Comparing the Models

# In[89]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[90]:


accuracy_ada = accuracy_score(y_test, final_mod_ADA.predict(x_test))
accuracy_dt = accuracy_score(y_test, dt.predict(x_test))
accuracy_rf = accuracy_score(y_test, rf_model.predict(x_test))
accuracy_gb = accuracy_score(y_test, final_gb_model.predict(x_test))

accuracies = {
    'AdaBoost': accuracy_ada,
    'Decision Tree': accuracy_dt,
    'Random Forest': accuracy_rf,
    'Gradient Boost': accuracy_gb}

plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 1])
plt.show()


# In[91]:


accuracy_ada = accuracy_score(y_test, final_mod_ADA.predict(x_test))
accuracy_dt = accuracy_score(y_test, dt.predict(x_test))
accuracy_rf = accuracy_score(y_test, rf_model.predict(x_test))
accuracy_gb = accuracy_score(y_test, final_gb_model.predict(x_test))

accuracies = {
    'AdaBoost': accuracy_ada,
    'Decision Tree': accuracy_dt,
    'Random Forest': accuracy_rf,
    'Gradient Boost': accuracy_gb}

best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

models = list(accuracies.keys())
accuracies = list(accuracies.values())
colors = ['blue' if model != best_model else 'orange' for model in models]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 1])

# Highlight the best model
plt.annotate(f'Best: {best_model}\nAccuracy: {best_accuracy:.2f}',
             xy=(models.index(best_model), best_accuracy),
             xytext=(models.index(best_model), best_accuracy + 0.05),
             ha='center', color='black', weight='bold')

plt.show()


# # Classification Reports

# ## Ada Boosting

# In[97]:


print('Classification report for train data is : \n',
     classification_report(y_train,train_pred))


# In[98]:


print('Classification report for test data is : \n',
     classification_report(y_test,test_pred))


# ## Decision Tree

# In[99]:


print('Classification report for train data is : \n',
     classification_report(train['Attrition'],train['Predicted']))


# In[100]:


print('Classification report for test data is : \n',
     classification_report(test['Attrition'],test['Predicted']))


# ### After using grid search

# In[101]:


print('Classification report for train data is : \n',
     classification_report(train['Attrition'],train['Predicted']))


# ## Random Forest

# In[102]:


print('Classification report for train data is : \n',
     classification_report(train_rf['Attrition'],train_rf['Predicted']))


# In[103]:


print('Classification report for test data is : \n',
     classification_report(test_rf['Attrition'],test_rf['Predicted']))


# ## Gradient Boosting

# In[104]:


print('Classification report for train data is : \n',
     classification_report(y_train, train_pred_gb))


# In[105]:


print('Classification report for test data is : \n',
     classification_report(y_test, test_pred_gb))


# # For Business

# In[92]:


import os
os.chdir(r"F:\Github\IBM HR Analytics Employee Attrition & Performance")


# In[93]:


import pickle
pickle.dump(dt, open(r"F:\Github\IBM HR Analytics Employee Attrition & Performance\build.pkl",'wb'))


# # Live Data

# In[94]:


LD= x_test.iloc[0:4,:]
LD


# In[95]:


FM= pickle.load(open('build.pkl','rb'))


# In[96]:


LD['Predicted']=FM.predict(LD)
LD

