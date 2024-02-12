#!/usr/bin/env python
# coding: utf-8

# ## Project: BREAST CANCER DETECTION

# # Using Machine Learning To Predict Diagnosis of a Breast Cancer
# ## 1. Identify the problem
# ### Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a results of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.
# 
# ## 1.1 Expected outcome
# ### Given breast cancer results from breast fine needle aspiration (FNA) test (is a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore or swelling) with a fine needle similar to a blood sample needle). Since this build a model that can classify a breast cancer tumor using two training classification:
# 
# ### 1= Malignant (Cancerous) - Present
# ### 0= Benign (Not Cancerous) -Absent
# # 1.2 Objective
# ### Since the labels in the data are discrete, the predication falls into two categories, (i.e. Malignant or benign). In machine learning this is a classification problem.
# 
# ### Thus, the goal is to classify whether the breast cancer is benign or malignant and predict the recurrence and non-recurrence of malignant cases after a certain period. To achieve this we have used machine learning classification methods to fit a function that can predict the discrete class of new input.
# 
# # 1.3 Identify data sources
# ### The Breast Cancer datasets is available machine learning repository maintained by the University of California, Irvine. The dataset contains 569 samples of malignant and benign tumor cells.
# 
# ### The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively.
# ### The columns 3-32 contain 30 real-value features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant.

# In[28]:


# import libraries
import pandas as pd
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns


# In[4]:


# load data set
df=pd.read_csv("breast cancer detection.csv")
df.head()


# In[13]:


df.shape


# In[16]:


df.columns


# In[8]:


df.head()


# In[8]:


# check categary level of all columns
for col in df.columns:
    print("-----------{}-----------" .format(col))
    print(df[col].value_counts())
    print()


# In the above values counts we can see in two columns we have category and numbers so we need to replace the categories with numbers using label encoder

# In[12]:


# check missing values
df.isnull()


# In[13]:


# check basic info
df.info()


# we need to perform label encoder on all the columns bcoz all the columns contains object data type

# In[21]:


df.isna().sum()


# In[23]:


df = df.drop('Unnamed: 32',axis=1)
df.info()


# In[25]:


# count the malignant and benignate
df['diagnosis'].value_counts()


# In[29]:


sns.countplot(df,x="diagnosis")
plt.show()


# In[30]:


df.dtypes


# In[34]:


# encoding categorical data
le = LabelEncoder()
#df.iloc[:,0]= le,fit_transform(df.iloc[:,values])
df[df.columns[0]] = le.fit_transform(df.iloc[:, 0].values)


# In[14]:


df.head()


# ### seperate columns into smaller dataframes to perform visualization 

# In[35]:


data_mean =df.iloc[:,1:11]


# In[36]:


data_mean.boxplot()


# In[37]:


#Plot histograms of CUT1 variables
hist_mean=data_mean.hist(bins=10, figsize=(15, 10),grid=False,)


# In[38]:


#Heatmap
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, vmax=1.0, vmin=-1.0)


# In[39]:


#Density Plots
plt = data_mean.plot(kind= 'density', subplots=True, layout=(4,3), sharex=False, 
                     sharey=False,fontsize=12, figsize=(15,10))


# ###  Splitting the data

# In[40]:


# train test split
from sklearn.model_selection import train_test_split


# In[41]:


X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis'].values


# In[42]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[43]:


df.shape


# In[44]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Data Normalization
# 

# In[45]:


scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)


# ### Model building
# 

# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[47]:


# convert y_train and y_test data into int from object 
y_train = y_train.astype(int)
y_test = y_test.astype(int)


# ### 1. Logistic Regression
# 

# In[48]:


# Logistic Regression

# train the model
reg = LogisticRegression()
reg.fit(X_train_scale,y_train)

# prediction on test data 
y_pred = reg.predict(X_test_scale)

# evalation 
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------------------")
print("Accuracy : ")
print(accuracy_score(y_test, y_pred))
print("-----------------------------------------------")
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# ### 2. Support Vector Machine
# 

# In[49]:


from sklearn.svm import SVC
# Support Vector Machine

# train the model
svc = SVC()
svc.fit(X_train_scale,y_train)

# prediction on test data 
y_pred = svc.predict(X_test_scale)

# evalation 
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------------------")
print("Accuracy : ")
print(accuracy_score(y_test, y_pred))
print("-----------------------------------------------")
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# ### 3. Naive Bayes
# 

# In[50]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_scale, y_train)

# prediction on test data
y_pred = nb.predict(X_test_scale)

# evalation 
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------------------")
print("Accuracy : ")
print(accuracy_score(y_test, y_pred))
print("-----------------------------------------------")
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# ### 4.Decision Tree
# 

# In[51]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(X_train_scale, y_train)

# prediction on test data
y_pred = DT.predict(X_test_scale)

# evalation 
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------------------")
print("Accuracy : ")
print(accuracy_score(y_test, y_pred))
print("-----------------------------------------------")
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# ### 5. Random forest
# 

# In[52]:


from sklearn.ensemble import RandomForestClassifier
RF = DecisionTreeClassifier()
RF.fit(X_train_scale, y_train)

# prediction on test data
y_pred = RF.predict(X_test_scale)

# evalation 
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------------------")
print("Accuracy : ")
print(accuracy_score(y_test, y_pred))
print("-----------------------------------------------")
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# In[54]:


# Create DataFrames for each algorithm
logi_df = pd.DataFrame({
    'Algorithm': ['Logistic Regression'],
    'Confusion Matrix': [confusion_matrix(y_test, y_pred)],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Classification Report': [classification_report(y_test, y_pred)]
})

svc_df = pd.DataFrame({
    'Algorithm': ['Support Vector Machine'],
    'Confusion Matrix': [confusion_matrix(y_test, y_pred)],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Classification Report': [classification_report(y_test, y_pred)]
})
nb_df = pd.DataFrame({
    'Algorithm': ['Naive Bayes'],
    'Confusion Matrix': [confusion_matrix(y_test, y_pred)],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Classification Report': [classification_report(y_test, y_pred)]
})

DT_df = pd.DataFrame({
    'Algorithm': ['Dicision Tree'],
    'Confusion Matrix': [confusion_matrix(y_test, y_pred)],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Classification Report': [classification_report(y_test, y_pred)]
})
RF_df = pd.DataFrame({
    'Algorithm': ['Random Forest'],
    'Confusion Matrix': [confusion_matrix(y_test, y_pred)],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Classification Report': [classification_report(y_test, y_pred)]
})

xgb_df = pd.DataFrame({
    'Algorithm': ['XGBOOST'],
    'Confusion Matrix': [confusion_matrix(y_test, y_pred)],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Classification Report': [classification_report(y_test, y_pred)]
})

# Concatenate the DataFrames
result_df = pd.concat([logi_df, svc_df, nb_df, DT_df, RF_df, xgb_df], ignore_index=True)

# Display the result DataFrame
from tabulate import tabulate
print(tabulate(result_df, headers='keys', tablefmt='pretty', showindex=False))


# ### Conclusion
# -based on above analysis we can conclude that logistic regression, Support Vector Machine,Naive Bayes,Decisoin tree,Random forest,XGBOOST models perform shows same accuracy on test data  

# In[ ]:




