#!/usr/bin/env python
# coding: utf-8

# Logistic regression using a dataset from keggle to predict 10 year risk of coronary heart disease in the form of yes or no.
# 
# This analysis will be completed using python which will allow for supervised machine learning with sklearn. 
# 
# Attributes
# 
# Demographic
# 
# Sex: male or female
# Age: Age of the patient
# 
# Education: no information provided
# 
# Behavioral
# 
# Current Smoker: whether or not the patient is a current smoker
# Cigs Per Day: the number of cigarettes that the person smoked on average in one day
# 
# Information on medical history
# 
# BP Meds: whether or not the patient was on blood pressure medication
# Prevalent Stroke: whether or not the patient had previously had a stroke
# Prevalent Hyp: whether or not the patient was hypertensive
# Diabetes: whether or not the patient had diabetes
# 
# Information on current medical condition
# 
# Tot Chol: total cholesterol level
# Sys BP: systolic blood pressure
# Dia BP: diastolic blood pressure
# BMI: Body Mass Index
# Heart Rate: heart rate - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.
# Glucose: glucose level
# 
# Target variable to predict
# 
# TenYearCHD: 10 year risk of coronary heart disease (binary: “1:Yes”, “0:No”)

# In[1]:


#import libraries and packages for log
import pandas as pd
from pandas import Series
from pandas import DataFrame

import numpy as np
from numpy import ndarray
from numpy.random import randn
from numpy import loadtxt, where

import matplotlib 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rc("font", size=14)
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import random

import scipy

from tkinter import *

from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingCVClassifier

import os

import pylab
from pylab import scatter, show, legend, xlabel, ylabel

from collections import Counter
import pandas_profiling as pp

#ignore future warnings
import warnings 
warnings.filterwarnings('ignore') 


# In[2]:


#read csv into Pandas dataframe
df = pd.read_csv('logistic_regression_heart.csv') 


# In[3]:


#view data layout
df.head()


# Data Processing

# In[4]:


#obtain index and datatypes
df.info()


# In[5]:


#number of columns and rows
df.shape


# In[6]:


#check data for null values
df.isnull().sum()


# In[7]:


#impute null values with mean
df.fillna(df.mean(), inplace=True)


# In[8]:


#confirm imputation worked- expect to see 0 null values
df.isnull().sum()


# In[9]:


#check data for duplicates
df.duplicated().sum()


# Data Visualization

# In[10]:


#obtain summary statistics
df.describe()


# In[11]:


# Bar chart of class ratio 
target_pd = pd.DataFrame(index = ["No Disease","Disease"], columns= ["Quantity", "Percentage"])
# No disease
target_pd.loc["No Disease"]["Quantity"] = len(df[df.columns[-1]][df[df.columns[-1]]==0].dropna())
target_pd.loc["No Disease"]["Percentage"] = target_pd.iloc[0,0]/len(df[df.columns[-1]])*100
# Disease
target_pd.loc["Disease"]["Quantity"] = len(df[df.columns[-1]][df[df.columns[-1]]==1].dropna())
target_pd.loc["Disease"]["Percentage"] = target_pd.iloc[1,0]/len(df[df.columns[-1]])*100
# Plot barchart
fig = plt.figure(figsize = (10, 5))
plt.bar(list(target_pd.index), target_pd.iloc[:,0], color ='maroon',width = 0.4)
plt.ylabel("Number of cases")
plt.title("Distribution of disease and non-disease cases");
# Print the dataframe
target_pd


# In[12]:


# Histogram of features (check for skew)
fig=plt.figure(figsize=(20,20))
for i, feature in enumerate(df.columns):
    ax=fig.add_subplot(8,4,i+1)
    df[feature].hist(bins=20,ax=ax,facecolor='black')
    ax.set_title(feature+" Distribution",color='DarkRed')
    ax.set_yscale('log')
fig.tight_layout()  


# Correlation

# In[13]:


pp.ProfileReport(df)


# In[14]:


#Scaling and splitting the data into the train and test data
y = df["TenYearCHD"]
X = df.drop('TenYearCHD',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[15]:


#logistic regression 
m1 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("confussion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(y_test,lr_predict))


# Additional Models

# In[16]:


m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=20, random_state=2,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))


# In[17]:


#extreme gradient boost
m4 = 'Extreme Gradient Boost'
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
xgb.fit(X_train, y_train)
xgb_predicted = xgb.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("confussion matrix")
print(xgb_conf_matrix)
print("\n")
print("Accuracy of Extreme Gradient Boost:",xgb_acc_score*100,'\n')
print(classification_report(y_test,xgb_predicted))


# In[18]:


#k-neighbors classifier
m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report(y_test,knn_predicted))


# In[19]:


#decision tree classifier
m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))


# In[20]:


#support vector classifier
m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))


# In[ ]:




