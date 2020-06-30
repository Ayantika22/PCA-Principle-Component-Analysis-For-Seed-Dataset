#!/usr/bin/env python
# coding: utf-8

# # PCA cluster for SEED dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model 
#from sklearn import linear_model.fit
from sklearn.linear_model import LinearRegression 
from sklearn.decomposition import PCA 
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer


# In[2]:


# loading dataset into Pandas DataFrame
df = pd.read_csv("Seed_data.csv")
df


# In[3]:


#Importing libraries from SKLEARN
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA


# In[4]:


seed = pd.read_csv('Seed_data.csv')

X = seed.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
Y = seed['target']


pca = PCA(n_components=7)
X_r = pca.fit(X).transform(X)


# # PCA plot for Seed dataset

# In[66]:


plt.scatter(X_r[Y == 0, 2], X_r[Y == 0, 3], s =50, c = 'orange', label = 'Target 0')
plt.scatter(X_r[Y == 1, 0], X_r[Y == 1, 5], s =50,  c = 'yellow', label = 'Target 1')
plt.scatter(X_r[Y == 2, 0], X_r[Y == 2, 4], s =50,  c = 'green', label = 'Target 2')
plt.title('PCA plot for Seed dataset')
plt.legend()


# In[3]:


'''KNN classifier which is a type of supervised Machine Learning Technique. 
This is used to detect the accuracy and classification  of the given dataset'''

# Importing Libraries for Modelling.
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[68]:


# Assigning values of X and y from dataset

y = df['target']          # Split off classifications
X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values 
''' Here X is assigned as all the column data and
y is assigned as Target value of seed dataset'''

#Setting training and testing values

Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

# Modeling is done using KNN classifiers.
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)


# Display the Output

print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Confusion matrix \n',  confusion_matrix(y_test, y_pred))
print('Classification \n', classification_report(y_test, y_pred))


# In[69]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[57]:


y = df['target']          # Split off classifications
X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values 


Xtrain, Xtest, y_train, y_test = train_test_split(X, y) 


# # Logistic Regression Accuracy 

# In[59]:


#Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Logistic Regression :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for LR

# In[60]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # K-Nearest Neighbors Accuracy

# In[37]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("K Nearest Neighbors :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for KNN

# In[38]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Support Vector Machine Accuracy

# In[68]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# In[69]:


#Support Vector Machine
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Support Vector Machine:")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for SVM

# In[70]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Gaussian Naive Bayes Accuracy

# In[42]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
classifier = GaussianNB()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Gaussian Naive Bayes :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for GNB

# In[43]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Decision Tree Classifier Accuracy

# In[46]:


#Decision Tree Classifier
from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

classifier = DT(criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for DTC

# In[47]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Random Forest Classifier Accuracy

# In[55]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF
Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
classifier = RF(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for RFC

# In[56]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[ ]:




