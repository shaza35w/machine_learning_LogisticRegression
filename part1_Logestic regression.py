#!/usr/bin/env python
# coding: utf-8

# # Logestic regression

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import math


# In[2]:


def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s


# In[3]:


def loss_fun(y, y_bar):
    m = y.shape[0]
    loss = (1 / m) * np.sum( -y.T * np.log(y_bar) - (1 - y).T * np.log( 1 - y_bar))
    return loss


# In[4]:


def compute_grad(X, y, y_bar):

    m=len(y)
    g=(X.T.dot(y_bar-y))/m
    return g


# In[5]:


def f(X, theta):
    z=X.dot(theta)
    return z


# In[6]:


def predict(X, theta):
    
    y1= sigmoid(f(X,theta))
    l=list()
    l = np.where(y1 >= 0.5, 1, 0 ) 
   
    print(y1)
    return l


# In[7]:


def score(y, y_bar):
    
    s=0
    for i in range(len(y)):
        if y[i]==y_bar[i]:
            s+=1
    s=s/len(y)    
       
         
    return s  


# ## Let’s also define the train function which will be used to find the model parameters that minimizes the cost function using functions above.

# In[8]:


def train(X,y, lr=0.02, iter=100):
    loss=[]
    theta=np.zeros(X.shape[1])
    for i in range(iter):
        z=f(X,theta)
        
        a=sigmoid(z)
        loss_=loss_fun(y,a)
        g=compute_grad(X,y, a)
        theta=theta-lr *g
        loss.append(loss_)
    print(theta)
    ##print(loss)
    
    rr=list(range(iter))
    plt.plot(rr, loss)
    plt.show()
    return(theta)
        


# #### <font color='purple'>  - Load cancer dataset from sklearn. Split the data into 70% for traing and 30% for testing. Train the training set using train function. Then, cacluate the accuracy on testing set</font>

# In[9]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
x_=StandardScaler().fit_transform(dataset.data)
y_=dataset.target


X_train,X_test,y_train,y_test = train_test_split(x_,y_, test_size = .30)
X_test.shape


# In[10]:


theta=train(X_train,y_train)


# In[11]:


y_pred=predict(X_test,theta)


# In[14]:


acc=score(y_test, y_pred)
print("Accuracy score with logistic regrission by function:" ,acc)


# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(max_iter=3000)
lgr.fit(X_train,y_train)
y_pred2 = lgr.predict(X_test)

print("Accuracy score with logistic regrission from sklearn:" ,(accuracy_score(y_test, y_pred2)))


# In[ ]:




