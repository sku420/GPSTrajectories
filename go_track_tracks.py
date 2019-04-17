#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv("go_track_tracks.csv")
del data['linha']
del data['id']
data.head()


# In[3]:


x=data.iloc[:,0:6]
y=data.iloc[:,7]


# In[10]:


clf=svm.SVC(kernel='linear',C=1.0)
rbf_clf=svm.SVC(kernel='rbf',C=1.0,gamma='auto')
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=0);


# In[11]:


clf.fit(x_train,y_train)
rbf_clf.fit(x_train,y_train)


# In[12]:


y_pred=clf.predict(x_test)
y_pred_rbf=clf.predict(x_test)


# In[13]:


score=clf.score(x_test,y_test)
score_rbf=rbf_clf.score(x_test,y_test)


# In[14]:


print("LINEAR: ",score)
print("RBF: ",score_rbf)


# In[15]:


scr=0.0
scr2=0.0
c1,c2=0,0
c=np.arange( 1, 10+1,0.1).tolist()
print(c)


# In[16]:


for i in c:
    clf=svm.SVC(kernel='linear',C=i)
    rbf_clf=svm.SVC(kernel='rbf',C=i,gamma='auto')

    clf.fit(x_train,y_train)
    rbf_clf.fit(x_train,y_train)

    y_pred=clf.predict(x_test)
    y_pred_rbf=clf.predict(x_test)

    score=clf.score(x_test,y_test)
    score_rbf=rbf_clf.score(x_test,y_test)
    
    if scr<score:
        scr=score
        c1=i
    if scr2<score_rbf:
        scr2=score_rbf
        c2=i


# In[18]:


print("BEST C for linear: ",c1,"SCORE: ", scr)
print("BEST C for rbf: ", c2,"SCORE: ", scr2)


# In[19]:


y_pred,y_test


# In[20]:


y_pred_rbf,y_test

