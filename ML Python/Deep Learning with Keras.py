
# coding: utf-8

# In[1]:

import keras


# In[2]:

import matplotlib.pyplot as plt


# In[4]:

import seaborn as sns


# In[5]:

import numpy as np


# In[7]:

import pandas as pd


# In[8]:

from sklearn.cross_validation import train_test_split


# In[9]:

from sklearn.linear_model import LogisticRegressionCV


# In[11]:

from keras.models import Sequential


# In[12]:

from keras.layers.core import Dense, Activation


# In[13]:

from keras.utils import np_utils


# In[14]:

iris = pd.read_csv('gphlab/iris.data')


# In[21]:

iris.head()
iris.columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal Width', 'Species']


# In[30]:

sns.pairplot(iris, hue='Species')


# In[33]:

y=iris.values[:,4];x=iris.values[:,:4]


# In[29]:

get_ipython().magic(u'matplotlib inline')


# In[27]:

iris.head()


# In[55]:

train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.5, random_state=0)


# In[56]:

lr = LogisticRegressionCV()


# In[57]:

lr.fit(train_x,train_y)


# In[65]:

test_x.shape


# In[68]:

print("Accuracy = {:.2f}".format(lr.score(test_x,test_y)))


# In[109]:

lr.score(test_x, test_y)


# In[ ]:




# In[72]:

def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


# In[73]:

train_y_ohe = one_hot_encode_object_array(train_y)


# In[94]:

test_y_ohe = one_hot_encode_object_array(test_y)


# In[95]:

model = Sequential()


# In[96]:

model.add(Dense(16,input_shape=(4,)))


# In[97]:

model.add(Activation('sigmoid'))


# In[98]:

model.add(Dense(3))


# In[99]:

model.add(Activation('softmax'))


# In[100]:

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


# In[101]:

model.fit(train_x, train_y_ohe, nb_epoch=100, batch_size=1, verbose=0)


# In[107]:

loss, accuracy = model.evaluate(test_x, test_y_ohe)


# In[108]:

print ("Accuracy{:.2f}".format(accuracy))


# In[ ]:



