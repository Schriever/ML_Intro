
# coding: utf-8

# # Deepl learning

# In[2]:

import graphlab


# In[ ]:




# In[ ]:




# In[2]:

image_test=graphlab.SFrame('coursera ML/image_test_data/')


# In[ ]:




# In[6]:

graphlab.canvas.set_target("ipynb")


# In[13]:

image_train=graphlab.SFrame('coursera ML/image_train_data/')


# In[12]:

raw_pixel_model=graphlab.logistic_classifier.create(image_train, target='label', features['image_array'])


# In[14]:

image_test['image'].show()


# In[20]:

raw_pixel_model=graphlab.logistic_classifier.create(image_train, target='label', features=['image_array'])


# In[19]:

image_test.num_rows()


# # MAke a prediction with the simple model based on raw pixels

# In[22]:

image_test[0:3]['image'].show()


# In[24]:

image_test[0:3]['label']


# In[25]:

raw_pixel_model.predict(image_test[0:3])


# # evaluating raw pixel model on test data

# In[29]:

raw_pixel_model.evaluate(image_train)


# # Can we improve the model using deep features

# In[30]:

deep_learning_model = graphlab.load_model('imagenet_model')


# In[31]:

from sklearn.neural_network import MLPClassifier


# In[1]:

image_test.head[1]


# In[ ]:



