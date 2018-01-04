
# coding: utf-8

# In[1]:

import keras


# In[4]:

((raw_training_images, training_classes),
 (test_images, test_classes))=keras.datasets.mnist.load_data()


# In[8]:

raw_training_images[1].view


# In[9]:

get_ipython().magic(u'matplotlib inline')


# In[14]:

from keras.utils import np_utils


# In[13]:

import numpy as np


# In[10]:

import matplotlib.pyplot as plt


# In[17]:

#plt.imshow(test_images[1])
print 1


# In[15]:

def prepare_images_and_classes( images, image_classes):
    images=images.reshape(images.shape[0], 28*28).astype('float32')
    return (images/255,
           np_utils.to_categorical(image_classes))


# In[18]:

training_images, training_classes = prepare_images_and_classes(raw_training_images, training_classes)


# In[19]:

test_images, test_classes = prepare_images_and_classes(test_images, test_classes)


# In[24]:

training_images.shape


# In[26]:

raw_training_images.shape


# In[27]:

print 28*28


# In[28]:

temp_image=training_images*255


# In[32]:

#print temp_image[0]


# In[33]:

#print training_images[0]


# In[35]:

model = keras.models.Sequential()


# In[39]:

model.add(keras.layers.Dense(
         28*28,
        input_dim =28*28,
         init='normal',
         activation='relu'))


# In[42]:

model.add(keras.layers.Dense(
    10,
        init='normal',
        activation='softmax'
    ))


# In[54]:

model.compile(loss='categorical_crossentropy',
             metrics=['accuracy'],
             optimizer='adam')


# In[52]:

#model.summary()


# In[55]:

history = model.fit( training_images, training_classes,
                   validation_data=(test_images, test_classes),
                   nb_epoch=10,
                   batch_size=200,
                   verbose=2).history


# In[56]:

image_number_to_predict=0


# In[90]:

#plt.imshow(raw_training_images[image_number_to_predict])


# In[61]:

class_probabilities = model.predict(training_images[image_number_to_predict:image_number_to_predict+1])[0]


# In[63]:

#class_probabilities


# In[64]:

plt.bar(np.arange(0,10),np.sqrt(class_probabilities))


# In[65]:

print np.sqrt(class_probabilities)


# In[72]:

model.predict_classes


# In[66]:

print (class_probabilities)


# In[75]:

predicted_number = np_utils.to_categorical(class_probabilities)


# In[76]:

#print model.predict_classes(class_probabilities)


# In[77]:

def predic_class_from_probabilities(probabilities, classes):
    return (probabilities[probabilities==probabilities.max]*classes)


# In[79]:

test_classes.shape


# In[80]:

((raw_training_images2, training_classes2),
 (test_images2, test_classes2))=keras.datasets.mnist.load_data()


# In[91]:

test_classes2.shape


# In[ ]:



