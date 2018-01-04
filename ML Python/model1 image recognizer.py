
# coding: utf-8

# In[ ]:

x


# In[1]:

from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img


# In[2]:

dataGen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')


# In[3]:

img = load_img('/Users/ifeanyichukwuagu/coursera ML/kerasData/train/cat.8.jpg')


# In[4]:

x=img_to_array(img)
x=x.reshape((1,)+x.shape)


# In[5]:

i=0
for batch in dataGen.flow(x,batch_size=1, save_to_dir='/Users/ifeanyichukwuagu/coursera ML/kerasData/train/preview', save_prefix='cat', save_format='jpeg'):
        i +=1
        if i>20:
            break


# In[9]:

from keras import backend as K
K.set_image_dim_ordering('th')


# In[6]:

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout


# In[10]:

model1 = Sequential()
model1.add(Conv2D(32,(3,3), input_shape=(3,150,150)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))


# In[13]:

model1.add(Conv2D(32,(3,3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))


# In[14]:

model1.add(Conv2D(64,(3,3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))


# In[15]:

model1.add(Flatten())
model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1))
model1.add(Activation('sigmoid'))


# In[24]:

model1.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[18]:

batch_size=16


# In[19]:

train_dataGen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
)


# In[20]:

test_dataGen = ImageDataGenerator(rescale=1./255)


# In[29]:

train_generator = train_dataGen.flow_from_directory('/Users/ifeanyichukwuagu/coursera ML/kerasData/train/',
                                                    target_size=(150,150),
                                                    batch_size=batch_size,
                                                    class_mode='binary'
                                                   )


# In[28]:

validation_generator = test_dataGen.flow_from_directory('/Users/ifeanyichukwuagu/coursera ML/kerasData/validation/',
                                                    target_size=(150,150),
                                                        batch_size=batch_size,
                                                        class_mode='binary'
                                                       )


# In[30]:

model1.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model1.save_weights('first_try.h5')


# In[88]:

img_1 = load_img('/Users/ifeanyichukwuagu/coursera ML/kerasData/train/dog.6049.jpg', target_size=(150, 150))
print img_1.size
plt.imshow(img_1)
xxx=img_to_array(img_1)
xxx=xxx.reshape((1,)+xxx.shape)
probxxx=model1.predict(xxx)


# In[89]:

print probxxx[0][0]


# In[46]:

import matplotlib.pyplot as plt


# In[43]:

get_ipython().magic(u'matplotlib inline')


# In[47]:

plt.imshow(img_1)


# In[ ]:



