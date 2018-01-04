
# coding: utf-8

# In[2]:

print house1


# In[1]:

import graphlab


# In[8]:

sales = graphlab.SFrame('coursera ML/home_data.gl/')


# In[11]:

simple_model=graphlab.linear_regression.create(sales, target="price")


# In[13]:

train_data, test_data = sales.random_split(0.8,seed="0")


# In[14]:

sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'])


# In[15]:

import matplotlib.pyplot as plt


# In[25]:

get_ipython().magic(u'matplotlib inline')


# In[33]:

plt.plot(test_data['sqft_living'], test_data['price'], '.',
        test_data['sqft_living'], sqft_model.predict(test_data),'x')


# In[19]:

print train_data.num_rows()


# In[34]:

my_features = ['bedrooms','bathrooms','floors','sqft_living','zipcode','sqft_lot']


# In[75]:

my_features_model = graphlab.linear_regression.create(train_data, target='price', features=my_features, validation_set=None)


# In[76]:

sqft_model.evaluate(test_data)


# In[77]:

my_features_model.evaluate(test_data)


# In[48]:

graphlab.canvas.set_target("ipynb");sales.show(view='BoxWhisker Plot', x='zipcode',y='price')


# In[49]:

houses_in_zipcode_with_highest_avg_price=sales[sales['zipcode']=="98039"]


# In[55]:

print houses_in_zipcode_with_highest_avg_price['price'].mean()


# In[64]:

houses_price_range_2000to4000=sales[(sales['sqft_living']>2000) & (sales['sqft_living']<=4000)]


# In[88]:

print (float(houses_price_range_2000to4000.num_rows()) / float(sales.num_rows()))


# In[67]:

advanced_features=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
                  ]


# In[68]:

print advanced_features


# In[74]:

advanced_features_model=graphlab.linear_regression.create(train_data, target='price', features=advanced_features, validation_set=None)


# In[78]:

my_features_model.evaluate(test_data)


# In[79]:

advanced_features_model.evaluate(test_data)


# In[80]:

print my_features_model.evaluate(test_data)['rmse']- advanced_features_model.evaluate(test_data)['rmse']


# In[ ]:



