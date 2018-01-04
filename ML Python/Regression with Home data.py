
# coding: utf-8

# In[3]:

import graphlab


# In[4]:

sales=graphlab.SFrame('home_data.gl/')


# In[5]:

sales


# # Exploring the data for house sales

# In[7]:

graphlab.canvas.set_target("ipynb");sales.show(view="Scatter Plot",x="sqft_living", y="price")


# # Create a simple regression model
train_data, test_data = sales.ran
# In[8]:

train_data, test_data = sales.random_split(.8,seed="0")


# # Build regression model

# In[9]:

sqft_model=graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'])


# # Evaluate the simple model

# In[10]:

print test_data['price'].mean()


# In[11]:

print sqft_model.evaluate(test_data)


# # Lets show what our predictions look like

# In[12]:

import matplotlib.pyplot as plt


# In[14]:

get_ipython().magic(u'matplotlib inline')


# In[15]:

plt.plot(test_data['sqft_living'], test_data['price'], '.',
        test_data['sqft_living'], sqft_model.predict(test_data),'-')


# In[16]:

sqft_model.get('coefficients')


# # explore other features in the data

# In[18]:

my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# In[19]:

sales[my_features].show()


# In[21]:

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# # Build a regressoin model with more features

# In[22]:

my_features_model=graphlab.linear_regression.create(train_data, target='price', features=my_features)


# In[23]:

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)


# # Apply models to print prices of threee houses

# In[31]:

house1 = sales[sales['id']=='5309101200']


# In[32]:

house1


# <img src="house-5309101200.jpg">

# In[33]:

print house1['price']


# In[42]:

print sqft_model.predict(house1)


# In[43]:

print my_features_model.predict(house1)


# # PRediction for a second fancier house

# In[44]:

house2=sales[sales['id']=='1925069082']


# In[45]:

print house2


# <img src="1925069082.jpg">

# In[47]:

sqft_model.predict(house2)


# In[48]:

my_features_model.predict(house2)


# ## LAst house is going to be super fancy

# In[49]:

bill_gates={'bedrooms':[8],
            'bathrooms':[25],
            'sqft_living':[50000],
            'sqft_lot':[225000],
            'floors':[4],
            'zipcode':['98039'],
            'condition':[10],
            'grade':[10],
            'waterfront':[1],
            'view':[4],
            'sqft_above':[37500],
            'sqft_basement':[12500],
            'yr_built':[1994],
            'yr_renovated':[2010],
            'lat':[47.627606],
            'long':[-122.242054],
            'sqft_living15':[5000],
            'sqft_lot15':[40000]
           }


# In[50]:

print my_features_model.predict(graphlab.SFrame(bill_gates))


# In[ ]:



