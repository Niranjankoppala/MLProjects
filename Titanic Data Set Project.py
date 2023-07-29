#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
# Let's import some libraries to get started!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # The Data
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[2]:


train = pd.read_csv('Titanic.csv')
train.head()


# ### Exploratory Data Analysis
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# ###Missing Data
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[3]:


train.isnull().sum()


# In[4]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in train.columns if train[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values
features_with_na


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[13]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=20)


# In[14]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ### Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. 
# One way to do this is by filling in the mean age of all the passengers (imputation). 
# However we can be smarter about this and check the average age by passenger class. For example:

# In[15]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense.
# We'll use these average age values to impute based on Pclass for Age.

# In[16]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# Now apply that function!

# In[17]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[19]:


train.drop('Cabin',axis=1,inplace=True)


# In[20]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[21]:


train.head()


# ### Converting Categorical Features
# We'll need to convert categorical features to dummy variables using pandas!
# Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[22]:


train.info()


# In[23]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[24]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[25]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[26]:


train.head()


# In[27]:


train = pd.concat([train,sex,embark],axis=1)


# In[28]:


train.head()


# ### Building a Logistic Regression model
# Let's start by splitting our data into a training set and test set (there is another test.csv file 
# that you can play around with in case you want to use all this data for training).

# In[30]:


train.drop('Survived',axis=1).head()


# In[31]:


train['Survived'].head()


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# ### Training and Predicting

# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[36]:


predictions = logmodel.predict(X_test)


# In[37]:


from sklearn.metrics import confusion_matrix


# In[38]:


accuracy=confusion_matrix(y_test,predictions)


# In[39]:


accuracy


# In[40]:


from sklearn.metrics import accuracy_score


# In[41]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[42]:


predictions


# ### Evaluation
# We can check precision,recall,f1-score using classification report!

# In[43]:


from sklearn.metrics import classification_report


# In[44]:


print(classification_report(y_test,predictions))


# In[ ]:




