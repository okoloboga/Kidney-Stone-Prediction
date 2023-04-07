#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix

from keras import models
from keras import layers
from tensorflow import keras
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
train.head(10)


# Ok, binary classification quest with metric based on confusion matrix.
# Some features are different, so i'll use standart scaler.

# In[ ]:


target = train['target']

del train['id']
del train['target']

scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train),
                     columns = train.columns)
train.head(5)


# Visualization, maybe some features could be groupped by dimensionality reduction by PCA

# In[ ]:


colors = {1: 'orange', 0: 'blue'}

def visual(df):
    
    scatter_matrix = pd.plotting.scatter_matrix(
                     df, s = 50, alpha = 1, 
                     figsize = (25, 25), grid = True, 
                     marker = 'o', c = target.map(colors)) 

    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(),
                      fontsize = 20,
                      rotation = 45)
        ax.set_ylabel(ax.get_ylabel(),
                      fontsize = 20,
                      rotation = 45)
    return scatter_matrix

visual(train)


# 

# Oh, it so similar and mixed... Should be difficult, no way to use PCA. Let's see correlation heatmap 

# In[ ]:


print(train.corr())
  
dataplot = sb.heatmap(train.corr(),
                      cmap="YlGnBu",
                      annot = True)
x_train, x_test, y_train, y_test = train_test_split(train, target, 
                                                    test_size = 0.2,
                                                    random_state = 17)
x_train 
plt.show()


# I will train models on the most difficult pair, with coldest correlation

# In[ ]:


def result(model):
    
    pred = model.predict(x_test)
    accuracy = confusion_matrix(y_test, pred)
    print(f'{accuracy}')

    colors = {1: 'orange', 0: 'blue'}
    
    a = plt.scatter(x_test['gravity'], x_test['ph'],
                    s = 25, alpha = 1, marker = 'o', c = pred)
    plt.show(a)

    b = plt.scatter(x_test['gravity'], x_test['ph'],
                    s = 25, alpha = 1, marker = 'o', c = y_test)
    plt.show(b)


# CatBoost - 81.92%

# In[ ]:


model_0 = CatBoostClassifier(
    iterations = 1000, 
    learning_rate = 0.02,
    random_seed = 43,
    ).fit(
    x_train, y_train,
    eval_set = (x_test, y_test),
    verbose = False,
    )
result(model_0) 


# K-Neighbors - 80.72%

# In[ ]:


model_1 = KNeighborsClassifier(n_neighbors = 17).fit(x_train, y_train)

result(model_1)


# Random Forest - 83.13%

# In[ ]:


model_2 = RandomForestClassifier(n_estimators = 2000,     
                                 criterion='gini',
                                 max_depth = 25,                    
                                 max_features = None,
                                 min_weight_fraction_leaf = 0.015,
                                 random_state = 1337,
                                 ).fit(x_train, y_train)
result(model_2)


# Dense NN - 86.75% - i will use that model for making submission

# In[ ]:


model = keras.Sequential([
        layers.Dense(256, activation = "relu"),
        layers.Dropout(0.1),
        layers.Dense(128, activation = "relu"),
        layers.Dropout(0.1),
        layers.Dense(64, activation = "relu"),
        layers.Dense(32, activation = "relu"),
        layers.Dense(16, activation = "relu"),
        layers.Dense(1, activation = "sigmoid")
        ])

optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001)
model.compile(optimizer = optimizer,
             loss = "binary_crossentropy",
             metrics = ["accuracy"]
             )
model.fit(x_train, y_train,
          epochs = 37, batch_size = 8)

pred = np.round(model.predict(x_test))
accuracy = confusion_matrix(y_test, pred)
print(f'{accuracy}')

a = plt.scatter(x_test['gravity'], x_test['ph'],
            s = 25, alpha = 1, marker = 'o', c = pred)
plt.show(a)

b = plt.scatter(x_test['gravity'], x_test['ph'],
            s = 25, alpha = 1, marker = 'o', c = y_test)
plt.show(b)


# In[ ]:


test = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')

del test['id']

scaler = StandardScaler()
test = pd.DataFrame(scaler.fit_transform(test),
                    columns = test.columns)
test.head(10)


# In[ ]:


prediction = model.predict(test)

submission = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')
submission = pd.DataFrame(submission['id'])
target = pd.DataFrame(data = prediction, 
                     columns = ['target'])
submission = pd.concat([submission, target], axis = 1)
submission.to_csv('/kaggle/working/submission.csv', index = False)
submission.head(5)

