#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import random
import gzip
import dateutil.parser
import math


# In[2]:


answers = {}
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


### Question 1
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[4]:


X = [len(d['review_text']) for d in dataset]
Y = [d['rating'] for d in dataset]
max_length = max(X)
X_scaled = [x / max_length for x in X]


# In[5]:


X_scaled = np.array(X_scaled).reshape(-1, 1)
Y = np.array(Y)
model = LinearRegression().fit(X_scaled, Y)
predictions = model.predict(X_scaled)
MSE = mean_squared_error(Y, predictions)


# In[6]:


answers['Q1'] = [model.intercept_, model.coef_[0], MSE]
answers


# In[7]:


assertFloatList(answers['Q1'], 3)


# In[8]:


### Question 2
for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t


# In[9]:


def feature(datum):
    t = datum['parsed_date']
    weekday = [0] * 6
    month = [0] * 11
    if t.weekday() != 0:
        weekday[t.weekday()-1] = 1
    if t.month != 1:
        month[t.month - 2] = 1
    return [1] + [len(datum['review_text'])/max_length] + weekday + month


# In[10]:


X2 = [feature(d) for d in dataset]
Y2 = [d['rating'] for d in dataset]


# In[11]:


answers['Q2'] = [X2[0], X2[1]]
assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)


# In[12]:


X2 = np.array(X2).reshape(-1, 19)
Y2 = np.array(Y2)
model2 = LinearRegression().fit(X2, Y2)
predictions2 = model2.predict(X2)
mse2 = mean_squared_error(Y2, predictions2)


# In[13]:


### Question 3


# In[14]:


def feature3(datum):
    t = datum['parsed_date']
    return [1] + [len(datum['review_text'])/max_length] + [t.weekday()] + [t.month]


# In[15]:


X3 = [feature3(d) for d in dataset]
Y3 = [d['rating'] for d in dataset]


# In[16]:


X3 = np.array(X3).reshape(-1, 4)
Y3 = np.array(Y3)
model3 = LinearRegression().fit(X3, Y3)
predictions3 = model3.predict(X3)
mse3 = mean_squared_error(Y3, predictions3)


# In[17]:


answers['Q3'] = [mse2, mse3]


# In[18]:


assertFloatList(answers['Q3'], 2)


# In[19]:


### Question 4


# In[20]:


random.seed(0)
random.shuffle(dataset)


# In[21]:


X2 = [feature(d) for d in dataset]
X3 = [feature3(d) for d in dataset]
Y = [d['rating'] for d in dataset]


# In[22]:


train2, test2 = X2[:len(X2)//2], X2[len(X2)//2:]
train3, test3 = X3[:len(X3)//2], X3[len(X3)//2:]
trainY, testY = Y[:len(Y)//2], Y[len(Y)//2:]


# In[23]:


train2 = np.array(train2).reshape(-1, 19)
train3 = np.array(train3).reshape(-1, 4)
trainY = np.array(trainY)
modeltrain2 = LinearRegression().fit(train2, trainY)
modeltrain3 = LinearRegression().fit(train3, trainY)


# In[24]:


predictiontext2 = modeltrain2.predict(test2)
predictiontext3 = modeltrain3.predict(test3)
test_mse2 = mean_squared_error(testY, predictiontext2)
test_mse3 = mean_squared_error(testY, predictiontext3)


# In[25]:


answers['Q4'] = [test_mse2, test_mse3]


# In[26]:


assertFloatList(answers['Q4'], 2)


# In[27]:


### Question 5


# In[28]:


f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))
dataset[:1]


# In[49]:


X = [[len(d['review/text'])] for d in dataset]
y = [1 if d['review/overall'] >= 4 else 0 for d in dataset]


# In[50]:


clf = LogisticRegression(class_weight='balanced')
clf.fit(X, y)
y_pred = clf.predict(X)


# In[51]:


TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()


# In[52]:


BER = 0.5 * (FP / (TN + FP) + FN / (TP + FN))


# In[53]:


answers['Q5'] = [TP, TN, FP, FN, BER]
answers['Q5'] 


# In[54]:


assertFloatList(answers['Q5'], 5)


# In[62]:


#Question 6
from sklearn.metrics import precision_score


# In[68]:


precs = []
y_scores = clf.predict_proba(X)[:, 1] 
def precision_at_k(k):
    sorted_indices = np.argsort(y_scores)[::-1]
    y_np = np.array(y)
    y_pred_np = np.array(y_pred)

    top_k_true = y_np[sorted_indices][:k]
    top_k_pred = y_pred_np[sorted_indices][:k]
    
    return precision_score(top_k_true, top_k_pred)


# In[69]:


for k in [1,100,1000,10000]:
    precs.append(precision_at_k(k))


# In[70]:


answers['Q6'] = precs
answers['Q6']


# In[71]:


assertFloatList(answers['Q6'], 4)


# In[72]:


### Question 7


# In[73]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[74]:


style_encoder = LabelEncoder()
styles = [d['beer/style'] for d in dataset]
style_encoder.fit(styles)


# In[75]:


X = []

for d in dataset:
    text_length = len(d['review/text'])
    style_encoded = style_encoder.transform([d.get('beer/style', '')])[0]
    abv = d.get('beer/ABV', 0)
    appearance = d.get('review/appearance', 0)
    palate = d.get('review/palate', 0)
    taste = d.get('review/taste', 0)
    aroma = d.get('review/aroma', 0)
    profile_name_length = len(d.get('user/profileName', ''))
    
    X.append([text_length, style_encoded, abv, appearance, palate, taste, aroma, profile_name_length])


# In[76]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[77]:


clf.fit(X, y)
y_pred = clf.predict(X)
TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
BER = 0.5 * (FP / (TN + FP) + FN / (TP + FN))
BER


# In[78]:


its_test_BER = 0.1750115410564314


# In[79]:


answers['Q7'] = ["Enhanced the model by incorporating features like the encoded beer style, alcohol by volume (ABV), individual ratings for appearance, palate, taste, and aroma, as well as the length of the user's profile name, in addition to the original review text length.", its_test_BER]


# In[80]:


f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:




