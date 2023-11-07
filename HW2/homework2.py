#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
import dateutil.parser
import time
import math

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
    
answers = {}

def parseData(fname):
    for l in open(fname):
        yield eval(l)
        
data = list(parseData("beer_50000.json"))

random.seed(0)
random.shuffle(data)

dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1
    
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
catID = dict(zip(list(categories),range(len(categories))))


# In[2]:


max_length_overall = max([len(entry['review/text']) for entry in data])


# In[3]:


def feat(d, includeCat=True, includeReview=True, includeLength=True):
    features = [0] * len(catID)
    
    if includeCat:
        if d['beer/style'] in catID:
            features[catID[d['beer/style']]] = 1
    
    if includeReview:
        features.append(d['review/aroma'])
        features.append(d['review/overall'])
        features.append(d['review/appearance'])
        features.append(d['review/palate'])
        features.append(d['review/taste'])
    
    if includeLength:
        scaled_length = len(d['review/text']) / max_length_overall
        features.append(scaled_length)
    
    return features


# In[4]:


def pipeline(reg, includeCat = True, includeReview = True, includeLength = True):
    # ...
    XTrain = [feat(d,includeCat,includeReview,includeLength) for d in dataTrain]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced', max_iter=1000)
    model.fit(XTrain, yTrain)
    
    XValid = [feat(d,includeCat,includeReview,includeLength) for d in dataValid]
    validPredictions = model.predict(XValid)
    validBER = 1 - sum(validPredictions == yValid) / len(yValid)
    
    XTest = [feat(d,includeCat,includeReview,includeLength) for d in dataTest]
    testPredictions = model.predict(XTest)
    testBER = 1 - sum(testPredictions == yTest) / len(yTest)
    
    return model, validBER, testBER


# In[5]:


### Question 1


# In[6]:


mod, validBER, testBER = pipeline(10, True, False, False)
answers['Q1'] = [validBER, testBER]
assertFloatList(answers['Q1'], 2)


# In[7]:


### Question 2


# In[8]:


mod, validBER, testBER = pipeline(10)
answers['Q2'] = [validBER, testBER]
assertFloatList(answers['Q2'], 2)


# In[9]:


### Question 3


# In[10]:


best_valid_BER = float('inf') 
for c in [0.001, 0.01, 0.1, 1, 10]:
    mod, validBER, testBER = pipeline(c)
    print(f"For C = {c}, Validation BER = {validBER}")  
    if validBER < best_valid_BER:
        best_valid_BER = validBER
        best_C = c


# In[11]:


mod, validBER, testBER = pipeline(best_C)
answers['Q3'] = [best_C, validBER, testBER]
assertFloatList(answers['Q3'], 3)


# In[12]:


### Question 4


# In[13]:


mod, validBER, testBER_noCat = pipeline(1,False,True,True)
mod, validBER, testBER_noReview = pipeline(1,True,False,True)
mod, validBER, testBER_noLength = pipeline(1,True,True,False)
answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]
assertFloatList(answers['Q4'], 3)


# In[14]:


### Question 5


# In[15]:


path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')
dataset = []
pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        #print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    t = dateutil.parser.parse(d['review_date'])
    d['parsed_date'] = t
    dataset.append(d)


# In[16]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[17]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
timestampDict = {} # To retrieve a timestamp for a specific user/item pair
reviewsPerUser = defaultdict(list)

for d in dataTrain:
    user, item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user, item)] = d['star_rating']
    reviewsPerUser[user].append(d)
    timestampDict[(user, item)] = d['parsed_date']


# In[18]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    userAverages[u] = sum([ratingDict[(u, i)] for i in itemsPerUser[u]]) / len(itemsPerUser[u])
    
for i in usersPerItem:
    itemAverages[i] = sum([ratingDict[(u, i)] for u in usersPerItem[i]]) / len(usersPerItem[i])

allRatings = [d['star_rating'] for d in dataTrain]
ratingMean = sum(allRatings) / len(allRatings)


# In[19]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[20]:


def mostSimilar(i, N=10):
    users = usersPerItem[i]
    similarities = [(Jaccard(users, usersPerItem[i2]), i2) for i2 in usersPerItem if i2 != i]
    similarities.sort(reverse=True)
    return similarities[:N]


# In[21]:


query = 'B00KCHRKD6'
ms = mostSimilar(query, 10)
answers['Q5'] = ms
assertFloatList([m[0] for m in ms], 10)


# In[22]:


### Question 6


# In[23]:


def MSE(y, ypred): ##L2 
    differences = [(a - b)**2 for a, b in zip(y, ypred)]
    return sum(differences) / len(differences)


# In[24]:


def predictRating(user, item):
    ratings = []
    similarities = []
    items_rated_by_user = itemsPerUser[user]
    for j in items_rated_by_user:
        if j != item:
            ratings.append(ratingDict[(user, j)] - itemAverages[j])
            similarities.append(Jaccard(usersPerItem[item], usersPerItem[j]))
    
    if sum(similarities) > 0:
        return sum([a*b for a,b in zip(ratings, similarities)]) / sum(similarities) + itemAverages[item]
    else:
        return itemAverages.get(item, ratingMean)


# In[25]:


labels = [d['star_rating'] for d in dataTest]
simPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataTest]


# In[26]:


answers['Q6'] = MSE(simPredictions, labels)
assertFloat(answers['Q6'])


# In[27]:


### Question 7


# In[28]:


def mypredictRating(user, item):
    numerator = 0  
    denominator = 0 
    target_timestamp_unix = 0   # Set a default value
    items_rated_by_user = itemsPerUser[user]
    target_timestamp = timestampDict.get((user, item), None)
    if target_timestamp:
        target_timestamp_unix = time.mktime(target_timestamp.timetuple())

    lambda_param = 10
    
    for j in items_rated_by_user:
        if j != item:
            rating_diff = ratingDict[(user, j)] - itemAverages[j]
            
            current_timestamp = timestampDict.get((user, j), None)
            if current_timestamp:
                current_timestamp_unix = time.mktime(current_timestamp.timetuple())
                time_diff_in_seconds = abs(target_timestamp_unix - current_timestamp_unix)
                time_diff_in_days = time_diff_in_seconds / (24 * 60 * 60)  # Convert seconds to days
            else:
                time_diff_in_days = 0

            time_decay = math.exp(-lambda_param * time_diff_in_days)
            similarity = Jaccard(usersPerItem[item], usersPerItem[j])
            weighted_similarity = similarity * time_decay

            numerator += weighted_similarity * rating_diff
            denominator += weighted_similarity

    if denominator > 0:
        return itemAverages[item] + numerator / denominator
    else:
        return itemAverages.get(item, ratingMean)


# In[29]:


labels = [d['star_rating'] for d in dataTest]
mysimPredictions = [mypredictRating(d['customer_id'], d['product_id']) for d in dataTest]


# In[30]:


itsMSE = MSE(mysimPredictions, labels)


# In[31]:


answers['Q7'] = ["This algorithm predicts a user's rating for a given item by calculating a time-decayed weighted similarity between items the user has rated in the past and the target item. The time decay is modeled using an exponential function controlled by a parameter lambda, which is empirically determined through experimentation.", itsMSE]


# In[32]:


assertFloat(answers['Q7'][1])


# In[33]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




