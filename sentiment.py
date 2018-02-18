# $ 0 U l $ h i f t 3 r


# IMPORTING DATA HANDLING LIBRARIES

import numpy as np

import pandas as pd
pd.set_option("display.height",2000)
pd.set_option("display.max_rows",2000)
pd.set_option("display.max_columns",2000)
pd.set_option("display.width",2000)
pd.set_option("display.max_colwidth",-1)


import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use("ggplot")

import seaborn as sns
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")

# DATA LOAD

train = pd.read_csv("/home/maniac/Desktop/Kaggle/Computational/train.csv",encoding="ISO-8859-1",header=None)
train = pd.DataFrame({"target":train[0],"id":train[1],"date":train[2],"flag":train[3],"user":train[4],"text":train[5]},columns=["text","id","date","flag","user","target"])

# DATA CHECK

# print(train.shape)
# print(train.describe())
# print(train.info())
# print(train.head())
# print(train["target"].unique())
# print(train["flag"].unique())

# REDUCING DATA

from sklearn.utils import shuffle
train = shuffle(train)
train = train.sample(frac=1).reset_index(drop=True)
train = train[:10000]
# print(train.shape)


# DATA PREPROCESSING

train["target"] = train["target"].replace(4,1)
# print(train["target"].unique())
# print(train.head())
train.drop(["id","date","flag","user"],axis=1,inplace=True)
# print(train.shape)
# print(train.head())


# TEXT DATA PREPROCESSING

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
# nltk.download()
# print(stopwords.words("english"))
lmtzr = WordNetLemmatizer()
stemmer = PorterStemmer()
cv = CountVectorizer(analyzer="word",stop_words=stopwords.words("english"),max_features=10000)

def cleaning(text) :
    text = BeautifulSoup(text).get_text()
    text = "".join([item for item in text if item not in string.punctuation])
    text = " ".join(re.sub(r"(@[A-Za-z0-9]+(tweeted:)?)|([^A-Za-z \t])|(http?\S*)|(https?\S*)|(\w+:\/\/\S+)", "", text).split())
    text = text.lower().split()
    text = [lmtzr.lemmatize(word) for word in text]
    text = [stemmer.stem(word) for word in text]
    return (" ".join(text))

training_clean = []
# print(train["text"].size)
for i in range(0,train["text"].size) :
    training_clean.append(cleaning(train["text"][i]))

train_feats = cv.fit_transform(training_clean)
train_feats = train_feats.toarray()
# print(train_feats)
# print(train_feats.shape)
# print(training_clean)
# print(train["text"][0])
# print("-----------------")
# train["text"][0] = cleaning(train["text"][0])
# print(train.head())
# print("-----------------")
# print(train["text"][0])
# train["text"] = train["text"].apply(lambda x:cleaning(x))
# print(train.head())
# print(train.shape)

# MODELLING

target = train["target"]
train.drop("target",axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
param = {"C":[0.1]}
clf = GridSearchCV(lr,param_grid=param,cv=10,n_jobs=-1)
clf.fit(train_feats,target)
print(clf.best_score_)