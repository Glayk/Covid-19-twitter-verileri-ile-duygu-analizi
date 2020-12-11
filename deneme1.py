# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:39:15 2020

@author: Gul_a
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
import nltk
import re

from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from nltk import word_tokenize
from nltk.corpus import stopwords
from snowballstemmer import TurkishStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

#For display all output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


data=pd.read_csv('tertemiz.csv',delimiter=(';'))
#df = df[['veri']]

#sayisallastirma
data["duygu"].replace(1, value = "pozitif", inplace = True)
data["duygu"].replace(0, value = "negatif", inplace = True)

labels = Counter(data['duygu']).keys()
sum_ = Counter(data['duygu']).values()
df = pd.DataFrame(zip(labels,sum_), columns = ['duygu', 'Toplam'])

"""
#etiketlerin görselleştirilmesi - çubuk grafiği
df.plot(x = 'duygu' , y = 'Toplam',kind = 'bar', legend = False, grid = True, figsize = (15,5))
plt.title('Kategori Sayılarının Görselleştirilmesi', fontsize = 20)
plt.xlabel('Kategoriler', fontsize = 15)
plt.ylabel('Toplam', fontsize = 15);
#etiketlerin görselleştirilmesi - pasta grafiği
fig, ax = plt.subplots(figsize=(15, 10))
ax.pie(df.Toplam, labels =df.duygu, autopct = '%1.2f%%',  startangle = 90 )
ax.axis('equal')
"""

# veri onisleme
tweet = data.copy()
tweet['veri'] = tweet['veri'].str.lower()
for i in range(tweet.shape[0]) :
    tweet['veri'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([^\w\s])|(\d)", " ", tweet['veri'][i]).split())

stoplist=stopwords.words('turkish')
def remove_stopwords(veri):
    return " ".join([word for word in str(veri).split() if word not in stoplist])
tweet['stop_veri'] = tweet['veri'].apply(lambda veri: remove_stopwords(veri))

"""
def stemming_tokenizer(veri): 
    stemmer = TurkishStemmer()
    return [stemmer.stemWord(w) for w in word_tokenize(veri)]
tweet['token_veri'] = tweet['stop_veri'].apply(lambda veri: stemming_tokenizer(veri))
print(tweet['token_veri'].head(20))
"""

#a=tweet.loc[tweet.duygu == "pozitif" ]
a=tweet.loc[tweet.duygu.isin(['pozitif', 'negatif'])]


tweetDoc = a['stop_veri'].values.tolist()
tweetClass = a['duygu'].values.tolist()


#test ve train olarak ayırma 
x_train, x_test, y_train, y_test = train_test_split(tweetDoc, tweetClass, test_size = 0.25, random_state = 42)
print('The length of the trainning set are {}'.format(len(x_train)))
print('The length of the test set are {}'.format(len(x_test)))

#tfidf 
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df = 5)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#logistic regresyon
model = LogisticRegression()
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)
print ("Accuracy={}".format(accuracy_score(y_test, y_pred)))
logisticpred = accuracy_score(y_test, y_pred)


"""
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(accuracy_score), size = 15)
plt.show()

"""
"""
print(model.score(x_train_tfidf, y_train))
print(model.score(x_test_tfidf, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
"""




        