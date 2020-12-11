
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:39:15 2020

@author: Gul_a
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS)

from textblob import TextBlob

import re

from collections import Counter
import warnings
warnings.filterwarnings("ignore")

    
df=pd.read_csv('tertemiz.csv',delimiter=(';'))

#df = df[['veri']]

tweet = df.copy()

# data preprocessing
tweet['veri'] = tweet['veri'].str.lower()
for i in range(tweet.shape[0]) :
    tweet['veri'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([^\w\s])|(\d)", " ", tweet['veri'][i]).split())


def show_wordcloud(data , title = None):
    wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))
  
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    plt.title(title, size = 25)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

show_wordcloud(tweet['veri'])

#Removing Stop Words
#tweet['veri'] = tweet['veri'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

def remove_stopwords(df_fon):
    stopwords = open('stopwords', 'r').read().split()
    df_fon['veri'] = df_fon['veri'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

remove_stopwords(tweet)

print(remove_stopwords)

#lemmi
from textblob import Word
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in