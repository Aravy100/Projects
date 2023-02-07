# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:40:02 2023

@author: Aravindh Saravanan
"""

import requests  
import re
import os
import tweepy   
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt


# Decide on topics first 

############################################################################
################################ 1. News API ###############################
############################################################################

topics = ['indian cuisine', 'chinese cuisine', 
          'italian cuisine', 'mediterranean cuisine', 'french cuisine']

df = pd.DataFrame(columns=['source','author','title','description',
                           'url','urlToImage','publishedAt','content','Label'])
for topic in topics:
    endpoint = "https://newsapi.org/v2/everything"
    URLPost = {'apiKey':'7eed1a9ac23a4c6daac50395f479db3e',
               'q':topic,
               'searchIn':'content'}    
    response = requests.get(endpoint, URLPost)
    json = response.json()
    df=df.append(pd.DataFrame(json['articles']),ignore_index=True)
    df['Label']=df['Label'].replace(np.nan,topic)
    
# We will retain the description and content for now. Description is a short summary 
# while content is the entire contents of the news article
df = df[['Label','description','content']]

# Creating a true copy of the original data to perform cleaning
test = df.copy()

############## Data Cleaning ##############
# Label was created manually and requires no cleaning. Description and Content are
# Obtained from NewsAPI and therefore require extensive cleaning.

# Replace the new line character with space
test['description'] = test['description'].str.replace('\n', ' ')
test['content'] = test['content'].str.replace('\n', ' ')

# Remove non alpha characters (BUT keep spaces) from description and content
test['description'] = test['description'].str.replace('[^a-zA-Z ]', '')
test['content'] = test['content'].str.replace('[^a-zA-Z ]', '')

# Convert all text to lowercase
test['description'] = test['description'].str.lower()
test['content'] = test['content'].str.lower()

# Remove all stopwords from description and content
# Stopwords also include our label words and the word chars which occurs in our data for some reason
stop = stopwords.words('english') + ['cuisine', 'indian', 'italian', 'french', 'mediterranean', 'chinese', 'chars']
test['description'] = test['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test['content'] = test['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Remove very long words. Here we remove any word that is longer than 10 characters
test['description'] = test['description'].apply(lambda x: ' '.join([word for word in x.split() if len(word)<=10]))
test['content'] = test['content'].apply(lambda x: ' '.join([word for word in x.split() if len(word)<=10]))

# Remove very long words. Here we remove any word that is longer than 10 characters
test['description'] = test['description'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>=4]))
test['content'] = test['content'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>=4]))


# Stemming: This reduces variations of a word by clipping the ends of group words. Cooking, Cooked => Cook
a_stemmer=PorterStemmer()   # Instantiate our Stemmer object
stemmed = test.copy()       # Create a deep copy of the data so far to compare the results
stemmed['content'] = stemmed['content'].apply(lambda x: ' '.join([a_stemmer.stem(word) for word in x.split()]))

# Lemmatization: Does the same thing as Stemming but is smarter as it takes into account the context.
lemmatizer = WordNetLemmatizer()    # Instantiate our Stemmer object
lemmatized = test.copy()            # Create a deep copy of the data so far to compare the results
lemmatized['content'] = lemmatized['content'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Final dataframe with label and content from NewsAPI data. Remove description here.
newsapi_clean = lemmatized[['Label','content']]
# Stemming has yielded some poor words and therefore we will rely on lemmatization




############################################################################
################################ 2. Twitter ################################
############################################################################

# Source: https://towardsdatascience.com/how-to-extract-data-from-the-twitter-api-using-python-b6fbd7129a33

# Defining Twitter keys. WARNING! KEYS HAVE BEEN REMOVED AND HENCE WILL NOT RUN.
consumer_key = 'E96gnldPbFw5zJsk74eTNh6ZZ'
consumer_secret = 'Qo7cOwEOgqtbZA83U7P4wONtYK5cHVqXLr8ek6lV3Fg3cmzS1O'
access_token = '1622350542340227072-oxKEQyUnqxYQPZgDpIrlWhRRU7el18'
access_token_secret = 'hU0gT795jJsWFDLDcn3l5DEYe35dH34yDTLQCiq7aYX1m'

# Instantiating Twitter API and passing on the keys
auth = tweepy.OAuth1UserHandler(
  consumer_key, 
  consumer_secret, 
  access_token, 
  access_token_secret
)
api = tweepy.API(auth)

# Create two empty lists one for label and another for the actual tweets from the API
tweet_label = []
extracted_tweets = []

# Extracting tweets for our topics
for topic in topics:
    for status in tweepy.Cursor(api.search_tweets, 
                                topic, 
                                lang="en",
                                count=100,                          # extract 100 per topic
                                tweet_mode="extended").items(250):  # items is redundant here
        extracted_tweets.append(status._json['full_text'])
        tweet_label.append(topic)
        

############## Data Cleaning ##############
# First store the two lists into a dataframe        
tweets = pd.DataFrame({'tweet_label':tweet_label,'extracted_tweets':extracted_tweets})

# Remove hyperlinks from the tweets
tweets['extracted_tweets']=tweets['extracted_tweets'].str.replace('http[^\s]+', '')

# Remove user handles ('@') from the tweets
tweets['extracted_tweets']=tweets['extracted_tweets'].str.replace('@[^\s]+', '')

# Remove non alpha characters from the tweets
tweets['extracted_tweets']=tweets['extracted_tweets'].str.replace('[^a-zA-Z ]', '')

# Convert to lowercase
tweets['extracted_tweets']=tweets['extracted_tweets'].str.lower()

# Remove all stopwords from description and content
# Stopwords also include our label words and the word chars which occurs in our data for some reason
stop = stopwords.words('english') + ['cuisine', 'indian', 'italian', 'french', 'mediterranean', 'chinese', 'rt']
tweets['extracted_tweets'] = tweets['extracted_tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Stemming: This reduces variations of a word by clipping the ends of group words. Cooking, Cooked => Cook
a_stemmer=PorterStemmer()   # Instantiate our Stemmer object
stemmed_tweets = tweets.copy()       # Create a deep copy of the data so far to compare the results
stemmed_tweets['extracted_tweets'] = stemmed_tweets['extracted_tweets'].apply(lambda x: ' '.join([a_stemmer.stem(word) for word in x.split()]))

# Lemmatization: Does the same thing as Stemming but is smarter as it takes into account the context.
lemmatizer = WordNetLemmatizer()    # Instantiate our Stemmer object
lemmatized_tweets = tweets.copy()            # Create a deep copy of the data so far to compare the results
lemmatized_tweets['extracted_tweets'] = lemmatized_tweets['extracted_tweets'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Final dataframe with label and content from NewsAPI data. Remove description here.
twitter_clean = lemmatized_tweets[['tweet_label','extracted_tweets']]

# Remove very long words. Here we remove any word that is longer than 10 characters
twitter_clean['extracted_tweets'] = twitter_clean['extracted_tweets'].apply(lambda x: ' '.join([word for word in x.split() if len(word)<=10]))

# Likewise, Remove very short words. Here we remove any word that is shorter than 4 characters
twitter_clean['extracted_tweets'] = twitter_clean['extracted_tweets'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>=4]))



############################################################################
################################ 3. Analysis ###############################
############################################################################

# Word Clouds: Word clouds are an excellent way to visualize data.
# We will compare data from NewsAPI and twitter before and after cleaning

# Pass the required dataframe column to text and then run the wordcloud code chunk below.
# NewsAPI Raw:
text = " ".join(i for i in df.content)
# NewsAPI Clean:
text = " ".join(i for i in newsapi_clean.content)

# Twitter Raw:
text = " ".join(i for i in extracted_tweets)
# NewsAPI Clean:
text = " ".join(i for i in twitter_clean.extracted_tweets)


# WordCloud Code: The below code generates a wordcloud image by taking in a string under "text"
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Let us combine the newsapi and twitter
twitter_clean.rename(columns={"tweet_label":"Label", "extracted_tweets":"content"}, inplace=True)
data = newsapi_clean.append(twitter_clean, ignore_index=True)

# CountVectorizer
vectorizer = CountVectorizer(input="content",lowercase=True,max_features=5000)          # Initiate Count Vectorizer object
content_list = data['content'].tolist() # Convert dataframe column to a list
vec_matrix = vectorizer.fit_transform(content_list)
vec_array = vec_matrix.toarray()
count_vectorized_df = pd.DataFrame(vec_array, columns=vectorizer.get_feature_names())
count_vectorized_df['Label']=data['Label']

# TF-IDF Vectorizer
t_vectorizer = TfidfVectorizer(input="content",lowercase=True, max_features=5000)
t_matrix = t_vectorizer.fit_transform(content_list)
t_array = t_matrix.toarray()
df_tfidf = pd.DataFrame(data=t_array, columns=t_vectorizer.get_feature_names())
df_tfidf['Label']=data['Label']

################################### THE END ###################################
