import tweepy
import pandas as pd
import numpy as np
import re
from html.parser import HTMLParser
import itertools
from autocorrect import Speller
from textblob import TextBlob
import matplotlib.pyplot as plt

spell = Speller(lang='en')
plt.style.use('dark_background')


#DATA EXTRACTION

#api variables
consumerKey = '79mgkI86A8DNg2v1tKmzITH4F'
consumerSecret = 'S6c8LTVYqRIGafJ7ZK5cPmoO1UtPJflk1qtsRh7aiufuD0TmSw'
accessToken = '1396109358472204294-kPER4XQo8w5JE17vXYO0ISfvuT1dq4'
accessTokenSecret = 'IfG38xxcIfsgAzLHQmRKLFBfzzFeCa1ihJl55DoX28MRH'

#authentication of twitter app
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)
api=tweepy.API(authenticate, wait_on_rate_limit=True)

#extracting relevant tweets
search_term = 'bitcoin -filter:retweets'
tweets = tweepy.Cursor(api.search, q=search_term, lang='en', since= '2021-04-01', tweet_mode= 'extended').items(1000)
all_tweets = [tweet.full_text for tweet in tweets]

df = pd.DataFrame(all_tweets, columns=['Tweets'])
#print(df)

#PREPROCESSING

def cleanTwt(t):
    t=HTMLParser().unescape(t)   #remove HTML characters
    t = re.sub(r'https?:\/\/.\S+', "", t)  #remove hyperlinks
    t = re.sub(r'#', '', t)   #remove hashtags
    t = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)",t) if s])   #splitting joint words
    t = t.lower()
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t))   #standardizing - one letter should not be present more than twice consecutively
    t = spell(t)   #spellcheck
    return t

df['Cleaned_Tweets'] = df['Tweets'].apply(cleanTwt)
print(df)
df.to_csv('C:/Users/ip330s/Desktop/IT/Lab Docs/bitcoin/output.csv',encoding='utf-8')


#FEATURE EXTRACTION AND SELECTION

#get subjectivity and polarity of tweets
def getSub(t):
    return TextBlob(t).sentiment.subjectivity

def getPol(t):
    return TextBlob(t).sentiment.polarity

df['Subjectivity'] = df['Cleaned_Tweets'].apply(getSub)
df['Polarity'] = df['Cleaned_Tweets'].apply(getPol)

#plotting polarity and subjectivity
plt.figure(figsize=(10,10))
for i in range(0, df.shape[0]):
    plt.scatter( df['Subjectivity'][i], df['Polarity'][i] )
plt.title('Bitcoin Sentiment Analysis- Scatter Plot')
plt.xlabel('Subjectivity: Objective to Subjective')
plt.ylabel('Polarity: Negative to Positive')
plt.show()


#CLASSIFICATION

#extract sentiment text
neg = 0
pos = 0
neutral = 0

def getSen(val):
    global neg
    global pos
    global neutral
    if val<0:
        neg+=1
        return 'Negative'
    elif val>0:
        pos+=1
        return 'Positive'
    else:
        neutral+=1
        return 'Neutral'

df['Sentiment'] = df['Polarity'].apply(getSen)

#plotting bar graph
df['Sentiment'].value_counts().plot(kind='bar')
plt.title('Bitcoin Sentiment Analysis- Bar Graph')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

#plotting pie chart
neg = neg/1000.0 * 100
pos = pos/1000.0 * 100
neutral = neutral/1000.0 * 100

labels = ['Positive ['+str(pos)+'%]' , 'Neutral ['+str(neutral)+'%]' ,'Negative ['+str(neg)+'%]']
sizes = [pos, neutral, neg]
patches, texts = plt.pie(sizes, startangle=90)
plt.legend(labels)
plt.title( 'Bitcoin Sentiment Analysis- Pie Chart' )
plt.axis('equal')
plt.show()
