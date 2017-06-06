import tweepy
import json
from pprint import pprint
import preprocessor as pre
from bs4 import BeautifulSoup

with open('creds.json') as data_file:    
    data = json.load(data_file)

consumer_key = data['consumer_key'];
consumer_secret = data['consumer_secret'];
access_token = data['access_token'];
access_token_secret = data['access_token_secret'];

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
 
search_text = raw_input("Enter a hashtag #")
tag_freq_table = {};

f = open('tweets.csv', 'w');

for tweet in tweepy.Cursor(api.search,
                           q=search_text,
                           count=1,
                           result_type="recent",
                           lang="en").items():
  text = tweet.text;
  text = BeautifulSoup(text.lower(), 'lxml').get_text()
  text = ''.join((c for c in text if ord(c) < 128))
  text = pre.clean(text);
  text = text + '\n';
  f.write(text);
