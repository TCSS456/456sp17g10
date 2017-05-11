import tweepy
import json

with open('creds.json') as data_file:    
    data = json.load(data_file)

consumer_key = data['consumer_key'];
consumer_secret = data['consumer_secret'];
access_token = data['access_token'];
access_token_secret = data['access_token_secret'];

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text
