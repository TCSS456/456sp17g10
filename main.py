import tweepy
import json
import pprint

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

for tweet in tweepy.Cursor(api.search,
                           q=search_text,
                           count=10,
                           result_type="recent",
                           include_entities=True,
                           lang="en").items():
  tags = tweet.entities['hashtags'];
  for tag_obj in tags:
    tag = tag_obj['text'].lower();
    if tag in tag_freq_table:
      tag_freq_table[tag] += 1;
    else:
      tag_freq_table[tag] = 1;

for tag in tag_freq_table:
  freq = tag_freq_table[tag];
  print(tag + " : " + str(freq));
