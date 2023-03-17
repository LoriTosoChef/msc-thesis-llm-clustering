from typing import List

import logging
import tweepy
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)
twitter_logger = logging.getLogger("tweepy")
twitter_logger.setLevel(logging.INFO)

class TwitterBot:
    def __init__(self,
                 bearer: str,
                 api: str,
                 api_secret: str,
                 access: str,
                 access_secret: str) -> None:
        
        logger.debug('Creating twitter bot instance')
        
        self.bearer = bearer
        self.api = api
        self.api_secret = api_secret
        self.access = access
        self.access_secret = access_secret
        
        self.client = tweepy.Client(bearer_token=self.bearer,
                                    consumer_key=self.api,
                                    consumer_secret=self.api_secret,
                                    access_token=self.access,
                                    access_token_secret=self.access_secret)
        
    def get_recent_tweets(self,
                          query: str,
                          tweet_fields: list = ['created_at', 'context_annotations'],
                          limit: int = 10) -> List[dict]:
        if limit <= 100:
            try:
                logger.debug(f'Receiving {limit} tweets')
                tweets_data = self.client.search_recent_tweets(query=query,
                                                               tweet_fields=tweet_fields,
                                                               max_results=limit)
                tweets = tweets_data.data
            except Exception as e:
                logger.warning(f'Could not get recent tweets, returning empty dicts - {e}')
                return [{}]    
        else:
            try:
                logger.debug(f'Receiving {limit} tweets using paginator')
                tweets = tweepy.Paginator(self.client.search_recent_tweets,
                                          query=query,
                                          tweet_fields=tweet_fields,
                                          max_results=100).flatten(limit=limit)
            except Exception as e:
                logger.warning(f'Could not get recent tweets, returning empty dicts - {e}')
                return [{}]
            
        tweet_list = []
        for tweet in tweets:
            d = {}
            d['id'] = tweet.id
            d['created_at'] = tweet.created_at
            d['text'] = tweet.text
            d['entities'] = set([en['entity']['name'] for en in tweet.context_annotations])
            
            tweet_list.append(d)
        
        return tweet_list