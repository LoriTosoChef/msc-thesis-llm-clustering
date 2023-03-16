import logging
import tweepy
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)

class TwitterBot:
    def __init__(self,
                 bearer: str,
                 api: str,
                 api_secret: str,
                 access: str,
                 access_secret: str,
                 limit: int = 10) -> None:
        
        logger.debug('Creating twitter bot instance')
        
        self.bearer = bearer
        self.api = api
        self.api_secret = api_secret
        self.access = access
        self.access_secret = access_secret
        self.limit = limit
        
        self.client = tweepy.Client(bearer_token=self.bearer,
                                    consumer_key=self.api,
                                    consumer_secret=self.api_secret,
                                    access_token=self.access,
                                    access_token_secret=self.access_secret)