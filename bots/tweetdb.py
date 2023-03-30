import os
import sys
import time
import logging
import pandas as pd

from twitter import TwitterBot

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import config
from helpers.data_helpers import save_to_parquet

DOMAINS = [
    '11',  # sport
    '12',  # sports team
    '13',  # place
    '45',  # brand vertical
    '46',  # brand category
    '47',  # brand
    '48',  # product
    '65',  # interest and hobbies vertical
    '66',  # interest and hobbies category
    '67',  # interest and hobbies
    '71',  # video game
    '152', # food
    '162', # exercise and fitness
    '163', # travel
    '165', # technology
    '173', # google product taxonomy
]

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # read csv (downloaded from https://github.com/twitterdev/twitter-context-annotations/tree/main/files)
    contexts = pd.read_csv('../context-entities.csv')
    # get only selected domains
    contexts = contexts.loc[contexts['domains'].isin(DOMAINS)].copy()
    # convert to int id to str
    contexts['entity_id'] = contexts['entity_id'].astype('str')
    # build query col
    contexts['query'] = contexts['domains'] + '.' + contexts['entity_id']
    # export to list
    queries_list = contexts['query'].to_list()
    entities_list = contexts['entity_name'].to_list()
    
    queries_db = dict(zip(queries_list, entities_list))
    
    # initialize twitter bot
    twitter_bot = TwitterBot(bearer=config.TWTR_BEARER_TOKEN,
                            api=config.TWTR_API,
                            api_secret=config.TWTR_API_SECRET,
                            access=config.TWTR_ACCESS_TOKEN,
                            access_secret=config.TWTR_ACCESS_TOKEN_SECRET)
    
    dfs = []
    running_total = 0
    for i, (q, e) in enumerate(queries_db.items()):
        query = f'context:{q} -is:retweet lang:en'
        try:
            tweets = twitter_bot.get_recent_tweets(query=query, limit=5000)
        except Exception as e:
            logger.info(f'Hitting limit, waiting...')
            time.sleep(900)
            
        df = pd.DataFrame.from_dict(tweets)
        if len(df) != 0:
            dfs.append(df)
            
        current_len = len(df)
        running_total += current_len    
        logger.info(f'{i} | Found {current_len} - Running Total {running_total} | Context {q} - {e}')
        

    tweets = pd.concat(dfs, ignore_index=True)
    tweets = tweets.loc[~tweets['id'].duplicated()].copy()
    save_to_parquet(data_dir=config.RAW_DATA_DIR, df=tweets, name='tweets_db')