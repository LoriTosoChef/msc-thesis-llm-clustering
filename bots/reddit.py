from typing import List

import logging
import praw

logger = logging.getLogger(__name__)
for logger_name in ("praw", "prawcore"):
    reddit_logger = logging.getLogger(logger_name)
    reddit_logger.setLevel(logging.INFO)

class RedditBot:
    def __init__(self,
                 client_id: str,
                 client_secret: str,
                 redirect_url: str,
                 user_agent: str = 'MyApp 0.0.1') -> None:
        
        logger.debug('Creating reddit bot instance')
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_url = redirect_url
        self.user_agemt = user_agent
        
        self.client = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.client_secret,
                                  redirect_url=self.redirect_url,
                                  user_agent=user_agent)
        
    def get_reddit_posts(self,
                         subreddit_list: List[str],
                         limit: int = 10) -> List[dict]:
        
        try:
            logger.debug(f'Receiving {limit} reddit posts')
            posts_data = self.client.subreddit(display_name=subreddit_list).top(time_filter='all', limit=limit)
        except Exception as e:
            logger.warning(f'Could not get recent reddit posts, returning empty dicts - {e}')
            return [{}]
        
        posts = []
        for post in posts_data:
            d = {}
            d['post_id'] = post.id
            d['text'] = post.title + '\n' + post.selftext
            posts.append(d)
            
        return posts