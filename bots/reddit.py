import logging
import praw

logger = logging.getLogger(__name__)
for logger_name in ("praw", "prawcore"):
    reddit_logger = logging.getLogger(logger_name)
    reddit_logger.setLevel(logging.DEBUG)

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