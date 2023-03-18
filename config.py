import os
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)-8s | %(name)-10s | %(message)s',
                    level=logging.INFO)

logger.info('Loading environment variables')

load_dotenv()

# Global Variables
DATA_DIR = 'raw_data'
try:
    os.mkdir(f'{DATA_DIR}')
except FileExistsError:
    logger.warning(f'Directory {DATA_DIR} already exists')

# Twitter
TWTR_API_DEV = os.environ.get('TWTR_API_DEV')
TWTR_API_DEV_SECRET = os.environ.get('TWTR_API_DEV_SECRET')
TWTR_BEARER_DEV_TOKEN = os.environ.get('TWTR_BEARER_DEV_TOKEN')

TWTR_API = os.environ.get('TWTR_API')
TWTR_API_SECRET = os.environ.get('TWTR_API_SECRET')
TWTR_BEARER_TOKEN = os.environ.get('TWTR_BEARER_TOKEN')

TWTR_ACCESS_TOKEN = os.environ.get('TWTR_ACCESS_TOKEN')
TWTR_ACCESS_TOKEN_SECRET = os.environ.get('TWTR_ACCESS_TOKEN_SECRET')

# Reddit
REDDIT_APP_ID = os.environ.get('REDDIT_APP_ID')
REDDIT_SECRET = os.environ.get('REDDIT_SECRET')
REDDIT_REDIRECT_URL = os.environ.get('REDDIT_REDIRECT_URL')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT')