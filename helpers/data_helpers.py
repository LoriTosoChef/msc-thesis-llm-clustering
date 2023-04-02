import logging
import re
import pandas as pd

from datetime import datetime

logger = logging.getLogger(__name__)


def clean_tweets_df(df: pd.DataFrame, text_col: str, n: int) -> pd.DataFrame:
    """Helper function to clean dataframes containing tweets"""
    
    logger.debug('Dropping tweets with less than {n} words')
    logger.debug('Adding starter word "Tweet"')
    
    # drop tweets with less than n words
    df['splits'] = df[text_col].str.split()
    df['len_splits'] = df['splits'].map(lambda x: len(x))
    df = df.loc[df['len_splits'] >= n].copy()
    
    # Add starter word
    df['full_text'] = '\nTWEET: ' + df[text_col]
    
    # Clean dataset
    df = df.drop(columns=['splits', 'len_splits', text_col])
    df = df.reset_index(drop=True)
    
    return df


def save_to_parquet(data_dir: str, df: pd.DataFrame, name: str) -> None:
    """Helper function to save dataframe to parquet file"""
    logger.debug('Saving {name} to .parquet...')
    
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    year = str(datetime.now().year)
    
    date_str = year + month + day
    
    df.to_parquet(f'{data_dir}/{name}_{date_str}.parquet', index=False)
    logger.info(f'{name}.parquet saved.')
    return


def clean_text(text: str) -> str:
    """Helper function to clean and normalize tweets text"""
    logger.debug('Cleaning text...')
    
    text = ' '.join(text.split())
    
    # removing urls
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    # removing mentions
    text = re.sub("@[A-Za-z0-9_]+","", text)
    # removing hashtags
    text = re.sub("#[A-Za-z0-9_]+","", text)
    # removing excess spaces
    text = re.sub('  ', '', text)
    # normalize text
    text = text.lower()
    
    return text


def save_to_text(df: pd.DataFrame, col: str, out_dir: str, name: str) -> None:
    """Helper function to convert dataframe to txt file"""
    all_text = ' '.join(df[col])
    
    with open(f'{out_dir}/{name}.txt', 'w') as f:
        f.write(all_text)
    
    logger.info(f'Saved corpus to .txt file: {name}.txt')
    
    return