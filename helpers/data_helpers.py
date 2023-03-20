import logging
import re
import pandas as pd

from datetime import datetime

logger = logging.getLogger(__name__)


def clean_tweets_df(df: pd.DataFrame, text_col: str, n: int) -> pd.DataFrame:
    logger.debug('Dropping tweets with less than {n} words')
    logger.debug('Adding starter word "Tweet"')
    
    df['splits'] = df[text_col].str.split()
    df['len_splits'] = df['splits'].map(lambda x: len(x))
    
    df = df.loc[df['len_splits'] >= n].copy()
    
    df['full_text'] = '\nTWEET: ' + df[text_col]
    
    df = df.drop(columns=['splits', 'len_splits', text_col])
    df = df.reset_index(drop=True)
    
    return df


def save_to_parquet(data_dir: str, df: pd.DataFrame, name: str) -> None:
    logger.debug('Saving {name} to .parquet...')
    
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    year = str(datetime.now().year)
    
    date_str = year + month + day
    
    df.to_parquet(f'{data_dir}/{name}_{date_str}.parquet', index=False)
    
    return


def clean_text(text: str) -> str:
    logger.debug('Cleaning text...')
    
    text = ' '.join(text.split())
    
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub('  ', '', text)
    
    return text


def save_to_text(df: pd.DataFrame, col: str, out_dir: str, name: str) -> None:
    all_text = ' '.join(df[col])
    
    with open(f'{out_dir}/{name}.txt', 'w') as f:
        f.write(all_text)
    
    logger.info(f'Saved corpus to .txt file: {name}.txt')
    
    return