import logging
import re
import pandas as pd

from datetime import datetime

from config import DATA_DIR

logger = logging.getLogger(__name__)

def save_to_parquet(data_dir: str, df: pd.DataFrame, name: str) -> None:
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    year = str(datetime.now().year)
    
    date_str = year + month + day
    
    df.to_parquet(f'{data_dir}/{name}_{date_str}.parquet', index=False)
    
    return


def clean_text(text: str) -> str:
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub("@[A-Za-z0-9_]+","", text)
    
    return text


def save_to_text(df: pd.DataFrame, col: str, out_dir: str, name: str) -> None:
    all_text = ' '.join(df[col])
    
    with open(f'{out_dir}/{name}.txt', 'w') as f:
        f.write(all_text)
    
    return