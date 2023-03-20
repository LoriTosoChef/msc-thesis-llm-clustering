import logging
import pandas as pd

from datetime import datetime

from config import DATA_DIR

logger = logging.getLogger(__name__)

def save_to_parquet(data_dir: str, df: pd.DataFrame, name: str) -> None:
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    year = str(datetime.now().year)
    
    date_str = year + '_' + month + '_' + day
    
    df.to_parquet(f'{data_dir}/{name}_{date_str}.parquet', index=False)
    
    return