import logging
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from models.generation import Model
from helpers.data_helpers import save_to_parquet

logger = logging.getLogger(__name__)


def generation_loop(model: Model,
                    model_col: str,
                    n: int,
                    tweets: pd.DataFrame,
                    fast_cool: int,
                    slow_cool: int,
                    out_dir: str,
                    out_name: str) -> None:
    """Helper function which handles the main generation loop of the models"""
    logger.info(f'Starting {model.model_name.upper()} generation...')
    # init output list
    outs = [' ' for _ in range(n)]
    for i, tweet in enumerate(tqdm(tweets['full_text'])):
        # generate
        llm, out = model.generate(inject_obj=tweet)
        # insert output
        outs[i] = out
        time.sleep(fast_cool)
        if (i+1) % 50 == 0:
            logger.info(f'Step: {i+1} - Saving checkpoint and cooldown for {slow_cool/60}m...')
            tweets[model_col] = np.array(outs)
            if not (i+1) == n:
                # if on last iteration (and multiple of 50) skip this and save only outside loop
                save_to_parquet(data_dir=out_dir, df=tweets, name=out_name)
            
    save_to_parquet(data_dir=out_dir, df=tweets, name=out_name)
    return tweets