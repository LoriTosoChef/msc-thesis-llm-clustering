import logging

from nomic.gpt4all import GPT4All

logger = logging.getLogger(__name__)


class GPT4ALL:
    def __init__(self, temp: float = 1e-10, ctx_size: int = 2048, n_predict: int = 256, threads: int = 4) -> None:
        self.temp = temp
        self.ctx_size = ctx_size
        self.threads = threads
        self.n_predict = n_predict
        
        logger.info(f'\nInitializing GPT4All model - Temp: {self.temp} - Context window: {self.ctx_size} - Threads: {self.threads}')
        
        self.model = GPT4All(decoder_config={'temp': self.temp,
                                             'ctx_size': self.ctx_size,
                                             'threads': self.threads,
                                             'n_predict': self.n_predict})
        
    
    def load_model(self):
        logger.info('Loading model...')
        return self.model.open()
    
    
    def init_prompt(self, prompt: str) -> str:
        self.prompt = prompt
    

    def generate(self) -> str:
        try:
            logger.debug(f'Running Text Generation\n')
            out = self.model.prompt(self.prompt)
        except Exception as e:
            logger.warning(f'{e} - Returning empty string')
            return ''
        
        return out