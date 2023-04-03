import logging

from nomic.gpt4all import GPT4All

logger = logging.getLogger(__name__)


class GPT4ALL:
    def __init__(self, temp: float = 1e-10, ctx_size: int = 2048, threads: int = 4) -> None:
        self.temp = temp
        self.ctx_size = ctx_size
        self.threads = threads
        
        logger.info(f'Initializing GPT4All model - Temp: {self.temp} - Context window: {self.ctx_size} - Threads: {self.threads}')
        
        self.model = GPT4All(decoder_config={'temp': self.temp,
                                             'ctx_size': self.ctx_size,
                                             'threads': self.threads})
        
    
    def load_model(self):
        logger.info('Loading model...')
        return self.model.open()
    
    
    def init_prompt(self, prompt: str) -> str:
        self.prompt = prompt
        return self.prompt
    

    def generate(self) -> str:
        try:
            out = self.model.prompt(self.prompt)
        except Exception as e:
            logger.warning(f'{e} - Returning empty string')
            return ''
        
        return out