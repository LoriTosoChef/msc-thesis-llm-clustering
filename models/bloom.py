from typing import List, Optional

import logging

from transformers import BloomTokenizerFast
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub

logger = logging.getLogger(__name__)


class Bloom:
    def __init__(self, hf_api: str, temp: float = 1e-10, max_length: int = 256, max_input_tokens: int = 2048) -> None:
        self.model_name = 'bigscience/bloom'
        self.hf_api = hf_api
        self.temp = temp
        self.max_length = max_length
        self.max_input_tokens = max_input_tokens
        
        logger.info(f'\nInitializing BLOOM model - Temp: {self.temp} - Context window: {self.max_input_tokens} - Max Length: {self.max_length}')
        
        self.model = HuggingFaceHub(huggingfacehub_api_token=self.hf_api,
                                    repo_id=self.model_name,
                                    model_kwargs={'temperature': self.temp,
                                                  'max_length': self.max_length})
    
    
    def init_prompt(self, template: str, input_vars: List[str]) -> PromptTemplate:
        self.input_vars = input_vars
        self.prompt = PromptTemplate(template=template, input_variables=self.input_vars)
        
        logger.info(f'Injecting Variables: {self.input_vars}')
        
        return self.prompt
    
    
    def count_prompt_tokens(self) -> int:
        try:
            logger.info('Initializing tokenizer')
            tokens = self.model.get_num_tokens(self.prompt.template)
        except Exception as e:
            logger.warning(f'{e}')
            return -1
        
        if tokens >= self.max_input_tokens:
            logger.warning(f'Returning -1, exceeded input tokens limit of {self.max_input_tokens} - Tokens: {len(tokens)}')
            return -1
        
        logger.info(f'Prompt tokens len: {tokens}')
        return tokens
    
    
    def generate(self, inject_obj: Optional[str]) -> str:
        llm = LLMChain(prompt=self.prompt, llm=self.model)
        try:
            logger.debug(f'Running Text Generation\n')
            out = llm.run(inject_obj)
        except Exception as e:
            logger.warning(e)
            return ''
        
        return llm, out
