from typing import List, Optional

import logging

from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from langchain.llms import LlamaCpp

logger = logging.getLogger(__name__)


class Model:
    def __init__(self,
                 model_name: str,
                 hf_api: str,
                 gpt4all_path: str,
                 temp: float = 1e-10,
                 ctx_window: int = 2048,
                 max_tokens: int = 256) -> None:
        
        self.model_name = model_name
        self.temp = temp
        self.ctx_window = ctx_window
        self.max_tokens = max_tokens
        self.hf_api = hf_api
        self.gpt4all_path = gpt4all_path
        
        logger.info(f'\nInitializing model {self.model_name} - Temp: {self.temp} - Context window: {self.ctx_window} - Max tokens: {self.max_tokens}')
        
        if 'bloom' in self.model_name:
            logger.debug('Bloom model has a preset context window of 2048')
            self.model = HuggingFaceHub(huggingfacehub_api_token=self.hf_api,
                                        repo_id='bigscience/bloom',
                                        model_kwargs={'temperature': self.temp,
                                                      'max_length': self.max_tokens})
        elif 'gpt4all' in self.model_name:
            logger.debug(f'Loading GPT4All model from {self.gpt4all_path}')
            
            self.model = LlamaCpp(model_path=self.gpt4all_path)
    
    
    def init_prompt(self, template: str, input_vars: List[str]) -> PromptTemplate:
        self.input_vars = input_vars
        self.prompt = PromptTemplate(template=template, input_variables=input_vars)
        
        logger.info(f'Injecting Variables: {self.input_vars}')
        
        return self.prompt
    
    
    def generate(self, inject_obj: Optional[str]) -> str:
        llm = LLMChain(prompt=self.prompt, llm=self.model)
        try:
            logger.debug(f'Running Text Generation\n')
            out = llm.run(inject_obj)
        except Exception as e:
            logger.warning(e)
            return ''
        
        return llm, out