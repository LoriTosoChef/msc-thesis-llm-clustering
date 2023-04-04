from typing import List, Optional, Tuple

import logging

from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from langchain.llms import LlamaCpp

logger = logging.getLogger(__name__)


class Model:
    def __init__(self,
                 model_name: str,
                 hf_api: str,
                 local_model_path: str,
                 temp: float = 1e-10,
                 ctx_window: int = 2048,
                 max_tokens: int = 256,
                 n_threads: int = 6) -> None:
        
        self.model_name = model_name
        self.temp = temp
        self.ctx_window = ctx_window
        self.max_tokens = max_tokens
        self.hf_api = hf_api
        self.local_model_path = local_model_path
        self.n_threads = n_threads
        
        logger.info(f'\nInitializing {self.model_name.upper()} model  - Temp: {self.temp} - Context window: {self.ctx_window} - Max tokens: {self.max_tokens}')
        
        if 'gpt4all' in self.model_name:
            logger.debug(f'Loading GPT4All model from {self.local_model_path}')
            
            self.model = LlamaCpp(model_path=self.local_model_path,
                                  n_ctx=self.ctx_window,
                                  n_threads=self.n_threads,
                                  max_tokens=self.max_tokens,
                                  temperature=self.temp)
        
        elif 'llama' in self.model_name:
            logger.debug(f'Loading Llama model from {self.local_model_path}')
            
            self.model = LlamaCpp(model_path=self.local_model_path,
                                  n_ctx=self.ctx_window,
                                  n_threads=self.n_threads,
                                  max_tokens=self.max_tokens,
                                  temperature=self.temp)
        
        else:
            logger.debug(f'Loading from HuggingFace Hub')
            logger.debug('BLOOM model has a preset context window of 2048')
            self.model = HuggingFaceHub(huggingfacehub_api_token=self.hf_api,
                                        repo_id='bigscience/bloom',
                                        model_kwargs={'temperature': self.temp,
                                                      'max_length': self.max_tokens})
    
    
    def init_prompt(self, template: str, input_vars: List[str]) -> PromptTemplate:
        self.input_vars = input_vars
        self.prompt = PromptTemplate(template=template, input_variables=input_vars)
        
        logger.info(f'Injecting Variables: {self.input_vars}')
        
        return self.prompt.template
    
    
    def generate(self, inject_obj: Optional[str]) -> Tuple[str, list]:
        llm = LLMChain(prompt=self.prompt, llm=self.model)
        try:
            logger.debug(f'Running Text Generation\n')
            out = llm.run(inject_obj)
        except Exception as e:
            logger.warning(e)
            return '', []
        
        return llm, out