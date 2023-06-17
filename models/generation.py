from typing import List, Optional, Tuple

import logging
import time

from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from langchain.llms import LlamaCpp

logger = logging.getLogger(__name__)


class Model:
    def __init__(self,
                 model_name: str,
                 hf_repo: str = 'bigscience/bloom',
                 hf_api: str = '',
                 openai_api: str = '',
                 openai_model: str = 'gpt-3.5-turbo',
                 local_model_path: str = '',
                 temp: float = 1e-10,
                 ctx_window: int = 2048,
                 max_tokens: int = 256,
                 n_threads: int = 6) -> None:
        
        self.model_name = model_name
        self.hf_repo = hf_repo
        self.temp = temp
        self.ctx_window = ctx_window
        self.max_tokens = max_tokens
        self.hf_api = hf_api
        self.local_model_path = local_model_path
        self.n_threads = n_threads
        self.openai_api = openai_api
        self.openai_model = openai_model
        
        logger.info(f'\nInitializing {self.model_name.upper()} model  - Temp: {self.temp} - Context window: {self.ctx_window} - Max tokens: {self.max_tokens}')
        
        if self.model_name == 'gpt4all':
            logger.debug(f'Loading GPT4All model from {self.local_model_path}')
            
            self.model = LlamaCpp(model_path=self.local_model_path,
                                  n_ctx=self.ctx_window,
                                  n_threads=self.n_threads,
                                  max_tokens=self.max_tokens,
                                  temperature=self.temp)
        
        elif self.model_name == 'llama':
            logger.debug(f'Loading Llama model from {self.local_model_path}')
            
            self.model = LlamaCpp(model_path=self.local_model_path,
                                  n_ctx=self.ctx_window,
                                  n_threads=self.n_threads,
                                  max_tokens=self.max_tokens,
                                  temperature=self.temp)
        
        elif self.model_name == 'openai':
            logger.debug(f'Loading from OpenAI')
            logger.debug(f'Using {self.openai_model}')
            if self.openai_model == 'text-davinci-003':
                from langchain.llms import OpenAI
                self.model = OpenAI(model_name=self.openai_model, temperature=temp, max_tokens=max_tokens)
        
            elif self.openai_model == 'gpt-3.5-turbo':
                from langchain.chat_models import ChatOpenAI
                self.model = ChatOpenAI(model_name=self.openai_model, temperature=temp, max_tokens=max_tokens)
        
        else:
            logger.debug(f'Loading from HuggingFace Hub')
            self.model = HuggingFaceHub(huggingfacehub_api_token=self.hf_api,
                                        repo_id=self.hf_repo,
                                        model_kwargs={'temperature': self.temp,
                                                      'max_length': self.max_tokens})
    
    
    def init_prompt(self, template: str, input_vars: List[str]) -> PromptTemplate:
        self.input_vars = input_vars
        self.prompt = PromptTemplate(template=template, input_variables=input_vars)
        
        logger.debug(f'Injecting Variables: {self.input_vars}')
        
        return self.prompt.template
    
    
    def generate(self, inject_obj: Optional[str]) -> Tuple[str, list]:
        llm = LLMChain(prompt=self.prompt, llm=self.model)
        try:
            logger.debug(f'Running Text Generation\n')
            out = llm.run(inject_obj)
        except Exception as e:
            logger.info(f'RATE LIMIT! Waiting for 5 minutes')
            logger.warning(e)
            time.sleep(300)
            return llm, ''
        
        return llm, out