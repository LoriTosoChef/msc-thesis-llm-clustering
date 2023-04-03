from typing import List, Optional

import os
import sys
import logging
import pandas as pd

from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub

logger = logging.getLogger(__name__)


class Bloom:
    def __init__(self, hf_api: str, temp: float, max_length: int = 256) -> None:
        self.model_name = 'bigscience/bloom'
        self.hf_api = hf_api
        self.temp = temp
        self.max_length = max_length
        
        self.model = HuggingFaceHub(huggingfacehub_api_token=self.hf_api,
                                    repo_id=self.model_name,
                                    model_kwargs={'temperature': self.temp,
                                                  'max_length': self.max_length})
        
        logger.info(f'Initializing BLOOM model - Temp: {self.temp} - Max Length: {self.max_length}')
    
    
    def init_prompt(self, template: str, input_vars: List[str]):
        self.input_vars = input_vars
        self.prompt = PromptTemplate(template=template, input_variables=self.input_vars)
        
        logger.info(f'Injecting Variables: {self.input_vars}')
        
        return self.prompt
    
    
    def run(self, inject_obj: Optional[str]) -> str:
        llm = LLMChain(prompt=self.prompt, llm=self.model)
        try:
            logger.info(f'Running Text Generation')
            out = llm.run(inject_obj)
        except Exception as e:
            logger.warning(e)
            return ''
        
        return out
