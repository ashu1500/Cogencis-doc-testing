import os
import logging
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import subprocess
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from accelerate import Accelerator
import textwrap
import re
import datetime
import torch
import numpy as np
from accelerate import Accelerator
import concurrent.futures

accelerator = Accelerator()


def load_llama_model():
    ''' Load llama model from the local folder'''
    try:
        logging.info("llama model loading")
        hf_token="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"
        subprocess.run(f'huggingface-cli login --token={hf_token}',shell=True)
        model_path= os.path.join("model")
        model_pipe = pipeline(task="text-generation", model = model_path,tokenizer= model_path,device_map="auto")
        model_pipe= accelerator.prepare(model_pipe)
        final_pipeline= HuggingFacePipeline(pipeline = model_pipe, model_kwargs = {'temperature':0})
        logging.info("model loaded successfully")
        return final_pipeline
    except Exception as e:
        logging.error(e)
        raise e
    
llm_model= load_llama_model()

nl_prompt = """
You are a conversational bot. Given the user's question, the corresponding PostgreSQL query result, and the question itself, respond in a brief, natural language sentence using only the result. Only return the final answer, without any additional explanation or context.
[Examples are]
Question: What was the total amount spent in 2024?
PostgreSQL Result: "[(123456.7890123456,)]"
Return the answer in natural language (in English):
 
"The total amount spent in 2024 was $123456.7890123456"
 
Question: What was the highest purchase we made in 2024?
PostgreSQL Result: "[(100000.0000000000,)]"
Return the answer in natural language (in English):
 
"The highest purchase we made in 2024 was $100000.0000000000"
 
Question: What was the lowest purchase we made in 2024?
PostgreSQL Result: "[(10.000000000000000,)]"
Return the answer in natural language (in English):
 
"The lowest purchase we made in 2024 was $10.000000000000000"
 
[Answer ]
 
Question: What was the average purchase we made in 2024?
PostgreSQL Result: "[(6766.391201716735,)]"
Return the answer in natural language (in English):
 
"""
prompt = PromptTemplate(template=nl_prompt)
print(datetime.datetime.now())
result = llm_model.generate(prompt)
print(datetime.datetime.now())
print(result)
