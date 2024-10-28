import os
import re
import torch
import logging
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline

def load_llama_model():
    try:
        model = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        return pipe
    except Exception as e:
        logging.error("Error loading Llama model: %s", e)

def load_evaluation_model():
    ''' Load llama model from huggingface'''
    try:
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        return pipe
    except Exception as ex:
        logging.error("Error loading evaluation model: %s", ex)
        raise ex


def save_model(model_path):
    pipe= load_evaluation_model()
    pipe.save_pretrained(model_path)
    model_pipe = pipeline("text-classification", model = model_path)
    logging.info("model_loaded successfully")