import os
import logging
from transformers import pipeline,AutoModel, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import subprocess
import psycopg2
import ast

def create_connection():
    ''' Create connection for the database'''
    try:
        connection = psycopg2.connect(database = "cogencis_db",
                                      user = "postgres",
                                      host= '3.239.71.203',
                                      password = "Cadmin123$",
                                      port = 5432)
        return connection
    except Exception as e:
        logging.error(e)

def get_required_file_data(file_id):
    ''' Get file_data for the uploaded file'''
    try:
        connection= create_connection()
        current_connection= connection.cursor()
        current_connection.execute("SELECT file_content from file_parsed_data where file_id=%s",(file_id,))
        file_data= current_connection.fetchall()
        current_connection.close()
        return ast.literal_eval(file_data[0][0])
    except Exception as e:
        logging.error(e)
        raise e
    
def get_title_list(combined_element_list):
    ''' Get index of all title contents'''
    try:
        title_list=[]
        for element in range(len(combined_element_list)):
            if 'title' in list(combined_element_list[element].keys()):
                title_list.append(element)
        return title_list

    except Exception as e:
        logging.error(e)
        raise e
    
def generate_combined_chunks(combined_chunk,title_list):
    ''' Generate combined chunks based on title content'''
    try:
        new_chunk_list=[]
        for x in range(len(title_list)-1):
            initial_chunk= combined_chunk[title_list[x]:title_list[x+1]]
            values_list=[list(x.values())[0] for x in initial_chunk]
            chunk_element= ".".join(values_list)
            new_chunk_list.append(chunk_element)
        final_chunk= combined_chunk[title_list[-1]:]
        final_value_list= [list(x.values())[0] for x in final_chunk]
        final_element= ".".join(final_value_list)
        new_chunk_list.append(final_element)
        return new_chunk_list
    except Exception as e:
        logging.error(e)
        raise e
    
def generate_exact_chunks(combined_chunk):
    ''' Check for token limit of 2000 and generate chunks'''
    try:
        title_list= get_title_list(combined_chunk)
        new_chunk_list= generate_combined_chunks(combined_chunk,title_list)
        actual_chunk_list=[]
        for chunk in new_chunk_list:
            if len(chunk)>2000:
                actual_chunk_list.append(chunk[:2000])
                actual_chunk_list.insert(new_chunk_list.index(chunk)+1,chunk[2000:])
            else:
                actual_chunk_list.append(chunk)
        return new_chunk_list
    except Exception as e:
        logging.error(e)
        raise e


def load_llama_model():
    try:
        logging.info("llama model loading")
        hf_token="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"
        subprocess.run(f'huggingface-cli login --token={hf_token}',shell=True)
        model_path= os.path.join("model")
        model_pipe = pipeline(task="text-generation", model = model_path,tokenizer= model_path)
        final_pipeline= HuggingFacePipeline(pipeline = model_pipe, model_kwargs = {'temperature':0})
        logging.info("model loaded successfully")
        return final_pipeline
    except Exception as e:
        logging.error(e)

def get_chunk_summary(llm,input_text):
    ''' Get summary of each chunk'''
    try:
        template = """
                    Write a concise summary of the following text delimited by triple backquotes.
                    Return your response in a paragraph in 1000 words.
                    ```{text}```
                    SUMMARY:
                 """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        text_summary= llm_chain.run(input_text)
        summary_parts= text_summary.split('SUMMARY:\n',1)
        chunk_summary=summary_parts[1].strip()
        return chunk_summary
    except Exception as e:
        logging.error(e)
        raise e       
  
def get_overall_document_summary(llm_model,chunk_list):
    ''' Get overall summary of the document'''
    try:
        summary=""
        for text_chunk in chunk_list:
            input_data= text_chunk+summary
            summary= get_chunk_summary(llm_model,input_data)
        return summary
    except Exception as e:
        logging.error(e)
        raise e


def main():
    titan_data= get_required_file_data(file_id=11)
    titan_actual_chunks= generate_exact_chunks(titan_data)
    print("actual_chunks_generated")
    logging.info(titan_actual_chunks)
    llm_model= load_llama_model()
    print("llm_model_loaded")
    overall_doc_summary= get_overall_document_summary(llm_model,titan_actual_chunks)
    print("overall summary generated")
    logging.info(overall_doc_summary)

main()
