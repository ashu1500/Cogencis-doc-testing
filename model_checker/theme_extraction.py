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

# THEME EXTRACTION
def theme_extraction_per_chunk(chunk_text, llm):
    ''' Extract themes for each chunk'''
    try:
        template = """<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        <</SYS>>
        Generate only 2 most important key headers relevant for financial information in maximum 3-4 words from the given text.Please do not include any explaination for the key headers.
        text: {text}
        key headers:
        """

        prompt = PromptTemplate(template=template, input_variables=["text"])
        result = llm.generate([prompt.format(text=chunk_text)])
        return result
    except Exception as e:
        logging.error(e)
        raise e

def extract_headers_from_themes(output_text):
    ''' Get headers list for themes'''
    try:
        start_index = output_text.find("key headers:")
        themes_section = output_text[start_index:]
        themes_lines = themes_section.split("\n")
        themes_lines = [line.strip() for line in themes_lines[1:] if line.strip()]
        headers_list = []
        for theme_line in themes_lines:
            if theme_line.strip().startswith(tuple(f"{i}." for i in range(1, 11))):
                if ":" in theme_line:
                    header = theme_line.split(":")[1].strip()
                    headers_list.append(header)
                else:
                    header = theme_line.split(".")[1].strip()
                    headers_list.append(header)

        return headers_list
    except Exception as e:
        logging.error(e)
        raise e

def extract_top_themes(theme_text,llm):
    ''' Identify top 15 themes from the given list'''
    try:
        template = """<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        <</SYS>>
        Identify 15 most important points relevant for financial information without any explanation and repetition from the given text below.
        text: {text}
        Important Points:
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        result = llm.generate([prompt.format(text=theme_text)])
        return result
    except Exception as e:
        logging.error(e)
        raise e

def extract_headers_from_important_points(output_text):
    ''' Get headers list for themes'''
    try:
        start_index = output_text.find("Important Points:")
        themes_section = output_text[start_index:]
        themes_lines = themes_section.split("\n")
        themes_lines = [line.strip() for line in themes_lines[1:] if line.strip()]
        headers_list = []
        for theme_line in themes_lines:
            if theme_line.strip().startswith(tuple(f"{i}." for i in range(1, 16))):
                if ":" in theme_line:
                    header = theme_line.split(":")[1].strip()
                    headers_list.append(header)
                else:
                    header = theme_line.split(".")[1].strip()
                    headers_list.append(header)

        return headers_list
    except Exception as e:
        logging.error(e)
        raise e

def get_final_transcript_themes(llm,input_list):
    '''Get final themes for the transcript document'''
    try:
        chunk_headers_list=[]
        all_chunk_header=[]
        actual_chunk_headers=[]
        for items in input_list:
            print("Theme generation")
            chunk_txt= theme_extraction_per_chunk(items,llm)
            chunk_header= extract_headers_from_themes(chunk_txt.generations[0][0].text)
            chunk_headers_list.append(chunk_header)
        for header in chunk_headers_list:
            all_chunk_header+=header
        print("All themes generated")
        ls=[actual_chunk_headers.append(x) for x in all_chunk_header if x not in actual_chunk_headers]
        top_themes= extract_top_themes(actual_chunk_headers,llm)
        generated_themes=extract_headers_from_important_points(top_themes.generations[0][0].text)
        fixed_themes=["Financial performance","Merger and acquisition","Risks and challenges","Market trends and outlook","Competitive positioning","ESG"]
        combined_themes= set(fixed_themes+generated_themes)
        final_themes= set(list(map(lambda x: str(x).title(), combined_themes)))
        return final_themes
        
    except Exception as e:
        logging.error(e)
        raise e
    

#OVERALL DOCUMENT SUMMARY
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
        llm_chain = prompt | llm
        text_summary= llm_chain.invoke(input_text)
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
            print("Chunk summary started")
            input_data= text_chunk+summary
            summary= get_chunk_summary(llm_model,input_data)
            print("Chunk summary generated")
        return summary
    except Exception as e:
        logging.error(e)
        raise e


#CHUNKS FILTERING

def generate_embeddings(e5_model,chunk_text):
    ''' Generate embeddings for the document chunks'''
    try:
        chunk_embeddings= e5_model.encode(chunk_text, normalize_embeddings=True)
        return chunk_embeddings
    except Exception as e:
        logging.error(e)
        raise e

def get_relevant_chunks_per_theme(e5_model,theme,chunk_embedding_dict):
    ''' Get relevant chunks for a theme'''
    try:
        relevant_chunk_list=[]
        theme_embedding= generate_embeddings(e5_model,theme)
        for chunk_text,chunk_embedding in chunk_embedding_dict.items():
            if cos_sim(theme_embedding,chunk_embedding).item()>0.78:
                relevant_chunk_list.append(chunk_text)
        return relevant_chunk_list
    except Exception as e:
        logging.error(e)
        raise e


def filter_relevant_chunks(e5_model,themes_list,chunk_embeddings_dict):
    ''' Get relevant chunks for each theme in the document'''
    try:
        required_chunk_data={}
        for theme in themes_list:
            required_chunk_data[theme]= get_relevant_chunks_per_theme(e5_model,theme,chunk_embeddings_dict)
        return required_chunk_data
    except Exception as e:
        logging.error(e)
        raise e


#THEME-BASED SUMMARIZATION

def extract_keywords_section(text):
    """Post processing to extract keywords section from the text."""
    try:
        start_marker = "Extracted Keywords:\n"
        keyword_prefix = r"\d+\. "
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.find("\n\n", start_index)
        keywords_section = text[start_index:end_index]
        keywords_list = [keyword.strip() for keyword in keywords_section.split("\n")]
        cleaned_list = [re.sub(keyword_prefix, "", keyword).strip() for keyword in keywords_list]
        return cleaned_list
    except ValueError as e:
        logging.warning("Keywords section not found in the text: %s", e)
        raise e
    except Exception as e:
        logging.error("Unexpected error while extracting keywords: %s", e)
        raise e
    
def keywords_theme_extraction(theme, text, llm):
    """ Extract keywords from the text based on the theme."""
    try:
        template = """
        As an AI assistant specializing in thematic analysis and keyword extraction, your task is to identify and list the most significant keywords from the given text that are highly relevant to the specified theme.
        Focus on extracting keywords that capture the essence of the theme and its representation in the text.Identify keywords that are directly related to the theme.
        Include industry-specific terms, jargon, or technical vocabulary relevant to the theme.
        Extract keywords that represent key concepts, ideas, or trends within the theme.
        Consider both explicit mentions and implicit references to the theme.
        Prioritize keywords that would be most valuable for a financial equity analyst.Extract proper nouns (e.g., company names, products) that are significant to the theme.
        Theme: {theme}
        Context:
        {text}

        Extracted Keywords:
        """
        prompt = PromptTemplate(template=template, input_variables=["theme", "text"])
        result = llm.generate([prompt.format(theme=theme, text=text)])
        final = extract_keywords_section(result.generations[0][0].text)
        return final
    except Exception as e:
        logging.error("Error extracting keywords: %s", e)
        raise e

def extract_summary_section_perchunk(text):
    """Post processing to extract summary section from the text."""
    try:
        keyword = "SUMMARY:"
        keyword_pos = text.find(keyword)
        if keyword_pos != -1:
            summary = text[keyword_pos + len(keyword):].strip()
            return summary
        else:
            logging.warning("Keyword 'SUMMARY' not found in the text.")
            return None
    except Exception as e:
        logging.error("Unexpected error while extracting summary section per chunk: %s", e)
        raise e
    
def summary_generation_perchunk(keyword_list, text, llm):
    """Generate summary for each chunk based on the keywords."""
    try:
        template = """
            Analyze the following text enclosed in curly brackets based on the given keywords enclosed in brackets.
            Generate a summary that includes both factual and inferential points, building a coherent narrative around the theme.
            Your summary should consist of exactly 5 bullet points, each point having at least 20 words long.Include a mix of direct observations and inferences drawn from the text.
            Build a story that flows logically from one point to the next.Prioritize information relevant to a financial equity analyst.Avoid using question formats or explicit headers.
            Please don't generate summary on the asked questions.
            {text}
            [Keywords: {keywords}]
            SUMMARY:
            """
        prompt = PromptTemplate(template=template, input_variables=["text", "keywords"])
        result = llm.generate([prompt.format(text=text, keywords=keyword_list)])
        final = extract_summary_section_perchunk(result.generations[0][0].text)
        return final
    except Exception as e:
        logging.error("Error generating summary per chunk: %s", e)
        raise e

def extract_summary_section(text):
    """Extract the final summary from the text."""
    keyword = "FINAL SUMMARY:"
    try:
        keyword_pos = text.find(keyword)
        if keyword_pos != -1:
            summary = text[keyword_pos + len(keyword):].strip()
            points = re.findall(r'\d+\.\s.*', summary)
            bullet_points = '\n'.join([f'• {point.split(" ", 1)[1]}' for point in points])
            return bullet_points
        else:
            logging.warning("Keyword 'FINAL SUMMARY' not found in the text.")
            return None
    except Exception as e:
        logging.error("Unexpected error while extracting summary section: %s", e)
        raise e
    
def get_final_summary(text, llm):
    """Generate the final summary"""
    try:
        template = """
        You are an AI assistant where you analyze the following text enclosed in curly brackets and generate a summary.
        Your summary should  consist of exactly 15 bullet points, each at least 20 words long. Blend factual information with insightful inferences.
        Ensure a logical flow between points, telling a story about the theme.Prioritize information relevant to a financial equity analyst.
        Avoid repetition and ensure each point adds new value to the narrative. Remove any category labels, focusing solely on the content.
        {text}
        FINAL SUMMARY:
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        result = llm.generate([prompt.format(text=text)])
        final = extract_summary_section(result.generations[0][0].text)
        return final
    except Exception as e:
        logging.error("Error generating final summary: %s", e)
        raise e

def remove_unwanted_headers(text):
    """Remove numbered headers and generate as bullet points"""
    try:
        lines = text.strip().split("\n")
        processed_lines = []
        for line in lines:
            line= line.strip()
            if not line:
                continue
            line = re.sub(r'\d+\. ', '\n• ', line).strip()
            if line.startswith("•"):
                colon_pos = line.find(":")
                if colon_pos != -1:
                    processed_line = "• " + line[colon_pos + 1:].strip()
                else:
                    processed_line = line.strip()
                    processed_lines.append(processed_line)
    
            else:
                processed_lines.append(line.strip())
        processed_text = "\n".join(processed_lines)
        final_processed_text= re.sub(r'\n\n', '\n', processed_text)
        return final_processed_text
    except Exception as e:
        print("Error removing headers: %s", e)
        raise e
    

def split_summary_into_chunks(summary, max_chunk_size):
    ''' Split the generated summary into equal parts'''
    try:
        summary_lines = summary.split('\n')
        lines= [x.strip() for x in summary_lines]
        chunks = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
    
        if current_chunk:
            chunks.append(current_chunk)
    
        return chunks
    
    except Exception as ex:
        print(ex)

def generate_chunk_summary(theme,chunk_text):
    ''' Generate final chunk summary'''
    try:
        keywords_list= keywords_theme_extraction(theme,chunk_text,llm_model)
        print("keywords generated")
        print(datetime.datetime.now())
        chunk_summary = summary_generation_perchunk(keywords_list, chunk_text, llm_model)
        print("Chunk theme summary generated")
        print(datetime.datetime.now())
        return chunk_summary

    except Exception as ex:
        logging.error(ex)

def process_files_in_parallel(chunk_content,theme, max_workers=2):
    results = ""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(generate_chunk_summary,theme, chunk_data): chunk_data for chunk_data in chunk_content}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results+= result
            except Exception as exc:
                results.append(f"{file_path} generated an exception: {exc}")
    return results

    
def generate_theme_summary(theme, chunk_data):
    ''' Generate summary for a theme'''
    try:
        combined_summary= ""
        result= process_files_in_parallel(chunk_data,theme)
        combined_summary+=result
        summary_list= split_summary_into_chunks(combined_summary,12000)
        output_summary=""
        for summary in summary_list:
            generated_summary= get_final_summary(summary,llm_model)
            output_summary+=generated_summary
        if len(output_summary.strip().split("\n"))>10:
            concised_summary= get_final_summary(output_summary,llm_model)
            final_summary= remove_unwanted_headers(concised_summary)
        else:
            final_summary= remove_unwanted_headers(output_summary)
        return final_summary
    except Exception as e:
        logging.error(e)
        raise e
    

def get_document_theme_summary(chunk_dictionary):
    '''Get theme-based summary of document'''
    try:
        theme_based_summary={}
        for theme,chunk in chunk_dictionary.items():
            if chunk:
                print("Theme summary started")
                theme_based_summary[theme]= generate_theme_summary(theme,chunk)
                print("Theme summary generated")
            else:
                continue
        final_theme_based_summary = {k: v for k, v in theme_based_summary.items() if v.strip() not in (None, '','•')}
        return final_theme_based_summary
    except Exception as e:
        logging.error(e)
        raise e


def remove_similar_summary_points(embedding_model,theme_summary):
    ''' Check similarity between summary points'''
    try:
        print("Removing similar summary points")
        indices_to_remove=set()
        summary_points= theme_summary.strip().split("\n")
        print("Summary length: ",len(summary_points))
        summary_embeddings= [generate_embeddings(embedding_model,summary) for summary in summary_points]
        for i in range(len(summary_embeddings)):
            for j in range(i+1,len(summary_embeddings)):
                if (cos_sim(summary_embeddings[i],summary_embeddings[j]).item())>0.89:
                  indices_to_remove.add(j)
        filtered_summary_points = [point for idx, point in enumerate(summary_points) if idx not in indices_to_remove]
        print("Summary length: ",len(set(filtered_summary_points)))
        final_theme_summary= "\n".join(set(filtered_summary_points))
        return final_theme_summary
    except Exception as ex:
        print(ex)
        raise ex


def compare_two_themes(embedding_model,theme1_summary,theme2_summary):
    ''' Check similarity between two themes'''
    try:
        print("Comparing two themes")
        similar_pairs=[]
        theme1_summary_points= theme1_summary.strip().split("\n")
        theme2_summary_points= theme2_summary.strip().split("\n")
        theme1_embeddings=[generate_embeddings(embedding_model,summary_point) for summary_point in theme1_summary_points]
        theme2_embeddings=[generate_embeddings(embedding_model,summary_point) for summary_point in theme2_summary_points]
        for i in range(len(theme1_embeddings)):
            for j in range(len(theme2_embeddings)):
                if (cos_sim(theme1_embeddings[i],theme2_embeddings[j])).item()>0.9:
                    similar_pairs.append((theme1_summary[i],theme2_summary[j]))
        if len(similar_pairs)>3:
            return True
        else:
            return False
    except Exception as ex:
        print(ex)
        raise ex


def check_similar_theme_summaries(embedding_model,theme_based_summary):
    ''' Get final theme based summaries'''
    try:
        themes_summary_list= list(theme_based_summary.values())
        themes_list= list(theme_based_summary.keys())
        for x in range (len(theme_based_summary)):
            for y in range(x+1,len(theme_based_summary)):
                if compare_two_themes(embedding_model,themes_summary_list[x],themes_summary_list[y]):
                    theme_based_summary[themes_list[y]]= " "
        final_theme_based_summary = {k: v for k, v in theme_based_summary.items() if v.strip() not in (None, '','•')}
        return final_theme_based_summary

    except Exception as ex:
        print(ex)
        raise ex

def get_refined_document_summary(chunk_dictionary,embedding_model):
    ''' Apply cosine similarity to remove similar data'''
    try:
        # final_doc_summary={}
        document_summary= get_document_theme_summary(chunk_dictionary)
        print("Complete document summary generated")
        # refined_summary= check_similar_theme_summaries(embedding_model,document_summary)
        # print("Refined summary generated")
        # for theme,summary in refined_summary.items():
        #     final_doc_summary[theme]= remove_similar_summary_points(embedding_model,summary)
        
        return document_summary
    except Exception as ex:
        print(ex)
        raise ex



def get_final_output(chunk_data):
    print(datetime.datetime.now())
    np.random.seed(42)
    transcript_themes= get_final_transcript_themes(llm_model,chunk_data)
    print("all themes generated")
    print(datetime.datetime.now())
    print(transcript_themes)
    # overall_doc_summary= get_overall_document_summary(llm_model,chunk_data)
    print("Overall summary generated")
    # print(overall_doc_summary)
    print(datetime.datetime.now())
    # transcript_themes=["Financial Performance","Revenue growth","Net addition of talent","Partnership expansion"]
    e5_embedding_model = SentenceTransformer('intfloat/e5-large')
    chunk_embedding_pair={}
    for chunk_text in chunk_data:
        chunk_embedding= generate_embeddings(e5_embedding_model,chunk_text)
        chunk_embedding_pair[chunk_text]= chunk_embedding
    relevant_chunks_dict= filter_relevant_chunks(e5_embedding_model,transcript_themes,chunk_embedding_pair)
    
    theme_based_summary= get_refined_document_summary(relevant_chunks_dict,e5_embedding_model)
    print("Final theme based summary generated")
    # print(theme_based_summary)
    return theme_based_summary


def main():
    hdfc_chunks=[
        'We see opportunities in the marketplace in the current environment supported by dynamic fiscal and monetary policy. Activity indicators released during April to June quarter indicates that economic activity continues to hold up well despite global risk. GST collections, manufacturing PMI, IIP, credit, rail freight, services PMI, etc., etc., show robust opportunities in the economy. The RBI raised the policy rate by 90 basis points in the quarter taking the repo rate to 4.9. The Monetary Policy Committee also voted to remain focused on withdrawal of accommodation in a calibrated fashion to ensure inflation remains within the RBI’s upper bank while supporting growth. Accordingly, we have responded with appropriate lending rate increases. Now let us talk about the five themes at a high level now. On the distribution expansion that is the first thing, we added 36 branches during the quarter and 250 more are in various stages of readiness to be rolled out. We have 15,618 business correspondences, an increase of 277 over prior quarter. Gold loans are now processed at just over 2000 branches as against 1340 branches in the prior quarter. It is well on the way to be a Page 2 of 22 July 16, 2022 product offering in most of our branches. Payment acceptance points have grown to 3.2 million a year-on-year growth of 42%. Wealth management is now offered in 357 locations through hub and spoke model, we have expanded to 141 new locations in the quarter. This is in accordance with our plan to take this to deeper geographies in over 900 locations in the current financial year. In commercial and global banking, SME is now offered in 640 districts in our drive to expand the SME market share. Next, let us talk about a few comments on the customer franchise building. During the quarter, we added 10900 plus people and 29,000 people over the past 12 months. Our people have acquired 2',
        'This is in accordance with our plan to take this to deeper geographies in over 900 locations in the current financial year. In commercial and global banking, SME is now offered in 640 districts in our drive to expand the SME market share. Next, let us talk about a few comments on the customer franchise building. During the quarter, we added 10900 plus people and 29,000 people over the past 12 months. Our people have acquired 2.6 million new liability relationships in the quarter, exhibiting a phenomenal growth of 59% over the same time last year and 10% over prior quarter. We have also acquired 1.9 lakh MSE accounts in the quarter. On cards, we have issued 1.2 million new cards during the quarter highest ever with a 47% growth over prior quarter. Total card base now stands at 17.6 million. Moving on to next, our focus on the granular deposit. Deposits at 16,04,000 Crores increased by approximately 46000 Crores in the quarter as against an addition of approximately 11000 Crores in last year June quarter. Deposits reflected a year-on-year growth of 19.2%. Retail deposits increased by approximately 50,000 Crores in the quarter up 19% year-on- year and 3.9% sequentially. CASA deposits recorded a strong growth of 20% year-on-year ending the quarter at 7,34,000 Crores with a CASA ratio at 45.8%. Term deposit grew by 18.5% year-on-year ending the quarter at 8,70,000 Crores. Next moving on to advances. Total advances were 13,95,000 Crores growth of sell downs we grew 22.5% year on year. Our retail advances growth continued during the quarter as well. Retail advances grew 21.7% year-on-year and 4.9% quarter-on-quarter. Excluding auto and also two- wheeler loans, which face supply chain disruptions during the quarter, the year-on-year retail growth excluding these two were 25%. Card spends have grown by 24% over prior quarter. Payment business advances, payment business loans have grown 27% over prior year and 4.4% over prior quarter. The bank has a market share of 22',
        'Our retail advances growth continued during the quarter as well. Retail advances grew 21.7% year-on-year and 4.9% quarter-on-quarter. Excluding auto and also two- wheeler loans, which face supply chain disruptions during the quarter, the year-on-year retail growth excluding these two were 25%. Card spends have grown by 24% over prior quarter. Payment business advances, payment business loans have grown 27% over prior year and 4.4% over prior quarter. The bank has a market share of 22.4% in cards, 48.9% in card receivables, 27.7% in card spends and 47% in merchant acquiring volumes. Page 3 of 22 July 16, 2022 Commercial and rural banking, which drives our MSME and PSL book continued its momentum with a year-on-year growth of 28.9%. In the wholesale segment with the rate dislocation, we let go assets aggregating to 40,000 to 50,000 Crores. Despite that, the book grew 15.7% year-on- year and lastly on technology and digital. As promised, the bank commenced digital launches to enable smooth customer experience. MyCards, which is a micro services architecture that is stateless and deployed on cloud making it highly scalable. This has emerged as a preferred service tool for our customers, it has simplified login and self- service features. We now have over 2 million registered card users, a growth of 1 million over prior quarter. We had 33 million customer service addressed digitally during the quarter on this platform. This micro services architecture design principle de-risks and removes clutter on our digital platform and enhance customer service. Xpress auto loans is an end-to- end digital service, which enables instant and hassle-free car loan disbursals for existing and new bank customers. 60% of our loan decisioning through this service are processed in less than five minutes with the disbursals taking less than 30 minutes. Within a month of launch, Xpress auto loans volumes have already reached more than 5% of our new car loan volume',
        'Xpress auto loans is an end-to- end digital service, which enables instant and hassle-free car loan disbursals for existing and new bank customers. 60% of our loan decisioning through this service are processed in less than five minutes with the disbursals taking less than 30 minutes. Within a month of launch, Xpress auto loans volumes have already reached more than 5% of our new car loan volume. HDFC Bank One our customer experience hub has been launched recently on multiple channels, email, social care, SMS, and WhatsApp and enhances our customer relationship management using AIML and conversational BOT enabling round-the-clock self-service capabilities with human interaction. We are continuously adding features to our SmartHub Vyapar app and see a significant increase in its adoption across our customer base. We now have more than 1.15 million customers since its launch on-boarded on this platform. In Q2 that is the current running quarter July to September we are poised to launch further digital initiatives such as PayZapp 2.0, customer on-boarding journeys across more products such as FDPL, Balance Transfer EMI, etc., implementing customer experience hub across additional service and sales channels such as phone banking and tele sales. For enhanced customer service and relationship management, we continue to work on developing applications for Q3 implementation for instance Payzapp revamping net banking, revamping corporate net banking, and launch of new mobile banking app in Q4. In Q1, we received a total of 231 million visits on our website averaging 28 plus million unique customers per month, which is a year-on- Page 4 of 22 July 16, 2022 year growth of about 20%. Business growth continued to gain momentum across diverse products and segments driven through relationship management and enhanced digital offering. Balance sheet remains resilient, average LCR for the quarter was at 108% and was at 120% as of June quarter. Capital adequacy ratio is at 18',
        'In Q1, we received a total of 231 million visits on our website averaging 28 plus million unique customers per month, which is a year-on- Page 4 of 22 July 16, 2022 year growth of about 20%. Business growth continued to gain momentum across diverse products and segments driven through relationship management and enhanced digital offering. Balance sheet remains resilient, average LCR for the quarter was at 108% and was at 120% as of June quarter. Capital adequacy ratio is at 18.1% with CET1 at 16.5% including profits for the current quarter. Let us start with net revenues. Core net revenues were at Rs.27181 Crores excluding trading and mark-to-market losses, which grew by 19.8% over prior year and 2.4% over prior quarter driven by advances growth of 22.5%, deposit growth of 19.2% and total balance sheet growth of 20.3%. Net interest income for the quarter at Rs. 19,481 Crores grew by 14.5% over prior year and 3.2% over prior quarter. The core net interest margin was at 4.0. Based on interest earning assets, the net interest margin was at 4.2%. Moving on to details of other income. First fees and commission income was at Rs.5360 Crores and grew by 38% over prior year and were lower 4.8% over prior quarter as a seasonally strong fourth quarter. Retail constitutes approximately 92% of fees. The fixed and derivatives income at Rs.1259 Crores was higher by 5% compared to prior year. Trading and mark-to-market losses were 1312 Crores primarily owing to spike in benchmark bond yields witnessed during the quarter. The mark-to-market losses come from our AFS, HFC and Government of India securities, corporate bonds, and pass-through certificates. Prior quarter was a negative 40% and prior year was a gain of 600 Crores. Other miscellaneous income of 1080 Crores includes recoveries from written of accounts and dividends from subsidiaries excluding trading on mark-to- market losses total other income at 7700 Crores grew by 35% over prior year',
        'The mark-to-market losses come from our AFS, HFC and Government of India securities, corporate bonds, and pass-through certificates. Prior quarter was a negative 40% and prior year was a gain of 600 Crores. Other miscellaneous income of 1080 Crores includes recoveries from written of accounts and dividends from subsidiaries excluding trading on mark-to- market losses total other income at 7700 Crores grew by 35% over prior year. Operating expenses for the quarter were at 10502 Crores, an increase of 28.7% over prior year due to a low base of prior year COVID wave two impacted quarter and increased by 3.4% over prior quarter. We added 725 branches and 2329 ATMs since last year taking the total network strength to 6378 branches, 18620 ATMs and 15294 business correspondence managed by common service centers. Core cost to income ratio for the quarter excluding trading and mark-to-market losses was at 38.6%. Page 5 of 22 July 16, 2022 Moving on to PPOP, our earnings trajectory improved with continued retail growth our core PPOP grew 14.7% year-on-year and 1.7% sequentially. Our pre-provision operating profit was at 15368. Coming to asset quality, the GNPA ratio was at 1.2% as compared to 1.4% prior year. Out of the 1.28%, about 18 basis points of standard thus the core GNPA ratio is 1.1. However, these are included by us in NPA as one of the other facilities of the borrower as a NPA, but we will talk about 1.28 we will have to anchor with that. As you have seen in the past several years, agricultural segment has a seasonal impact in June and December cycle. GNPA ratio excluding NPAs in agriculture segment and a one off was at 1.03%; prior year was at 1.26% and prior quarter was at 1.01%. Net NPA ratio was at 0.35% prior year was at 0.48% and proceeding quarter was at 0.32%. The slippage ratio for the current quarter is at 0',
        '28 we will have to anchor with that. As you have seen in the past several years, agricultural segment has a seasonal impact in June and December cycle. GNPA ratio excluding NPAs in agriculture segment and a one off was at 1.03%; prior year was at 1.26% and prior quarter was at 1.01%. Net NPA ratio was at 0.35% prior year was at 0.48% and proceeding quarter was at 0.32%. The slippage ratio for the current quarter is at 0.5%, which is 7200 Crores excluding the seasonal agri and one-off slippage, the slippage in the current quarter was approximately 38 basis points call it 0.4%. During the quarter recoveries and upgrades were approximately 3000 Crores of 22 basis points. Write- offs in the quarter were 2400 Crores are approximately 17 basis points. There were no sales of stressed or written-off accounts in the quarter. The check bounce rates across the products in June continues to remain lower than the pre-COVID levels for almost all of the retail products. The restructuring under the RBI resolution framework for COVID-19 as of June end stands at 76 basis points 10750 Crores. In addition, certain facilities of the same borrower which are not restructured is approximately 13 basis points are 1850 Crores that totals to 89 basis points. Provisions reported were around 3200 Crores as against 4800 Crores for the prior year and 3300 Crores during the prior quarter. The provision coverage ratio was at 73%, there were no technical write-offs, our head office and branch books are fully integrated. At the end of current quarter, contingent provisions and floating provisions remained close to prior quarter at 11,100 Crores, general provisions were 6500 Crores. Total provisions comprising specific floating, contingent and general provisions were about 170% of gross non- performing loans. This is in addition to the securities held as collateral in several of the cases. Floating contingent in general provisions were about 1.25% of gross advances as of June quarter end',
        'At the end of current quarter, contingent provisions and floating provisions remained close to prior quarter at 11,100 Crores, general provisions were 6500 Crores. Total provisions comprising specific floating, contingent and general provisions were about 170% of gross non- performing loans. This is in addition to the securities held as collateral in several of the cases. Floating contingent in general provisions were about 1.25% of gross advances as of June quarter end. Now coming to credit cost ratios, the total annualized credit cost for the quarter was at 91 basis points prior year was at 167 basis points, prior Page 6 of 22 July 16, 2022 quarter was at 96 basis points. Recoveries which are recorded as miscellaneous income amounts to 23 basis points of gross advances for the quarter as against 14 basis points in prior year and 26 basis points for prior quarter. Total credit cost ratio net of recoveries was at 68 basis points compared to 1.53% in prior year and 70 basis points in prior quarter. The reported PBT at 12180 Crores grew by 18% over prior year. Net profit after tax for the quarter at Rs.9196 Crores after factoring in the trading and mark-to-market losses of 1312 Crores in the quarter grew by 19% over prior year. That is after taking the charge for 1112 Crores grew by 19%. Now some highlights on HDBFS on an Ind-AS basis. HDBFS opened 29 branches in the quarter taking it to 1403 branches spread across more than 1000 cities, 1008 cities and towns. Branch addition continues to supplement the digital investments. Customer base grew to 9.8 million with 7.7% additions during the quarter and an increase of 35% over prior year. The uptick in disbursements in March quarter was sustained in the quarter ended June 2022 at 9000 Crores though disbursements in Q1 are traditionally lower as compared to March quarter. This disbursements reflect a growth of 130% year-on-year',
        'Branch addition continues to supplement the digital investments. Customer base grew to 9.8 million with 7.7% additions during the quarter and an increase of 35% over prior year. The uptick in disbursements in March quarter was sustained in the quarter ended June 2022 at 9000 Crores though disbursements in Q1 are traditionally lower as compared to March quarter. This disbursements reflect a growth of 130% year-on-year. The total loan book as on June end stood at 61814 Crores, secured loans comprising 76% of the total loan book. Net revenue for the quarter ended June 30, was at Rs.2194 Crores a growth of 13% over prior year and 2.4% sequentially. Cost to net income for the lending business was at 37%. Provisions and contingencies for the quarter were at 398 Crores as against 422 Crores for prior quarter and 870 Crores for quarter ended last year same time. Stage 3 as of June end stood at 4.95% after factoring in 1.18% impact of new RBI guidelines issued in November reflecting sustained healthy collections. The PCR secured and unsecured book stood at 48% and 92% respectively. Profit after tax for the quarter ended June was 441 Crores as against 89 Crores for last year same period. Earnings per share was Rs.5.58 and book value per share was at 125. The company remains well capitalized with a capital adequacy ratio of 20% and well positioned to sustain improvement in disbursements across segments. HSL, HDFC Securities Limited has a wide network of 216 branches across 147 cities and towns in the country. HSL has increased its overall client base to 3.99 million customers as of June end, an increase of 41% over prior year. The total reported revenue for the quarter was at 432 Crores as against 456 Crores in prior year. Net Page 7 of 22 July 16, 2022 profit after tax was at 189 Crores against 251 Crores of prior year. Earnings per share in the quarter was 119.5 and book value per share was at 1061',
        'HSL has increased its overall client base to 3.99 million customers as of June end, an increase of 41% over prior year. The total reported revenue for the quarter was at 432 Crores as against 456 Crores in prior year. Net Page 7 of 22 July 16, 2022 profit after tax was at 189 Crores against 251 Crores of prior year. Earnings per share in the quarter was 119.5 and book value per share was at 1061. In summary, over 152000 employees across the bank dedicated their tireless service to focus for customer engagement, product delivery and service providing highest standards of banking experience, which results in the quarter’s number of advances growth of 22%, deposits growth of 19%, core operating profit excluding the bond losses of 14.7%, delivering a consistent profit after tax growth of 19% after factoring in the bond losses of 7312 that I alluded to earlier. Again from a return on asset point of view 1.8% excluding the impact of the trading and mark-to-market it is slightly over 2% with an ROE of 17%. Earnings per share reported in the quarter is at Rs.16.6 book value per share increased in the quarter to 450.6. With that can I request Faizan to open up the line for questions, please. Moderator: Thank you very much. We will now begin the question-and-answer session. The first question is from the line of Mahrukh Adajania from Edelweiss. Please go ahead. Mahrukh Adajania: Hello Sir, my first question is on your CRB loans of course the Q-o-Q growth excluding agri has been good at 4% however we have been talking about doubling the booking in three years. So that would probably require a higher run rate of growth. So how do you see the outlook panning out for growth in CRB and also if you could throw some color on you said that you probably gave up some corporate loan growth in the commentary. So what was that about, that is my first question then I have two more. Srinivasan V: First let us talk about the CRB loans that you talked about',
        'So that would probably require a higher run rate of growth. So how do you see the outlook panning out for growth in CRB and also if you could throw some color on you said that you probably gave up some corporate loan growth in the commentary. So what was that about, that is my first question then I have two more. Srinivasan V: First let us talk about the CRB loans that you talked about. This year the loans had a robust growth of about 28%, 29% year-on-year in the quarter and we do have aggressive plans across various segments in CRB both on the MSME side as well as well as on the agri side on both sides where we have a significant growth. This growth I think we talked about it maybe a month ago in another forum that growth is predicated on one geographic expansion, we want to be present in more districts in the country to be able to capture the supply chain and the distribution chain flows. That is part of what we are trying to do to be present everywhere so that we capture all of Page 8 of 22 July 16, 2022 the chain, distribution chain, supply chain not just a part of it that we work with various other wholesale clients we are able to capture in wholesome not part. So that is part of what we are doing. The second aspect of that is also in terms of agri again its distribution expansion moving from about 1 lakh villages that we do today as a step we want to go through close to 2 lakh villages that is again part of how we want to operate and get to. There are enough opportunities we see that they are good and that can come only by where we put our salespeople, we put our relationship people in the local place where the customer is, that is part of the distribution',
        'The second aspect of that is also in terms of agri again its distribution expansion moving from about 1 lakh villages that we do today as a step we want to go through close to 2 lakh villages that is again part of how we want to operate and get to. There are enough opportunities we see that they are good and that can come only by where we put our salespeople, we put our relationship people in the local place where the customer is, that is part of the distribution. The second thing is the relationship management, which is in-addition to having a physical, we also want to have our relationship now because most of the CRB is about relationship management and we are expanding more, adding more people into that, so that we could get the right kind of a relationship to have that both from acquiring customers as well as broad basing the products that we could deliver to them, yes we are confident that segment is poised for growth and again we are not talking about it in an isolation, this is going to ride on the country’s macro growth that means we need the tailwind of the country growth also to be going up and with the MSME being almost a third of the GDP participation that is where we tend to we are focused on doing that and from a market penetration point of view again I think we told you how that growth is going to come from last time somewhere we talked which is we have only about 20% to 25% penetrated in the banking system itself. So the rest of them are outside of the banking system, they need to move in here and this is part of our both physical as well as the RM expansion strategy is to capture them and bring them into the banking system. On the wholesale loans bit something which I did not get what was the question on the wholesale. Mahrukh Adajania: No you said that 40 to 50 is or maybe I heard it wrong you said 40000 to 50000 Crores was given up because of competitive rates or something like that',
        'So the rest of them are outside of the banking system, they need to move in here and this is part of our both physical as well as the RM expansion strategy is to capture them and bring them into the banking system. On the wholesale loans bit something which I did not get what was the question on the wholesale. Mahrukh Adajania: No you said that 40 to 50 is or maybe I heard it wrong you said 40000 to 50000 Crores was given up because of competitive rates or something like that. Srinivasan V: Good point, yes, I did mentioned that and I specifically mentioned that I know that you will pick it up and ask, which is see there was a rate dislocation in the quarter sometime around starting May the rate started to move up there was a rate dislocation immediately after our bank and so also others started to move up on the rates and we did that and as we move up on the rate, there were some customers who were offered lower Page 9 of 22 July 16, 2022 rates by certain other market participants and we do not want to cut back on our rates to keep them. We said that is fine because we do have a relationship, we do continue to have relationship with those customers with 40000, 50000 payment and that we continue to have except that we did not endeavor by price to keep increasing those shares. So we said that is fine to let go, let somebody else can take it at a lower price than where we do and that is what I alluded to. Mahrukh Adajania: And Sir was that PSU banks or private bank. Srinivasan V: It is broadly across everywhere so like not going to the details, but it is across the banks. Mahrukh Adajania: And can you please quantify the slippage figure as in the absolute amount if you can. Srinivasan V: I think I gave the 7200 or something I did mentioned that is the 50 basis points or 0.5%. Mahrukh Adajania: And how much of that would be from restructured',
        'Mahrukh Adajania: And Sir was that PSU banks or private bank. Srinivasan V: It is broadly across everywhere so like not going to the details, but it is across the banks. Mahrukh Adajania: And can you please quantify the slippage figure as in the absolute amount if you can. Srinivasan V: I think I gave the 7200 or something I did mentioned that is the 50 basis points or 0.5%. Mahrukh Adajania: And how much of that would be from restructured. Srinivasan V: I did not give that, but I think I alluded to that the slippage amount has got it agri and at wholesale one off which contributed almost to little more than 10 basis points so net of that it was 0.4 of 38 basis points I alluded to, the some of them not the agri piece but the other piece is the part of the restructuring. Mahrukh Adajania: Sir and my last question is on this merger dispensation. So we did see a press release on RBI approving the merger and it said terms and conditions. So were there any dispensations and if not when would one hear about dispensations applied for and also any clarity on HDFC Life stake. Srinivasan V: Two things last one is about the conditions of the dispensation. The no objection from RBI is on our application and that the conditions I think we mentioned somewhere these conditions are for example I will give you some nature of some of those things how you can think about, when the merger happens the banking regulation shall apply across all the portfolios and all the business lines. So that is part of both that the kind of giving you Page 10 of 22 July 16, 2022 flavor of some of those conditions that is one and there are some entities that will merge and the licenses of those entities that will merge have to be surrendered and then intimated to RBI',
        'So that is part of both that the kind of giving you Page 10 of 22 July 16, 2022 flavor of some of those conditions that is one and there are some entities that will merge and the licenses of those entities that will merge have to be surrendered and then intimated to RBI. So that kind of those are some examples and then when we apply and get approvals from various other authorities we need to take those approvals to get back to the regulator with those approvals and when we go to shareholders whatever is the shareholder resolution and the approvals we get it back to the regulator. So you can see that these are, I will give you some flavor of how to think about this condition, but you alluded to what about the dispensation or the glide part of the portfolio and so that is not what it is, that is something separate and that is handled as an item different from the application per se and we continue to work with the regulators on that aspect. Mahrukh Adajania: Got it Sir. My last question is on EBLR repricing, so basically your reset for retail and corporate loans will be what three months, one month. Srinivasan V: Three months or six months, mostly I think it is three months. Mahrukh Adajania: Got it Sir, that was very, very helpful. Thank you so much. Moderator: Thank you. The next question is from the line of Hardik Shah from Goldman Sachs. Please go ahead. Hardik Shah: Hi! Sir, congratulations for a good quarter. My first question is on the MTM loss. Can you share some color on AFS mix modified duration and under what circumstances one can use the IFR. Srinivasan V: Thank you for bringing this up. See the AFS book broadly you can think about this at three components, broadly three components. One is the corporate bonds; the other is the participation certificate primarily in priority sector lending participation certificates and the third one is the Government of India’s security. These are the three broad components which are there',
        'Can you share some color on AFS mix modified duration and under what circumstances one can use the IFR. Srinivasan V: Thank you for bringing this up. See the AFS book broadly you can think about this at three components, broadly three components. One is the corporate bonds; the other is the participation certificate primarily in priority sector lending participation certificates and the third one is the Government of India’s security. These are the three broad components which are there. Most of these the other aspect that you asked about is the modified duration and how you think about it. See about for two years you can think about it is the tenure of the duration and that is the time it takes to pull this too far. So from that sense we expected in a couple of years we drift back over this time period. The other aspects of the investment fluctuation reserve and what it means to these things. The investment fluctuation Page 11 of 22 July 16, 2022 reserves an appropriation of profit to set some results up and we have investment fluctuation reserves which are slightly more than 2% and at the discretion of the bank at some point in time we can utilize this, but we have not chosen to utilize investment fluctuation reserves and because it is slightly more than 2% it has to be I think regulatory 2% so there is no point getting in and given that this pulls back to par in a couple of years’ time and we are quite not comfortable to pull down the reserves and use it right now. Hardik Shah: Got it thank you Sir. My second question is on the growth side. Growth on retail has been impressive, so what are your thoughts on its sustainability given the inflation concerns that you alluded to at the start of the call. Srinivasan V: Again another good point thank you. See the retail growth ever since we came back with a credit policy getting back to pre-COVID level. If you see over a period of two, three quarters having been quite good, the December quarter was close to 4',
        'Hardik Shah: Got it thank you Sir. My second question is on the growth side. Growth on retail has been impressive, so what are your thoughts on its sustainability given the inflation concerns that you alluded to at the start of the call. Srinivasan V: Again another good point thank you. See the retail growth ever since we came back with a credit policy getting back to pre-COVID level. If you see over a period of two, three quarters having been quite good, the December quarter was close to 4.5, 5, the March quarter was close to 5 similar rate and the June quarter is 5% sequential. So year-on-year it is now cross the 20% mark the year-on-year because of the base, because we kept going down and now we are starting to build up sequential momentum is there. Within the retail book if you look at the one that I called out for the vehicle segment has been hampered by various supply chain issues. Despite that it did grow well, we did have quite a good growth, but then if you put that to the side and give more time for that to grow the retail excluding that vehicle segment grew by almost 25% year-on-year so it is again a solid growth. Then the other aspect of how to think about the environment and the growth, we do see good amount of demand across most of the products from unsecured product to secured product to mortgage product to home loans and across everywhere we do see that including the gold loan and so on. I think we have published that list of various products and the growth rate, so you can see that it is balanced across. Card loans, let us talk about credit cards the last time they want to mention',
        'Then the other aspect of how to think about the environment and the growth, we do see good amount of demand across most of the products from unsecured product to secured product to mortgage product to home loans and across everywhere we do see that including the gold loan and so on. I think we have published that list of various products and the growth rate, so you can see that it is balanced across. Card loans, let us talk about credit cards the last time they want to mention. The card loans will have a very good spend into the 4% or so sequential spend increase again discretionary if you look at the discretionary spends has gone up even more and most this growth in the car spend is driven lot of by discretion and it is discretion of course you can take it as also seasonal in the summer months or holidays months a lot of travel, entertainment, hotels and so on and so forth they are all coming back to life and you are Page 12 of 22 July 16, 2022 seeing pick up, huge pick up on that. The second aspect of the spend is that is the spend translating into loans and which to some extent it is, but to a large extent it still needs to come more it is still not fully there from a loan growth point of view it will take some more time I think over five years what the prior quarter sequential is 4.4% is the sequential growth rate in payment terms. For it to pick up and go further, we will have to wait for our people to utilize their credit lines fully still the credit line utilization on cards is that I call it at 70% to 80% of the pre-COVID level. So a lot of credit lines utilization still left to go, and the liquidity in the hands of the customers is also there, these customers from a relationship point of view about five times our customers have for the 80000 Crores of payment business balances that we have five times that we have liabilities from the similar customer segments. So we do see that people have good amount of money and line utilization to happen',
        'So a lot of credit lines utilization still left to go, and the liquidity in the hands of the customers is also there, these customers from a relationship point of view about five times our customers have for the 80000 Crores of payment business balances that we have five times that we have liabilities from the similar customer segments. So we do see that people have good amount of money and line utilization to happen. So we expect that with the pickup for that is taking place right now we need to give some more time for that to do and similarly on the revolve rates you do not, but I am sure another person would be thinking about asking so I would allude the same thing. Their revolve rate pickup also has not happened yet, first spend their needs to happen which is happening now, two, three quarters we are seeing spend happening, spend translating into loans slightly picking up sequential 4.4 picking up then the next thing is that the line utilization happens and then comes the revolving to come with that. So we are a few quarters away to get there. Hardik Shah: Got it as a follow-up to that what are your thoughts on the sustainable revolve rate going forward in the industry. Srinivasan V: See as the economy starts to pick up and people spend which you are beginning to see discretionary spends you are seeing is happening once the discretionary spends happen you will see that the people will get back to the previous. See over a period of two years both either in our bank or in some other bank people who were call it for lack of some other chronic revolvers that means habitually revolving for more than six months, nine months out of the 12 months have come down because either they are having a bad score in the bureau or they are having a bad score with us and we have utilize their limits so we are not entitled on the limits are not given because we want to be cautious',
        'See over a period of two years both either in our bank or in some other bank people who were call it for lack of some other chronic revolvers that means habitually revolving for more than six months, nine months out of the 12 months have come down because either they are having a bad score in the bureau or they are having a bad score with us and we have utilize their limits so we are not entitled on the limits are not given because we want to be cautious. So we need to wait for the things to come back and then they will start to spend and the revolve, we are quite Page 13 of 22 confident that the customer base that we have and the type of spend that they do we will get back to what we have seen in pre-COVID from the spend habits and revolve kind of attitude on that. July 16, 2022 Hardik Shah: Got it. Last question on deposit rates, you have been taking the rates higher. So how should we think about this in the next few quarters as how much hike the bank would consider taking and how is the competitive intensity increasing on that front. Srinivasan V: The pricing which we are talking about more the time deposit pricing because the other cards of course nothing and the savings account is being stable. The time deposit we have slightly only increased over the last month to two months and I have taken it up all the way what has happened and the way we think about the pricing is there are two elements to it. One, customers, we are able to get to the right kind of a customer to have the deposits and what is the price sensitivity of the customer to get those volumes',
        'The time deposit we have slightly only increased over the last month to two months and I have taken it up all the way what has happened and the way we think about the pricing is there are two elements to it. One, customers, we are able to get to the right kind of a customer to have the deposits and what is the price sensitivity of the customer to get those volumes. So that is always a kind of what we do engaging with our frontline who in turn engages with the customer, but we get that intelligence and discussion in ALCO to say how we are able to get those volumes at kind of a price point that we can get and the second aspect of our determination of the price is also competitively pricing like competitively pricing means looking at certain other banks to see that we are relevant in the market and we do not want to be price leaders by pricing up anything, but at the same time we have to be competitive within certain range that is how these are a couple of considerations we do and we discuss within the ALCO as a team and decide how we want to pitch ourselves to the customer. Hardik Shah: Got it. Thank you for your time Sir and congratulations again for a good quarter. Moderator: Thank you. The next question is from the line of Kunal Shah from ICICI Securities. Please go ahead. Kunal Shah: Hi! Srini and team, thanks for taking the question. Firstly again just coming on with respect to the RBI’s approval, so any indication with respect to HDB Financial, so when we look at it in terms of the scheme of arrangement it says it has approved so would be hear further with respect Page 14 of 22 to HDB and HDFC Life or it is more or less there within the arrangements scheme. July 16, 2022 Srinivasan V: Good, thank you the other question I did not address. First the RBI approval is no objection to The Scheme of Amalgamation that has been filed. The Scheme of Amalgamation does not have a role for HDB there',
        'July 16, 2022 Srinivasan V: Good, thank you the other question I did not address. First the RBI approval is no objection to The Scheme of Amalgamation that has been filed. The Scheme of Amalgamation does not have a role for HDB there. HDB is a subsidiary, existing subsidiary of the bank and continues to be there and so the scheme of amalgamation does not have anything to do with HDB and so that is, if anything we need to do, the separate conversation is the separate process and so on. So it is not combined with this scheme, we find the scheme and the scheme does not have anything to do with HDB. HDFC Life is currently a subsidiary of HDFC Limited and it is envisaged that a merger that it will be a subsidiary of the bank. There are two things in this, one as RBI regulation a bank holding life insurance has to be 30% or below or 50% or above currently HDFC Life holding is about 47.8% or so that is a 2% plus percentage point increase that is required and that is part of another kind of a regulatory approval that we have sought that we can go to 50% plus and whatever the regulator finally tells us we will have to comply with that. So that is part of what we are waiting for, it is a continuous dialogue that happens to see how we can get to more than 50% either we get or HDFC Limited will get to 50% plus before consummation of the merger transaction. So that is on HDFC Life. Kunal Shah: But there are no timelines in terms of where can we expect, so the process is still on the communication is still on. Srinivasan V: That is correct. Kunal Shah: And secondly in terms of the overall PSL or maybe as we look at in terms of the build up towards the merger, so couple of points, one is in terms of the branch expansion we have been highlighting that 1500, 2000 odd branches could be added maybe the Q1 was not maybe we had not seen that much of a branch addition',
        'Kunal Shah: But there are no timelines in terms of where can we expect, so the process is still on the communication is still on. Srinivasan V: That is correct. Kunal Shah: And secondly in terms of the overall PSL or maybe as we look at in terms of the build up towards the merger, so couple of points, one is in terms of the branch expansion we have been highlighting that 1500, 2000 odd branches could be added maybe the Q1 was not maybe we had not seen that much of a branch addition. So when do we expect is it post like consummation of the merger do we see that run rate or we will start preparing for it from this fiscal and it will be more back ended and second related question is on the PSL build up. So should we say that whatever PSL certificates were bought in, in FY2022 and RIDF investments which have gone up from 9000 to 45000 Crores that was maybe with respect to Page 15 of 22 the earlier requirement and we will start building up further to meet up with the HDFC Limited’s merger how should one see that. July 16, 2022 Srinivasan V: Thank you again for that branch build up what you asked about. Yes, this quarter the branch build up was lower 36 or so, but we have about 250 branches in various stages of getting to be implemented. We are not going to wait for anything the branch build up is an organic process irrespective of any kind of an outcomes branch buildup is the right thing to do for the bank from a growth point of view that is where we are embarked on and we see opportunity. Branches that I think we talked about it again in the past branch has got two aspects to it, one you have a branch which develops the brand in the vicinity of where the branch is and draws in customers through brand attraction. The second thing is the branch is the congregation of our sales force, if we do not have branch you are going to have a sales office, you can call it that we would open x thousands of sales office',
        'Branches that I think we talked about it again in the past branch has got two aspects to it, one you have a branch which develops the brand in the vicinity of where the branch is and draws in customers through brand attraction. The second thing is the branch is the congregation of our sales force, if we do not have branch you are going to have a sales office, you can call it that we would open x thousands of sales office. So we rather open x thousands of sales because the kind of a travel that sales relationship managers need to do in their outreach to meet a customer or a prospective customer we want to keep it to 1 to 2 kilometers rather than to 4, 5 or 6 kilometers it gets in better productivity and gets in better influence to consummate that transaction right. That is part of what we have envisage everything. So the branch buildup will happen it is not waiting for anything, it is a question of a process to get that implemented, it is in progress to happen. So even in this financial year you will see some substantial branch accretion that happens. The second aspect that you touched upon is the PSL, and you touched upon the RIDF on PSL. See PSL there are several strategies to grow here, organic buildup of loans PSL eligible loans is the best method to do because it gives fantastic returns, it gives great returns going through our credit filters because we tried and tested credit filters it gives you the best returns that you can and the returns far more than the average of the bank. So we are quite enthused to do PSL organically to the extent it comes through our credit list',
        'See PSL there are several strategies to grow here, organic buildup of loans PSL eligible loans is the best method to do because it gives fantastic returns, it gives great returns going through our credit filters because we tried and tested credit filters it gives you the best returns that you can and the returns far more than the average of the bank. So we are quite enthused to do PSL organically to the extent it comes through our credit list. In the past year and two years when we have had muted retail to some extent the PSL component is also lower because we did not get the retail as much, but as we are now opening up more retail and going you see that the PSL comes back organically still it is only one of the components because we do not leave other components on the table we want all of them for example organic is one where I think we have said that it is little Page 16 of 22 July 16, 2022 more than call it in 30% to 35% or a little more than two-thirds to 70%-75% we get through organic and then there are other tools that we always use and we want to continue to use them one is the PSL certificates we get that too. the other one is the RIDF is also something where there is always a trade-off that is done what is the organic that you can build within your credit filters and if you go outside of your credit filters what sort of credit cost are you going to end up and so thereby what returns, what is the cost of the PSLC, what is the cost of RIDF and so this is always an equation that happens almost in a quarter few times that you balance this to see where is the breakeven and which is the right way to go about. So that is how decisions are done and when we did not do retail we have done more of the other things that will happen and then we do more of retail there will be more of organic that comes up. So that is how we think about the PSL',
        'So that is how decisions are done and when we did not do retail we have done more of the other things that will happen and then we do more of retail there will be more of organic that comes up. So that is how we think about the PSL. Kunal Shah: Sure so PSLC what we bought 100000 Crore maybe with HDFC it is that there is a scope for this to go up substantially from here on because 80000 has already gone up to 1 lakh last year and maybe with this requirement I think there will be more and more maybe purchase of PSLC which could happen. Srinivasan V: See purchase again as I told you these are the three, four elements that happens right PSLC, RIDF, organic PSL growth and we do participation certificates so that the several components are happening and we have to balance the cost towards the returns that each one gives. So there is no one particular target if you ask me do you know whether this 1 lakh Crores is going to go to x or y there is no predetermined formula that we operate, the formula is which gives you the best return what is the break even on indifference point between various instruments that is what drives the decision and that is as you know is a dynamic decision because the price in the market is dynamic, is not a fixed price and so that is how that is determined periodically and then the outcomes is what you are seeing. Kunal Shah: Sure, thanks a lot. Moderator: Thank you. The next question is from the line of Adarsh from CLSA. Please go ahead. Page 17 of 22 Adarsh: So just on the cost side any sense, it clearly is you are an investment mode in branches and employees. So any path towards cost to income over the next couple of years. Srinivasan V: Good, thank you for asking, but one thing normally resist from giving July 16, 2022 forward guidance on many things. But let me talk through so you will get an idea of what previously we have talked and our thought process so you will factor that',
        'Please go ahead. Page 17 of 22 Adarsh: So just on the cost side any sense, it clearly is you are an investment mode in branches and employees. So any path towards cost to income over the next couple of years. Srinivasan V: Good, thank you for asking, but one thing normally resist from giving July 16, 2022 forward guidance on many things. But let me talk through so you will get an idea of what previously we have talked and our thought process so you will factor that. One from a top line point of view the growth is picking up, you have seen that over a period of time that the top line growth, the top line means I meant the volume growth was anyway there, but the mix is also we will see like this quarter and similarly last quarter the mix is also changing to get that the top line revenue interest income, our interest income growth component also moving up you are seeing that come up, that gives you a little more kind of a confidence and an opportunity to make the right kind of investments that you want because we want to feed that from a growth point of view that is one from a balancing point of view. The second thing is in terms of the credit situation. So we have come after a pandemic credit kind of a scenario as the credit gets benign which is already you are seeing some benign credit environment and when I say credit benign means I meant from credit cost that is benign. We have that is part of what you have seen is mix of investments, making investments in people when the credit costs have been below what we have seen historically what we have seen before the COVID we have taken the opportunity to make those investments in expenses both people, technology as well as on branches. So these are the two considerations we have always given how to make those investments for the future by using the credit benign conditions and how to make an opportunity of the top line growth so that you can balance the expenses',
        'So these are the two considerations we have always given how to make those investments for the future by using the credit benign conditions and how to make an opportunity of the top line growth so that you can balance the expenses. Now coming to the last aspect which is the crux of what you are saying what is the cost to income on how we should think about. If you go back to the pre-COVID our cost to income has been 39.6 the full year before the COVID 39.6. You can call it 39 and call it 40. We have always said that as the retail picks up, retail is an upfront cost and the top line comes with the lag and comes over a two, three year period so you put the cost in and it comes over two, three years period, that is the nature of that retail once you want to grow retail that is the way it happens and you are seeing that pick up and we have said that through the COVID period even when we wanted to spend we did Page 18 of 22 July 16, 2022 not have the opportunity to spend and we have been saying that we have been waiting for that opportunity to spend to get that retail back up and now that is chugging along and so the cost to income on an overall basis call it 40 or so which is the pre-COVID quarter-to-quarter variations will happen and if you ask Sashi, I think he has told in the past in certain other meetings that quarter-to-quarter variations can happen because it is a question of the timing, but over a period of a year or two years if you see you can touch 40 but over a medium-term three to five years this is something as a forward guidance normally which we do not do but from a cost to income what we see as an opportunity we said it will get to the mid 30s and which is what we said pre-COVID but COVID has put halt to that changing the composition of the product mix as well as our spend mix and as we get back to normalization and execute we should get back to that kind of a trajectory over time',
        'Adarsh: On asset quality ex agri’s, safe to say that things are trended absolutely in the right direction. Srinivasan V: Yes. Adarsh: So what is the risk there given credit cost to income as of now it looks like it is most of the segments and what is foreseeable in future Srinivasan V: You talk about the credit right you talk about the NPA. Adarsh: No. Srinivasan V: Ex of agri it has been quite good if you see at an aggregate level and that it has got a component of the business as usual which is extremely benign because it is originated with a very tight credit conditions and it has also got a component of the restructuring some of them that to whom we are given the opportunity to redeem themselves to come back to normal life and some of them have taken that opportunity on the restructuring and used it to come back to normal life some of them who still struggle get into NPA, but on a combined basis you are seeing that it continues to get benign and better another one, two quarters you should see it is even more benign. Page 19 of 22 July 16, 2022 Adarsh: This is useful and that is it from my side. Thanks for answering my questions. Moderator: Thank you. The next question is from the line of Abhishek Murarka from HSBC. Please go ahead. Abhishek Murarka: Hi! Good evening Srini and team and congratulations for the quarter. I have two questions, one on NIM and one on Opex. On NIM the repo hike that happened in May, June when does it fully translate into yields would it be by the end of August or September and also if you can share the EBLR repo, non-repo and fixed and floating breakup for the loan book that would be useful. Srinivasan V: First on the NIM the repricing starts, it started in May and there is a cycle, there is at least a three-month cycle and some of them are a six-month cycle in terms of what happens. So that is on the NIM and so it is not just that, it has also got to do with the deposit cost',
        'Srinivasan V: First on the NIM the repricing starts, it started in May and there is a cycle, there is at least a three-month cycle and some of them are a six-month cycle in terms of what happens. So that is on the NIM and so it is not just that, it has also got to do with the deposit cost. So just the repricing on the repo or receivable, it also happens in the cost of funds, but then we do expect that the tailwind of the rates going up and if you think about the second aspect on the NIM that you asked in terms of the fixed and the variable about 45% of the book is fixed and the 55% is floating rate and some of them out of the 55%, 48% which is 27%, 28% of the total book is repo and a quarter 13%, 14% of the total bank book is achievable. So that is the kind from a mixed point of view, pricing point of view you can think that is how it moves on. Abhishek Murarka: So just extending that for the NIM outlook of course you know that there would be a certain amount of uptake in term deposit rates as well. So just generally would we still expect a retail and CRB proportions to rise in the loan mix and the expansion that you see in the yields sort of outpacing the TD uptick. So do we expect these two things to continue for the next three to four quarters. Srinivasan V: From a NIM point of view it is also rightfully you are focused on the mix because that is what makes it right because individually things can go if the mix do not come it takes a little more longer time. The mix as we speak now is still at 45-55, although the retail grew at 5% and the corporate was zero and the CRB was 2.7% sequentially. The mix is more or less the Page 20 of 22 July 16, 2022 same, one quarter does not take it, it takes a few quarters for the mix and last quarter we put out the chart in terms of how long it took for the mix to come retail 55% how long it took for that retail to come to 45% and there is the chart year-by-year it shows how long it took',
        'The mix as we speak now is still at 45-55, although the retail grew at 5% and the corporate was zero and the CRB was 2.7% sequentially. The mix is more or less the Page 20 of 22 July 16, 2022 same, one quarter does not take it, it takes a few quarters for the mix and last quarter we put out the chart in terms of how long it took for the mix to come retail 55% how long it took for that retail to come to 45% and there is the chart year-by-year it shows how long it took. So while on the way up it could be faster because the rate of growth on the retail and the demand in the micro environment we see on the retail is higher so it could be faster, but yes both the inherent demand we see in CRB and in retail is quite good and high and one other thing I want to be cautious and tell you to just because we think we see good demand if there is a great demand in wholesale we are not going to turn down the wholesale loan just because that the NIM has to come up, at the end of the day what matters in terms of this decision is does it give good returns at the end of the day ROA, ROE does it provide the right kind of return if it does it goes through, but from an inherent demand point because I did mention this because March the same conversation happened and we saw the wholesales come in with a greater vigor for a growth in March quarter and when it came I was not able to go back and say by the way we talked about retail and CRB having a faster growth rate inherent growth rate but the wholesale has come to decline wholesale, so we said we should go with the whatever is the demand which is there we like the customer, we like the credit right pricing gives you the return should go and so that is the kind of a decision making that happens but inherently retail and CRB are having a good amount of demand. Abhishek Murarka: Got it, thank you and the other question is on Opex',
        'Abhishek Murarka: Got it, thank you and the other question is on Opex. So can you share some sort of targets on how much you want to hire for the rest of the year and also what is your tech spend this quarter as a percentage of overall Opex where is that trending. Srinivasan V: Two things, one in terms of the hiring there is no predetermined that hiring depends on the productivity we measure all products, all geographies, branches, non-branches, customer segments, in terms of the productivity, which means the RM on the sales force give to the customer or to the sales unit. So it depends on the productivity that comes and continuously we drive the productivity up. So we have a model best-in-class model and we have a best tool and we periodically look at who and where it is sub- optimal and we drive the productivity. So that is part of how we do and the people addition we do as necessary to meet those opportunities when the Page 21 of 22 July 16, 2022 productivity is saturated we do need to add to get the more volume. So we are not shy of adding because it brings in better volumes and better relationships then your other aspect in terms of the technology, yes, I think in the past we have said the technology spend to total expenses 8%, 9% or so that is stable over a longer period of time that is the kind of range in which it operates. For quarter-to-quarter it can move around, but broadly that is where it is. Abhishek Murarka: So it would be in that 8% to 9% range this quarter as well. Srinivasan V: Yes, quarter-to-quarter it can be different but broadly that is where it goes. Abhishek Murarka: Okay got it Srini thank you that was useful and all the best for the following quarters. Moderator: Thank you. Ladies and gentlemen, this would be the last question for the day given the time I would now like to hand the conference over to Mr. Vaidyanathan for closing comments. Srinivasan V: Thank you Faizan and thank you to all the participants who dialed in today',
        'Srinivasan V: Yes, quarter-to-quarter it can be different but broadly that is where it goes. Abhishek Murarka: Okay got it Srini thank you that was useful and all the best for the following quarters. Moderator: Thank you. Ladies and gentlemen, this would be the last question for the day given the time I would now like to hand the conference over to Mr. Vaidyanathan for closing comments. Srinivasan V: Thank you Faizan and thank you to all the participants who dialed in today. If you still have more questions or need any clarifications feel free to get in touch with the investor relations team, we will be happy to engage. Thank you with that we will sign off for today. Moderator: Thank you. Ladies and gentlemen, on behalf of HDFC Bank Limited that concludes this conference call. Thank you for joining us and you may now disconnect your lines. Page 22 of 22'
    ]
    final_summary= get_final_output(hdfc_chunks)
    print("Final summary")
    print(final_summary)
    

main()
