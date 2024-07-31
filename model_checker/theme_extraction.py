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
    adani_chunks=[
        'Manan Vakharia from Investor Relations team. Good Morning Sir and we will start with opening remarks and post which we will have question and answer session. Thank you and over to you Sir! Robbie Singh: Thank you so much. Hi Good Morning all. This is Robbie Singh, CFO of Adani Enterprise. I welcome you all to the earnings call to discuss Q1 FY23 results. AEL continues to create value for its shareholders as a successful incubator for the past two-and-a-half decades. This incubation model has created leaders in the respective sectors like Adani Ports, Adani Transmission, Adani Green Energy, Adani Total Gas, and Adani Wilmar and has delivered returns at a compound annual growth rate of 36% to shareholders. AEL holds a portfolio of businesses - both established and incubating - which are spread across different verticals in energy and utility, transport and logistics, direct to consumer and primary industries. Within primary industries it has established businesses of mining services and integrated resource management along with the developing vertical of metals. As our established business continue to sustain long term growth, we are making significant progress in our attractive incubation pipeline comprising of energy and utility which is Adani New Industries - it is a green hydrogen ecosystem and full service data center business AdaniConneX. In the transport and logistics we have Adani Airport Holdings and Adani Road Transport Limited businesses which will further accelerate value creation for Adani Enterprise shareholders. We are happy to inform that AEL has completed primary equity transaction of Rs.7700 Crores with Abu Dhabi based International Holding Company for 3.5% stake. This validates our strong capital management philosophy of equity funded growth and conservative leverage targets. Let me give you a quick update of our incubating businesses',
        'We are happy to inform that AEL has completed primary equity transaction of Rs.7700 Crores with Abu Dhabi based International Holding Company for 3.5% stake. This validates our strong capital management philosophy of equity funded growth and conservative leverage targets. Let me give you a quick update of our incubating businesses. In Adani New Industry portfolio as all of you would know we have announced investment of USD 50 billion over the next decade in developing green hydrogen ecosystem. This will be housed under Adani New Industry Limited. ANIL will have three business streams – (i) Manufacturing ecosystem to include module, cell, ingots, wafers and wind turbines, electrolyzers and associated ancillary equipment ecosystem. (ii) The green hydrogen generation include development of solar and wind power plants to produce green hydrogen (iii) Downstream products depending on the usage for ammonia, urea, methanol, etc. During the quarter we announced our partnership with TotalEnergies to develop the world’s largest green H2 ecosystem. TotalEnergies will acquire 25% stake in ANIL. While the Page 2 of 9 Adani Enterprises Limited August 05, 2022 transaction will follow customary approval process, it takes the company one step ahead to produce the world’s least expensive electrons which will drive our ability to produce the world’s least expensive green hydrogen. Following are some of the updates on development: Existing capacity of 1.5 GW at Mundra is increasing to 3.5 GW and this additional 2 GW will be completed by September this year. With this the overall capacity will reach to 3.5 GW. Wind turbine erection for the first 5.2 MW wind turbine has been completed and testing and certification is underway. We expect completion in the next 6 months. We have identified three trial sites for initial testing of electrolyzers and we expect the testing to commence by end of this calendar year or early next year',
        '5 GW at Mundra is increasing to 3.5 GW and this additional 2 GW will be completed by September this year. With this the overall capacity will reach to 3.5 GW. Wind turbine erection for the first 5.2 MW wind turbine has been completed and testing and certification is underway. We expect completion in the next 6 months. We have identified three trial sites for initial testing of electrolyzers and we expect the testing to commence by end of this calendar year or early next year. From operational point of view, module sales from our manufacturing ecosystem within ANIL stood at 264 MW. EBITDA from these sales was at Rs. 42 Crores. In Adani Airport Holdings portfolio, passengers movement at the airports increased by 35% and it is now at 16.6 million which is approximately 85% of the pre-COVID numbers. Construction at Navi Mumbai International Airport has started and approximately 26% of the work is completed. In Adani Road Transport portfolio, we have signed concession agreement in May for Kagal-Satara road project of 65 KMs in Maharashtra under BOT basis. We have also received provisional COD for Bilaspur road project and construction activity is progressing well on other 8 road projects. The current road portfolio is now approximately Rs.38000 Crores of both operating and under development projects. A quick update of primary industries before I handover to my colleague Vinay. We achieved financial closure for our first metals business copper. This was led by SBI and it has been further down sold to various other banks. Financial performance of AEL for this quarter in terms of revenue number has increased 223% and now is at Rs.41066 Crores. Consolidated EBITDA increased by over 100% and is at Rs.1965 Crores with strong performance from both established and incubating businesses. Now I invite my colleague Vinay to take you through mining services and IRM business highlights. Vinay over to you! Vinay Prakash: Thanks Robbie. Good Morning to all',
        'Financial performance of AEL for this quarter in terms of revenue number has increased 223% and now is at Rs.41066 Crores. Consolidated EBITDA increased by over 100% and is at Rs.1965 Crores with strong performance from both established and incubating businesses. Now I invite my colleague Vinay to take you through mining services and IRM business highlights. Vinay over to you! Vinay Prakash: Thanks Robbie. Good Morning to all. In fact as far as the mining services business is concerned Adani Enterprise Limited is the pioneer of MDO concept in India with an integrated business model that spans across developing mines as well as the entire upstream and downstream activities. It provides the full service range - right from seeking various approvals, land acquisition, R&R, developing required infrastructure, mining, beneficiation and transportation to designated consumption plants. The company is also MDO for nine coal blocks and two iron ore blocks with combined peak capacity of 120 MMT per annum. These 11 projects are located in the state of Chhattisgarh, MP and Odisha. The mining production volume increased by 72% to 8.1 MMT on year-on-year basis and further dispatch increased by 58% to 7.2 MMT on year-on- year basis. The revenue from mining services increased by 18% to Rs. 677 Crores and EBITDA stood at Rs. 268 Crores versus Rs. 307 Cores on year-on-year basis on account of high operating costs. As Robbie told about copper business apart from financial closure, operational activities are progressing well and it is as per schedule. Additionally, we have also received PLI scheme approval for copper tube value added scheme for our copper business. As far as the IRM business is concerned, in terms of IRM business we have continued to develop business relationship with Page 3 of 9 Adani Enterprises Limited August 05, 2022 diversified customers across various end-user industries',
        'As Robbie told about copper business apart from financial closure, operational activities are progressing well and it is as per schedule. Additionally, we have also received PLI scheme approval for copper tube value added scheme for our copper business. As far as the IRM business is concerned, in terms of IRM business we have continued to develop business relationship with Page 3 of 9 Adani Enterprises Limited August 05, 2022 diversified customers across various end-user industries. We retained number one player position in India and having the endeavor to maintain this position going forward. The volume in Q1 FY23 increased by 52% to 26.7 MMT, the EBITDA has increased by 72% to Rs.950 Crores on account of higher volumes. Thank you. Moderator: Thank you very much. We will now begin the question and answer session. The first question is from the line of Mohit Kumar from DAM Capital Advisors. Please go ahead. Mohit Kumar: Good Morning Sir and congratulations on achieving success in number of initiatives and especially raising money. Sir my first question is on the mining operations given that you achieved 8.1 million tonne our profitability EBITDA is not improving it has happened in Q4 also and in this quarter as well. How do you expect this EBITDA ramp up to happen, I understand that we opened around 55 million tonne of capacity when do you expect this peak capacity to achieve? Vinay Prakash: As I told last time also in PEKB mine considering that we are getting higher cost because of the diesel explosive and the stripping ratio, our EBITDA has gone down slightly and we are working on various other technology initiatives to see as how we can recover it back. In fact in this volume the volume of Talabira and other mines have also included where we have lower revenue because of having lower scope and considering the mining cost per tonne is lower and definitely the EBITDA per tonne is also lower in those mines',
        'In fact in this volume the volume of Talabira and other mines have also included where we have lower revenue because of having lower scope and considering the mining cost per tonne is lower and definitely the EBITDA per tonne is also lower in those mines. As far as the increase in volumes are concerned we are working on both the sites trying to see if we can have alternate fuels to be used apart from going for the electric equipments and some technology changes. We are confident that we will be in position to increase the EBIDTA level going further. Mohit Kumar: The new mines which are under development, which are the next mines you are likely to open in the next 24 months? Vinay Prakash: So we have few mines which should get opened in the next 24 months like Parsa is ready to get opened subject to certain local approvals we should open that mine soon. We have Suliyari mine which is now going to go for its peak capacity. We have Talabira mine which did 7 million tonne last year should do 12 million tonne this year and going forward should reach to its peak capacity which is 20 million tonne. We have other commercial mines like Dhirauli and Bijahan which should also come on board. Mohit Kumar: Understood Sir and is it possible to break up the IRM operations into Carmichael and PO trading business? Is it possible are you sharing that number, in case you are sharing please share the revenue EBITDA number for Carmichael? Vinay Prakash: I think this IRM business is independent of Carmichael business, Mr. Shah will like to comment here but this IRM is excluding Carmichael. Saurabh Shah: So this number of IRM basically is only the trading business not the Carmichael commercial mines',
        'Shah will like to comment here but this IRM is excluding Carmichael. Saurabh Shah: So this number of IRM basically is only the trading business not the Carmichael commercial mines. Page 4 of 9 Adani Enterprises Limited August 05, 2022 Mohit Kumar: Understood Sir but can you share the revenue and EBITDA of Carmichael if possible? Saurabh Shah: So just to give an idea that we have done one million metric tonnes of sales in Carmichael mines this quarter with EBITDA of about Rs. 85 Crores is what we had. Mohit Kumar: First on the LoA side the government I think we are participating in the PLI scheme and we were one of the winners of that have we received the LoA from the government? Robbie Singh: That is continuing. We are not doing this business for PLI scheme. I think we should keep the focus on the fact the Adani New Industries is setting up green hydrogen ecosystem. Now if the PLI scheme happen it would be wonderful because that is part of Atmanirbhar programme of the government and it will benefit us but independent of that, it is a specific opportunity for India to finally have an energy source that is domestic and therefore it has its own economic merits. Having said that if you can kindly frame the questions in relation to that manner it is much, much better. Mohit Kumar: Understood Sir. Of course we are doing the first ramp up of 1.5 to 3.5 megawatt. What is the next in the solar manufacturing business and what duration you think you will do the third expansion? Robbie Singh: Basically the ramp up of the business is related to what our final targets to achieve the cheapest electron for production of green hydrogen. We will have in phase one a capacity of 10 gigawatt from all the way from polysilicon to ingots, wafer and cell and module line. We will also have initial capacity of 2.5 gigawatt of wind turbines which will also scale up to 7',
        'What is the next in the solar manufacturing business and what duration you think you will do the third expansion? Robbie Singh: Basically the ramp up of the business is related to what our final targets to achieve the cheapest electron for production of green hydrogen. We will have in phase one a capacity of 10 gigawatt from all the way from polysilicon to ingots, wafer and cell and module line. We will also have initial capacity of 2.5 gigawatt of wind turbines which will also scale up to 7.5 and we will also have capacity for electrolyzers plus glass aluminum frames and back sheet, so this whole ecosystem is to provide inputs into the production of cheapest electron so that we can convert that to the cheapest hydrogen. So, the scale up will be related to how we are developing and we expect the first production of hydrogen to commence late calendar year 2025 or first quarter of calendar year of 2026 so that is the ramp up schedule that we are working towards. The polysilicon should be full 10 gigawatts to start off because that is the minimal scale of its economics. Mohit Kumar: Timeline for polysilicon any tentative timeline? Robbie Singh: 30 months around that period. Mohit Kumar: Understood Sir. Thank you and best of luck. I will go back in the queue. Moderator: Thank you. The next question is from the line of Apoorva Bahadur from Investec. Please go ahead. Apoorva Bahadur: Hi Sir. Thank you so much for the opportunity. Sir I wanted to know if we have finalized the electrolyzer technology partnership? Page 5 of 9 Adani Enterprises Limited August 05, 2022 Robbie Singh: No, we are working on that. As I mentioned we are currently in the process of starting the testing programme for electrolyzer and parallel to that we are continuing to work on partnerships. But it is a dual track process both partnerships and indigenous capacity',
        'Please go ahead. Apoorva Bahadur: Hi Sir. Thank you so much for the opportunity. Sir I wanted to know if we have finalized the electrolyzer technology partnership? Page 5 of 9 Adani Enterprises Limited August 05, 2022 Robbie Singh: No, we are working on that. As I mentioned we are currently in the process of starting the testing programme for electrolyzer and parallel to that we are continuing to work on partnerships. But it is a dual track process both partnerships and indigenous capacity. Apoorva Bahadur: So we are confident that our factory will be up and running and start producing electrolyzer so that we can start producing hydrogen by end 2025. Those timelines are comfortable? Robbie Singh: Yes. Apoorva Bahadur: Sir also on the wind capacity side, can you guide by when this 2.5 gigawatt capacity be commercial and also will we be selling the turbines to third party in the commercial market or will it entirely be for our captive consumption? Robbie Singh: Mostly initially will be for captive consumption. We have commissioned or erected the first 5.2 megawatt wind turbine; it requires a certification process. We expect the certification process to complete in the next 6 months and by that time we will ramp up the production capacity and we would be ready to produce 2.5 gigawatt. Apoorva Bahadur: Right Sir and last question from my side and that is on the power storage front. So, I believe for producing hydrogen using alkaline which we will be going for initially we require sort of longer duration power available or more reliable power so what is our take on that',
        'We expect the certification process to complete in the next 6 months and by that time we will ramp up the production capacity and we would be ready to produce 2.5 gigawatt. Apoorva Bahadur: Right Sir and last question from my side and that is on the power storage front. So, I believe for producing hydrogen using alkaline which we will be going for initially we require sort of longer duration power available or more reliable power so what is our take on that. Which technology or which method of storage will we be backing ? Will it be from hydro or battery and in either cases if you have already done some tie ups? Robbie Singh: That is the configuration we are testing so we are going through as to what and how the configuration of best operating configuration works to give the cheapest hydrogen given a power input profile and so therefore we are going to be testing the electrolyzer purely at a wind power site, purely at solar site and a hybrid wind and solar site so we will complete that parallelly to then come up with the best design configuration under which the optimal hydrogen can be produced at an optimal cost so that we are in the process of doing. Once it is done we will continue to disclose it to the market as to how we are doing, but at this stage in phase one of the analysis it does not rely on battery for storage. Apoorva Bahadur: Okay Sir last question if I may squeeze in one more and I believe we intend to move hydrogen from Khavda to Mundra using a pipeline so have we started the construction of it and what type of capital cost can we expect? Robbie Singh: The pipeline is not a big cost. Pipeline is very, very small cost of the overall project but the right of way and all that analysis is being completed and it is only about 200 KMs so it is not a very long pipeline so we expect that to be ready and complete',
        'Pipeline is very, very small cost of the overall project but the right of way and all that analysis is being completed and it is only about 200 KMs so it is not a very long pipeline so we expect that to be ready and complete. Apoorva Bahadur: No specialized material needed to transport hydrogen because that is what we have read? Robbie Singh: They are specialized but they are all available. Page 6 of 9 Adani Enterprises Limited August 05, 2022 Apoorva Bahadur: Okay fair enough Sir. Thank you very much and all the very best. Moderator: Thank you. The next question is from the line of Sandip Bansal from ASK Investment Managers. Please go ahead. Sandip Bansal: Thank you Sir for the opportunity. Robbie, I wanted to understand what would be the capital requirements across our various businesses over the next few years especially on the equity side? Robbie Singh: From equity point of view for the time being we are fully funded. We did already in advance. From time-to-time we might look at appropriately funding various businesses but the bulk we completed that in May with IHC Rs.7700 Crores and overall from just pure capex perspective we expect the capex commitments over the next two years to be in the order for AEL of give or take about Rs.85000 Crores and including revenue plus EPC margin plus internal cash flows and already funded equity that is fully covered in our current planning so we do not anticipate any new equity for the already identified projects. Sandip Bansal: Sure so this Rs.85000 Crores if I understand correctly is FY2024 and 2025 or FY2023 and 2024? Robbie Singh: It is 2023 and 2024 but of this Rs.85000 approximately like about Rs.7000 Crores to Rs.8000 Crores which we have already spent on our manufacturing ecosystem it is already spent. It will just be completed this year part of it. From an outlay perspective it is completing in FY2023 just as to give break up for you so that you see that airports will be about Rs',
        'Sandip Bansal: Sure so this Rs.85000 Crores if I understand correctly is FY2024 and 2025 or FY2023 and 2024? Robbie Singh: It is 2023 and 2024 but of this Rs.85000 approximately like about Rs.7000 Crores to Rs.8000 Crores which we have already spent on our manufacturing ecosystem it is already spent. It will just be completed this year part of it. From an outlay perspective it is completing in FY2023 just as to give break up for you so that you see that airports will be about Rs.11000 Crores this year, roads would be around Rs.8900, we will have in the copper about Rs.2900, data center small amount Rs.300 odd Crores and in our other materials businesses around Rs.4400 Crores and that will be Rs.37000 odd Crores of which as I mentioned approximately Rs.8500 is already spent in the previous year completing this year and the following year in 2024 we expect to have another Rs.48000 Crores of capex. Sandip Bansal: Sure thanks for that. On the airport business side if you can help us understand some of your plans because now it has been a few months since we have taken over the Mumbai airport as well and Navi Mumbai construction has also started. So how should we think about the trajectory for revenues as well as EBITDA over the next few years? Robbie Singh: See the best way to explain the airport business is that we look at airports as hyperlocalized community economic assets which is that primarily the airport is an economic center for a regional area in which it is be it Jaipur, be it Ahmedabad, be it Lucknow, Navi Mumbai, Mumbai. So number one, we have a plan which we call the city side planning and the city side planning is purely catering to its local community so that is one',
        'So number one, we have a plan which we call the city side planning and the city side planning is purely catering to its local community so that is one. Second is related to passenger activity and associated activities which is related to aero and non aero income at the airports, There are three income streams, the city economic area activity and the passenger aero and non- aero activity, so we expect to complete our first phase of our city side which is non-passenger related but city side developments by first phase it should be completed by 2026 and fully completed by 2030 across our eight airport sites. Because it is so far out and those are the numbers not formally presented because of the period of which they are forecast overall the Page 7 of 9 Adani Enterprises Limited August 05, 2022 structure would work is the city side developments or local economic area should give us about 55% to 60% of our EBITDA and aero and non-aero should give us the other 40% EBIDTA of the business so I would not, till we formally start reporting on Adani Airport Holding as a business unit which we will shortly. We do not want to hazard unverified forecast for the airport business per se. This year though airports achieved just to give you a passenger movement of about 16.6 million this quarter and cargo of about 2.3 lakh metric tonnes and broadly we achieved approximately Rs.540 Crores of EBITDA that we achieved from the airport business this quarter, but this is without any income coming yet from the city side economic developments which will commence about three years from now. Sandip Bansal: Sure and just one clarification when you talk about city side development does this include the real estate monetization at the Mumbai airport as well or that is excluding that? Robbie Singh: We do not look at this as real estate monetization; there is no monetization of real estate',
        'Sandip Bansal: Sure and just one clarification when you talk about city side development does this include the real estate monetization at the Mumbai airport as well or that is excluding that? Robbie Singh: We do not look at this as real estate monetization; there is no monetization of real estate. It is actually what you are building for the local community and how the local community spends within the airport region so it is not specifically like trying to monetize the land and trying to undertake that kind of activity it is more related to what is needed in the specific local community you build facilities for that community and then you are earning income from that built capacity so it will depend from city-by-city. It is more of a consumer centric model rather than a model that is related to monetization of land, etc. Sandip Bansal: Sure Robbie Thank you so much. Moderator: Thank you. Next question is from the line of Mohit Kumar from DAM Capital Advisors. Please go ahead. Mohit Kumar: Thanks for the opportunity once again Sir. Few clarifications do you have access to all the 140 acre land in Mumbai airport? Robbie Singh: Yes of course we have access to all the lands. Mohit Kumar: My question is it completely with us or it is something which is still to be transferred to us for the development you want to do on the land? Robbie Singh: Majority of it is with us. Mohit Kumar: My second question is on the profitability of the solar business. I think this quarter it was impacted adversely I believe this is primary because of the movement in polysilicon prices. How do you see over the next couple of quarters given the fact that you are going to ramp up, do you think the profitability when will go back to old level of profitability? Robbie Singh: The solar business is not the business that we look at in isolation',
        'Mohit Kumar: My second question is on the profitability of the solar business. I think this quarter it was impacted adversely I believe this is primary because of the movement in polysilicon prices. How do you see over the next couple of quarters given the fact that you are going to ramp up, do you think the profitability when will go back to old level of profitability? Robbie Singh: The solar business is not the business that we look at in isolation. We look at is an integrated part of the green hydrogen ecosystem so there is no specific target that we want to achieve for the Page 8 of 9 Adani Enterprises Limited August 05, 2022 solar business. We want to achieve the target for green hydrogen per kg so that is the matrix that we are focused on. Mohit Kumar: But are you saying that this 2.5 gigawatt entirely will be for the captive requirements? Robbie Singh Mostly yes. Mohit Kumar: My request is to share the debt breakup which we used to do earlier business wise it used to be very helpful. Robbie Singh We still do it. It will be available in September results. It is available twice a year September and March. Mohit Kumar: Okay understood Sir. Thank you and all the best. Moderator: Thank you. Our next question is from the line of Nirav Shah from GeeCee Investments. Please go ahead. Nirav Shah: Hi Sir and thanks for the opportunity. Just one question. Most of them have been answered. On the MDO front we have given guidance of around 40 million tonnes in this year and around 75 odd tonnes next year so can you just update on that whether are we maintaining it or upgrading it? Vinay Prakash: We are maintaining this 40 million tonne for this year and for the next year 65 to 75 million tonne depending upon the timing of various approvals. Nirav Shah: Okay great Sir. Thanks Sir. Moderator: Thank you. As there are no further questions I now hand the conference over to management for closing comments. Robbie Singh: Firstly thank you so much for organizing',
        'Nirav Shah: Okay great Sir. Thanks Sir. Moderator: Thank you. As there are no further questions I now hand the conference over to management for closing comments. Robbie Singh: Firstly thank you so much for organizing. Thank you for the investors and team to ask questions. Also thanking Saurabh, Manan and team and my colleague Vinay for this call, so we take the opportunity to see you again in October. Saurabh Shah: Thanks Ankita from Elara to organize this and thank you so much. We look forward to have similar kind of organization in the next coming quarters. Ankita Shah: Thank you Sir. Thank you for the opportunity. Moderator: Thank you very much. On behalf of Elara Securities Limited that concludes this conference. Thank you for joining us. You may now disconnect your lines. Thankyou Page 9 of 9'
    ]
    final_summary= get_final_output(adani_chunks)
    print("Final summary")
    print(final_summary)
    

main()
