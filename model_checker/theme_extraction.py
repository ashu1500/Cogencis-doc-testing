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
    maruti_chunks=[
        'I also like to inform you that the call is being recorded, and the audio recording and the transcript will be available at our website. May please note that in case of any inadvertent error during this live audio call, the transcript will be provided with the corrected information. I would now like to invite our CFO, Mr. Seth. Over to you, sir. Ajay Seth: Thanks Pranav, Good afternoon Ladies and Gentlemen, I hope you and your families are healthy and safe. Let me start with some business highlights during the quarter. Maruti Suzuki celebrated ‘40 years of Suzuki’s partnership with the people of India’. During the event, Hon’ble Prime Minister laid foundation stone of Suzuki Motor Gujarat electric vehicle battery manufacturing facility at Hansalpur, Gujarat and Maruti Suzuki vehicle manufacturing facility in Kharkhoda, Haryana. The Company was incorporated to provide cars for the masses of India and also build a vibrant manufacturing Industry in India. We are happy to share that the Company has been true to its reason for existence even today. If we look back, one of the key success factors in our journey has been the strong focus on understanding and fulfilling the needs of customers by offering them relevant products, technologies, and services. Over the years, customers have evolved and accordingly our products, services, and business processes too have aligned, keeping the customers at the heart of it. MSIL Conference Call Transcript 28th October 2022 2 | P a g e The other factor has been how we have always thought of the long-term in all our actions. All management decisions are based on the long-term interests of our stakeholders. Last but not the least, we have a very good blend of Indian and Japanese culture in our Company. We were able to combine Japanese shop-floor practices and discipline with Indian innovation and zeal in our operations',
        'MSIL Conference Call Transcript 28th October 2022 2 | P a g e The other factor has been how we have always thought of the long-term in all our actions. All management decisions are based on the long-term interests of our stakeholders. Last but not the least, we have a very good blend of Indian and Japanese culture in our Company. We were able to combine Japanese shop-floor practices and discipline with Indian innovation and zeal in our operations. Our parent Suzuki, Japan, has been a silent support, trying to look at the future from its global experience and carefully selecting the best technology and products for Indian customers. Coming to the recent new model launches, • In September, the Company started retailing its newest flagship offering from NEXA, the Grand Vitara. With over 75,000 bookings in a short span of time, customer response for Grand Vitara is overwhelming. The Grand Vitara is a multi-product offering with cutting-edge Intelligent Electric Hybrid powertrain, Progressive Smart Hybrid technology and Suzuki ALLGRIP SELECT technology is designed to appeal to a varied customer base and will revolutionize the SUV space in India. • In August, the Company launched a full model change of its iconic brand Alto. The All-New Alto K10 is loaded with host of comfort, safety, convenience and connectivity features. • The Company further strengthened its green vehicles’ portfolio by introducing S-CNG powertrain technology in Swift and S-Presso. With this, Maruti Suzuki now offers 10 vehicles with factory-fitted S-CNG technology. Maruti Suzuki’s Research & Development facility conducts rigorous testing for its factory-fitted S-CNG cars to deliver unmatched safety, performance, durability and fuel efficiency. Going forward, the Company will stive to further strengthen its SUV portfolio to dominate the SUV segment, just like all other segments',
        'With this, Maruti Suzuki now offers 10 vehicles with factory-fitted S-CNG technology. Maruti Suzuki’s Research & Development facility conducts rigorous testing for its factory-fitted S-CNG cars to deliver unmatched safety, performance, durability and fuel efficiency. Going forward, the Company will stive to further strengthen its SUV portfolio to dominate the SUV segment, just like all other segments. Coming to the business environment during the quarter, On the back of better availability of electronic components, the Company reported its highest ever sales volume in any quarter. The electronics component shortages are still limiting our production volumes. In this quarter, the Company could not produce 35,000 vehicles. Limited visibility on availability of electronics components is a challenge in planning our production. Our Supply Chain, Engineering, production and sales teams are working towards maximizing the production volume from available semi-conductors. The supply situation of electronic components continues to remain unpredictable. Coming to the Highlights of Q2 (July-September), FY 2022-23 The Company sold a total of 517,395 vehicles during the quarter. Sales in the domestic market stood at 454,200 units. Exports were at 63,195 units. The same period previous year was marked MSIL Conference Call Transcript 28th October 2022 3 | P a g e by acute shortage of electronic components and consequently the Company could sell a total of 379,541 units comprising 320,133 units in domestic and 59,408 units in export markets. Pending customer orders stood at about 412,000 vehicles at the end of this quarter out of which about 130,000 vehicle pre-bookings are for recently launched models. During the quarter, the Company registered its highest-ever quarterly Net Sales of INR 285,435 million. During the same period previous year, the Net Sales were at INR 192,978 million',
        'Pending customer orders stood at about 412,000 vehicles at the end of this quarter out of which about 130,000 vehicle pre-bookings are for recently launched models. During the quarter, the Company registered its highest-ever quarterly Net Sales of INR 285,435 million. During the same period previous year, the Net Sales were at INR 192,978 million. The Operating Profit in quarter 2 FY2022-23 stood at INR 20,463 million as against INR 988 million in quarter 2 FY2021-22. The Operating Profit in Q2 of last year had dipped sharply owing to steep commodity price increases and electronic component supply constraints and hence results of Q2 FY2022-23 are not strictly comparable with those of Q2 FY2021-22. The Company has been making simultaneous efforts in securing electronic components availability, cost reduction and improving realization from the market to better its margins. With this, the Net Profit for the quarter rose to INR 20,615 million from INR 4,753 million in Q2 FY2021-22. Coming to the Highlights of H1 (April-September), FY 2022-23 The Company sold a total of 985,326 units during the period. Sales in the domestic market stood at 852,694 units. Exports in this half year were at 132,632 units. During the same period previous year which is H1 FY2021-22, the Company registered a total sale of 733,155 units including 628,228 units in domestic market and 104,927 units in the export market. In addition to electronic components shortage, the sales in H1 FY2021-22 were also severely affected due to COVID related disruptions and hence results of H1 FY2022-23 cannot be compared with those of H1 FY2021-22. The Company registered Net Sales of INR 538,298 million in H1 FY2022-23, which is the highest-ever half-yearly Net Sales. The Net Sales in H1 FY2021-22 were at INR 360,965 million. The Company made a Net Profit of INR 30,743 million in the H1 FY2022-23 as against INR 9,161 million in H1 FY2021-22',
        "The Company registered Net Sales of INR 538,298 million in H1 FY2022-23, which is the highest-ever half-yearly Net Sales. The Net Sales in H1 FY2021-22 were at INR 360,965 million. The Company made a Net Profit of INR 30,743 million in the H1 FY2022-23 as against INR 9,161 million in H1 FY2021-22. We are now ready to take your questions, feedback and any other observations that you may have. Thank you. Moderator: Thank you very much. We will now begin the question and answer session. We have the first question from the line of Kumar Rakesh from BNP Paribas. Please go ahead. Kumar Rakesh: My first question was around realization. So, sequentially, we have seen an increase in the realization by about 2%. Now, this is quite noteworthy given that in the context of the volume mix which we had during the quarter, mini and compact segments mix had increased while UVs MSIL Conference Call Transcript 28th October 2022 4 | P a g e and export mix or the volume mix was lower. And also discounting in September quarter usually is higher than what happens in the June quarter. So, despite all of this, we have seen an increase in realization. So, can you please help us understand that what led to this realization increase? Ajay Seth: Sequentially, there is an improvement in realization and this is attributed to again the mix because while we had lauched the new Alto, the price point of the old Alto and new Alto were different. So, that's one part. Second also, I think the proportion of the Brezza and other high- end vehicles were higher compared to the first quarter, which led to this higher realization. Also the fact that we had taken a price increase in the first quarter, which was partial in first quarter and fully absorbed in the second quarter. So, that also had its impact. Discounts are more or less same in the 2 quarters. It’s marginally higher in this quarter compared to first quarter, not very different",
        "So, that's one part. Second also, I think the proportion of the Brezza and other high- end vehicles were higher compared to the first quarter, which led to this higher realization. Also the fact that we had taken a price increase in the first quarter, which was partial in first quarter and fully absorbed in the second quarter. So, that also had its impact. Discounts are more or less same in the 2 quarters. It’s marginally higher in this quarter compared to first quarter, not very different. Kumar Rakesh: My second question was how to look at now at the installed capacity that we have access to at Maruti given that we’ll also have access to Toyota’s capacity, so what number we should be looking at the installed capacity for us? Rahul Bharti: So, as of now, we have about 22.5 lakh capacity at Haryana plus Gujarat. Of course, production at Karnataka is over and above this. And in times to come, we are in process of working on the Kharkhoda plant, which will be up and running in the year 2025. And if required, I think most likely we might have to add about 1 lakh capacity on a short term basis in Manesar to meet intermediate demand. Manesar 1 lakh might come by April ’24 and Kharkhoda in the subsequent year. Moderator: We have our next question from the line of Pramod Kumar from UBS. Please go ahead. Pramod Kumar: And just on the opening comments, you talked about future SUV launches to dominate the segment, like how you dominate the other categories, which is quite heartening because your current SUV market share, SUV plus MPV market share is 17 percentage points. So, if you can just help us understand between you or Rahul san as to what are the plans here because the understanding is that it's a pretty competitive segment with very well entrenched models and Maruti is kind of coming late in the category",
        "So, if you can just help us understand between you or Rahul san as to what are the plans here because the understanding is that it's a pretty competitive segment with very well entrenched models and Maruti is kind of coming late in the category. You're talking about dominance, but even if it's significant market share what are plans and how do you get there, sir? Ajay Seth: So, Pramod, I think let the excitement carry on some more time because we have said that we are commited to address this SUV segment. And therefore, we have mentioned that there will be more launches in these segments. But as you're aware that we don't give any details of the products, product plans and as such, there should be some excitement which will be visible to you as you saw in Grand Vitara, maybe soon you will see more excitement in the newer launches that we will have. But definitely, we are committed to the SUV segment, which will not only help us address the growing segment but also help us address the market share loss that we've had in the past. MSIL Conference Call Transcript 28th October 2022 5 | P a g e Pramod Kumar: And then sir, just related to this, generally the automotive thumb rule is that the pricing of a product goes higher, the profitability is generally better, of course, subject to scale. And SUVs are significantly more pricier than comparable products in every category. So, is that understanding right that as you make this pivot from a hatchback less portfolio to a higher price SUV segment, there is no reason why you should be kind of compromising your profitability, right, when you make the switch and transition. Ajay Seth: See, profitability is all dependent on what is your ability of pricing a product at a given point in time. And in the past with portfolio being the smaller cars and we were not present in the SUV segment, still our profitability was reasonably good",
        "Ajay Seth: See, profitability is all dependent on what is your ability of pricing a product at a given point in time. And in the past with portfolio being the smaller cars and we were not present in the SUV segment, still our profitability was reasonably good. I think it'll be a combination of what the market can absorb, where you can price your product and also, when the product matures over a period and as you localize and cost goes down, things change in that interim period. So, it will be a combination of many factors. So, giving an answer to that would be very complicated at this point in time. Pramod Kumar: Sir, and the last question then is on the financial, on the expenditure side, we have seen that it kind of outpaced the revenue growth, quarter-on-quarter, other expenditures, revenue growth. So, what is driving that sir, and if you can just throw more light on the sustainable number there, and even your employee expense has seen a reasonable jump. So, if you can just help us understand these 2 better, sir. Ajay Seth: So, in sequential, one thing that's built in and this is also other expenses is royalty. And with the volume going up, the royalty also as an absolute value goes up. And so there is an impact of that which is increased from Q1 to Q2, that's about INR 150 crore. Then there is increase in the advertisement and marketing costs. And as you are aware that we’ve had launches, and also we mentioned in the previous call as well that we will not be shying away from investing in marketing spend because that gives us a much longer visibility. So, that's gone up by another INR 150 crore. And also the manufacturing expenses have gone up because of the significant rise in the energy prices, the power and fuel costs have significantly gone up. Also, certain activities that we were scaling down earlier, and in a normal situation, we've restarted that. So, there's an increase on that account as well. So, these are broadly the heads where it's gone up",
        "So, that's gone up by another INR 150 crore. And also the manufacturing expenses have gone up because of the significant rise in the energy prices, the power and fuel costs have significantly gone up. Also, certain activities that we were scaling down earlier, and in a normal situation, we've restarted that. So, there's an increase on that account as well. So, these are broadly the heads where it's gone up. There is a small increase in other heads, including the employee costs, which is a normal increase that you have on account of the normal increments, etc, that happens during the year. But other than that, I think there is no other factor of increase at this point in time. Pramod Kumar: And would you expect the marketing intensity to continue like this? Or you would expect some bit of normalization or even on the royalty side, is there any launch related royalty pay off one- off when a new model is introduced? Ajay Seth: No. So, there is no launch related royalty that we pay. Royalty is basically linked to sales. And it will be based on the same formula that we have mentioned to you in the past. So, there will be no change as far as that is concerned. Marketing spend will depend on many factors. There is a kind of visibility that we need for the new models, the kind of visibility that we need for existing MSIL Conference Call Transcript 28th October 2022 6 | P a g e brands and existing models. And also, as you're aware, we mentioned that we'll be bringing in more new models, so obviously the spend will remain stepped up. Moderator: We have our next question from the line of Amyn Pirani from JPMorgan. Please go ahead. Amyn Pirani: Sir, just to go back to the question on discounts and royalty, can you mention the discount per vehicle number as well as the royalty number for the quarter? I know you answered the question directionally, but can you give us the numbers also",
        "And also, as you're aware, we mentioned that we'll be bringing in more new models, so obviously the spend will remain stepped up. Moderator: We have our next question from the line of Amyn Pirani from JPMorgan. Please go ahead. Amyn Pirani: Sir, just to go back to the question on discounts and royalty, can you mention the discount per vehicle number as well as the royalty number for the quarter? I know you answered the question directionally, but can you give us the numbers also. Ajay Seth: So, discounts in this quarter were at INR 13,840 per vehicle, and they were at INR 12,748 in the first quarter. So, they are about INR 1,000 higher than the first quarter. They were obviously much higher in the second quarter of last year. They were at about INR 18,500 in the second quarter of last year. So, that was on discount. The royalty percentage last year was at 3.5%, now it is at 3.8%. And the first quarter royalty was slightly lower than this which was between 3.6% and 3.7%. Amyn Pirani: And secondly, on your CapEx, I see that in your cash flow, I think you've already spent I think INR 3,500 crore on CapEx for the first half. So, can you help us understand what's the full year expectation number and what are the areas in which these spends are going? Is most of it going towards the Haryana CapEx? Or is there some other areas where you're spending this money? Ajay Seth: So, we will be spending upwards of INR 7,000 crore this year. And this includes of course, the Kharkhoda facilities where now we've started our construction work. And also we’ll have to place orders to various vendors. So, that will be one major portion of CapEx. Besides that, all the new model launches that we are doing where we have to have the investment on toolings, et cetera, I think that will be another large piece of CapEx. So, these are two areas where the CapEx will be maximum",
        "And this includes of course, the Kharkhoda facilities where now we've started our construction work. And also we’ll have to place orders to various vendors. So, that will be one major portion of CapEx. Besides that, all the new model launches that we are doing where we have to have the investment on toolings, et cetera, I think that will be another large piece of CapEx. So, these are two areas where the CapEx will be maximum. Then you have the other routine capital expenditure on the other aspects of the business, which is R&D, the regular maintenance CapEx. So, these are the key areas where we will be spending. Moderator: We have our next question from the line of Raghunandhan from Emkay Global. Please go ahead. Raghunandhan: Firstly, order book is huge at 4.1 lakh as of the end of September and new products are 1.3 lakh for the remaining portion which is large at 2.8 lakh. Can you indicate, which are the major models? Rahul Bharti: So, it's a mix, but mostly we have seen Ertiga has a high waitlist and anecdotally also you keep getting requests for early allotment. Of course, the new models we have discussed, the Baleno also has a high number and then the other models mostly equally spread. Raghunandhan: And CNG will be 130,000, 140,000 units? Rahul Bharti: Approximately Yes. MSIL Conference Call Transcript 28th October 2022 7 | P a g e Raghunandhan: Sir, given the strong response for hybrid, there is a scope for launch of hybrids and other existing model. What are the thoughts here? And typically, what is the timeline required for introducing a new powertrain in existing model? Rahul Bharti: Yes. So, we are also happy that the strong hybrid is getting a good response. In the Grand Vitara, more than 35% of the total bookings that we have today are of the strong hybrid. This may be slightly premature to conclude and so we'll watch this as it comes. And we'll try to look at other options in other models also",
        "What are the thoughts here? And typically, what is the timeline required for introducing a new powertrain in existing model? Rahul Bharti: Yes. So, we are also happy that the strong hybrid is getting a good response. In the Grand Vitara, more than 35% of the total bookings that we have today are of the strong hybrid. This may be slightly premature to conclude and so we'll watch this as it comes. And we'll try to look at other options in other models also. Raghunandhan: Lastly, the gross margin has improved 150 basis point quarter-on-quarter. So, can you indicate what would be the contribution of GP by depreciations and commodity benefits for Q2? Ajay Seth: So, sequentially, there has been a benefit on account of commodities because commodities have come off. And also the element of normal cost reduction that we do. Even on the exchange rates, we have gained there because the JPY depreciation has been steep during the quarter. So, there are combinations of factors this time now which are all positive. So, one, as I said commodities, second, I mentioned about regular cost reduction that we do and the JPY impact, overall impact of the currency depreciation. So, all put together, you see a combination of these three are impacting the gross margins to improve by what you've seen. Raghunandhan: And how do you see the commodity benefits going forward? Ajay Seth: Commodity benefits going forward is difficult to predict. Certain commodities have cooled off and certain commodities are higher than the earlier period. So, it's a combination. For example, anything related to oil, energy, et cetera is still expensive, where we've been shelling out more money than before. But things like steel and precious metals have shown improvement. Now, we will have to wait and watch in terms of how the future moves, I think it will at least remain steady in the third quarter. But the indication given by our supply chain is that there could be slight inching up in the fourth quarter",
        "So, it's a combination. For example, anything related to oil, energy, et cetera is still expensive, where we've been shelling out more money than before. But things like steel and precious metals have shown improvement. Now, we will have to wait and watch in terms of how the future moves, I think it will at least remain steady in the third quarter. But the indication given by our supply chain is that there could be slight inching up in the fourth quarter. Moderator: Thank you. We have our next question from the line of Chandramouli Muthiah from Goldman Sachs. Please go ahead. Chandramouli Muthiah: And thank you for taking my question. The first question is on the Grand Vitara profitability. Could you maybe share any additional color on how long you think this product could take to reach sort of corporate average EBIT margin? So, from a percentage margin perspective, maybe see scope for this product to reach the corporate average EBIT margin over time or is the outsource manufacturing arrangement likely to sort of continue a shared margin structure with Toyota. Rahul Bharti: So, we do not comment on individual segment or individual product margins as such. But the largest benefit was that it's a premium offering in the SUV space. And what we are excited about is that a fair percentage of the bookings are in the higher variants. And this is both for Grand Vitara and for the Brezza. So, a very good percentage of the bookings are from the upper or the MSIL Conference Call Transcript 28th October 2022 8 | P a g e top variants. So, that is positive. And once we have volumes and we have presence in these segments, profit automatically follows. Chandramouli Muthiah: My second question is on the semiconductor situation. So, despite still a bit of a nagging impact of semiconductor shortages, we seem to have hit sort of record production in volumes this quarter",
        "So, a very good percentage of the bookings are from the upper or the MSIL Conference Call Transcript 28th October 2022 8 | P a g e top variants. So, that is positive. And once we have volumes and we have presence in these segments, profit automatically follows. Chandramouli Muthiah: My second question is on the semiconductor situation. So, despite still a bit of a nagging impact of semiconductor shortages, we seem to have hit sort of record production in volumes this quarter. So, just trying to understand the few units that you are not able to produce this quarter, what is the typical model mix there? Is it on more premium vehicles or some CNG vehicles? Any color there will be very helpful. Rahul Bharti: No, it's not like that. It's basically most of the constraint is coming from one electronics part manufacturer and of course, it is in some specific models. So, hopefully, going forward, we hope that the situation eases, though it is very difficult to predict. Chandramouli Muthiah: And lastly, I just have a housekeeping question if you could maybe just give us the numbers on spare sales and export revenues for the quarter. Rahul Bharti: So, export revenue was about INR 3,400 crore for the quarter. And spares, generally, we do not have a separate disclosure. Moderator: We have our next question from the line of Jinesh Gandhi from MOFSL. Please go ahead. Jinesh Gandhi: A couple of questions from my side. First of all, are we seeing any material impact on CNG demand given the substantial price increases which we have seen and how do we see that segment considering the price differential now? Rahul Bharti: So, fortunately, till now, no, but there is a cause of concern, because of the high prices and we have represented to the government on this. But we are informed that in the commercial vehicle space, there has been an impact. So, CNG for us, this quarter was more than 20% penetration",
        'First of all, are we seeing any material impact on CNG demand given the substantial price increases which we have seen and how do we see that segment considering the price differential now? Rahul Bharti: So, fortunately, till now, no, but there is a cause of concern, because of the high prices and we have represented to the government on this. But we are informed that in the commercial vehicle space, there has been an impact. So, CNG for us, this quarter was more than 20% penetration. But we are engaging with the government to rein in the prices because this has nothing to do with Indian cost. It is only linked to a global index, which has a force majeure kind of situation. Jinesh Gandhi: Right. And similarly, are we seeing any impact on the export demand given that many of the end export markets are witnessing challenges on currency and similar macro pressures. So, obviously pressure in demand and this thing. Rahul Bharti: So, fortunately, nothing so far. But we are watching the situation. Jinesh Gandhi: And can you share retail sales in the current quarter and volumes in Gujarat? Rahul Bharti: Actually, this is a continuous period starting from the first Navratra till end of December. Some models are in transit, some we have stocked up for some particular orders. but we are expecting that by end of December we’ll be able to sell a lot of models and keep our closing stock low. Jinesh Gandhi: Sir, my question was retail sales for 2Q FY ’23. MSIL Conference Call Transcript 28th October 2022 9 | P a g e Rahul Bharti: Yes. Because part of it was in the festive period, so that’s what. So, it is better to club till end of December which means club it till Q3. Then have a view. Jinesh Gandhi: And Gujarat production, would that be around that similar 31%-32% range or has gone up? Rahul Bharti: About 31%. We did about 162,000 units from Gujarat, SMG. Moderator: We have our next question from the line of Pramod Amthe from InCred Capital. Please go ahead',
        'Because part of it was in the festive period, so that’s what. So, it is better to club till end of December which means club it till Q3. Then have a view. Jinesh Gandhi: And Gujarat production, would that be around that similar 31%-32% range or has gone up? Rahul Bharti: About 31%. We did about 162,000 units from Gujarat, SMG. Moderator: We have our next question from the line of Pramod Amthe from InCred Capital. Please go ahead. Pramod Amthe: Continuing on that CNG question, how has the mix of fleet versus personal bias changed in last 2 years? Can you give some color? Rahul Bharti: Fleet versus personal buyers? Pramod Amthe: Yes. Rahul Bharti: We have a very good response from the personal buyers within CNG. But generally what we have seen for example, the Ertiga is a very hot seller. So, that kind of impression that we are getting is that Wagon R and Ertiga, Ertiga is more than I think 2/3rd is CNG. Wagon R also has a high traction. So, the models with a higher boot space, they are going very well on CNG. Pramod Amthe: You mean the commercial is relatively higher in these 2 segments. I was looking for more commercials. Rahul Bharti: No, it is not linked with commercial segment. So, if you have a bigger model and bigger boot space, the CNG acceptance is far higher. Pramod Amthe: The reason why I ask you is that there is some pressure on demand in the commercial segment. Rahul Bharti: I’m so sorry. When I said that I meant trucks. Commercial does not mean taxis. It meant trucks. So, when we discuss within SIAM, the commercial vehicle manufacturer, the trucks, they are concerned about it. Not in passenger vehicles. Pramod Amthe: And what’s the industry for vehicles mix, mix of personal and commercial for industry for the car set for CNG. Pramod Amthe: Yes. sorry. I was saying for car industry, CNG segment, what’s the mix of fleet and personal? Rahul Bharti: I’ll have to get back with the figure, not readily available',
        'Commercial does not mean taxis. It meant trucks. So, when we discuss within SIAM, the commercial vehicle manufacturer, the trucks, they are concerned about it. Not in passenger vehicles. Pramod Amthe: And what’s the industry for vehicles mix, mix of personal and commercial for industry for the car set for CNG. Pramod Amthe: Yes. sorry. I was saying for car industry, CNG segment, what’s the mix of fleet and personal? Rahul Bharti: I’ll have to get back with the figure, not readily available. But there’s a fair amount of spread across all models, and even for example in models like Wagon R, we have a good level of penetration and Ertiga has very high level of penetration. Dzire Tour are obviously because in many places, it is mandated that they need to run on CNG, so we have 88% penetration. Even the normal Dzire has about 35% penetration, the non-taxi Dzire. MSIL Conference Call Transcript 28th October 2022 10 | P a g e Pramod Amthe: Second one is with regard to the strong hybrid which you have launched. Can you give more details in terms of your cell or battery sourcing. What’s the type of localization you have in that and sustenance of the current pricing, do you have more visibility on that considering that rupee has depreciated and local content versus important content in those cells or battery? Rahul Bharti: See, pricing is always dynamic. We keep watching the market, it’s a new product. And of course, we are very consumer centric. So, we’ll keep taking a view on the market on a regular basis. As you rightly mentioned the factors can change and if we get any kind of cost reduction along the way normally, we do consider it. Pramod Amthe: And any indication on localization there current and how you look at going forward, the cell? Rahul Bharti: So, it is being manufactured at in Karnataka. So, the local content will also depend on our OEM partner. Moderator: We have our next question from the line of Kapil Singh from Nomura. Please go ahead',
        'As you rightly mentioned the factors can change and if we get any kind of cost reduction along the way normally, we do consider it. Pramod Amthe: And any indication on localization there current and how you look at going forward, the cell? Rahul Bharti: So, it is being manufactured at in Karnataka. So, the local content will also depend on our OEM partner. Moderator: We have our next question from the line of Kapil Singh from Nomura. Please go ahead. Kapil Singh: Firstly, I just wanted to check on overall growth, what you are expecting for the full year and given the situation and supply constraints, do you expect Maruti to do better than industry in this financial year? Rahul Bharti: So, of course, your answer is linked to the supply of semiconductors. So, given whatever we get we should be able to produce and send to the market. Industry is expected to do about 3.8 million this year. Kapil Singh: The question was just trying to understand that order book is pretty high but the production has not matched the order book. So, if you could just help us understand why that is happening? Inventory has also increased. So, what is the technical issue here that we are facing? Rahul Bharti: See, in the festive months, we do stock Besides, given the total semiconductor supplies, you can maximize your production if you keep a slightly longer term view. So, we are keeping a view till let’s say end of December by which time we should be able to get both wholesales and retails at a higher level given the overall semiconductor constraints. The idea is to maximize within the constraint available if we improve the timeframe a bitt. Kapil Singh: And secondly I just wanted to check given your experience with strong hybrid and the kind of demand you are seeing, are you looking to add more models with strong hybrid option? Rahul Bharti: Slightly premature. Yes, obviously over a longer period of time, that would be the intent',
        'The idea is to maximize within the constraint available if we improve the timeframe a bitt. Kapil Singh: And secondly I just wanted to check given your experience with strong hybrid and the kind of demand you are seeing, are you looking to add more models with strong hybrid option? Rahul Bharti: Slightly premature. Yes, obviously over a longer period of time, that would be the intent. But we will get more feedback from the consumer and from our manufacturing experience, obviously, the efforts will be in that direction. And because it helps majorly in CO2 reduction also, so nothing that we can immediately offer to comment on, nothing specific, but that would be the direction in the future. Moderator: We have our next question from the line of Arvind Sharma from Citi. Please go ahead. MSIL Conference Call Transcript 28th October 2022 11 | P a g e Arvind Sharma: Sir, first question would be on the capacity expansion. You did mention something like a lakh at Manesar and further at the new plant. Is it possible to share some timeline about the net capacity expansion, especially in the new plant, how much will it add? I believe it would be in lieu of something which will go away at Gurgaon. So, what will be the net capacity addition at the new plant? Rahul Bharti: We are not looking at any kind of reduction in Gurgaon, in fact, at least in the shorter term, we might have to increase production in Gurgaon. Kharkhoda plant, all plants, generally there the optimum economic size is about 2.5 lakh per annum. Our first plant should be commissioned by the first quarter of calendar ’25. And I think we already have to start thinking on the second plant if demand growth continues in India. we are not looking at any kind of reduction in Gurgaon. Arvind Sharma: This 2.5 lakh would be in addition to current and add to it 1 lakh at Manesar, right? Rahul Bharti: Yes',
        'Kharkhoda plant, all plants, generally there the optimum economic size is about 2.5 lakh per annum. Our first plant should be commissioned by the first quarter of calendar ’25. And I think we already have to start thinking on the second plant if demand growth continues in India. we are not looking at any kind of reduction in Gurgaon. Arvind Sharma: This 2.5 lakh would be in addition to current and add to it 1 lakh at Manesar, right? Rahul Bharti: Yes. Arvind Sharma: Sir, second question more for the current quarter, what are the FX gains? Is it possible to quantify the FX gains and where do they reflect? And also as corollary, what is the import content both for your production in Gujarat and Haryana? What are the import content for these 2 plants in these 2 locations and the FX gains this quarter? Ajay Seth: Import content for both the plants would be similar. There is no difference because all the procurement is more or less on the same basis. So, our total direct import content is about 4%. So, it will fall in that category only both the plants. Similarly, I think the other thing is the indirect import content which also would be in a similar category because the vendors are common and the material that we have buying from the vendors are basically similar vendors. So, there is no difference in terms of import content. In terms of forex, so between different currencies, there have been gains. Largely on the import and export side, on the dollar-rupee exposure, we are naturally hedged. So, we use a natural hedge route. On the dollar-yen, there have been maximum gains in this quarter compared to last quarter and also last year because of the significant depreciation of the currency. And the net impact of exchange rate, have been about INR 158 crore of gain in this quarter compared to first quarter of this year. Arvind Sharma: This is the entire gain on the P&L, INR 158 crore? Ajay Seth: That’s right. And they will be under different heads',
        'So, we use a natural hedge route. On the dollar-yen, there have been maximum gains in this quarter compared to last quarter and also last year because of the significant depreciation of the currency. And the net impact of exchange rate, have been about INR 158 crore of gain in this quarter compared to first quarter of this year. Arvind Sharma: This is the entire gain on the P&L, INR 158 crore? Ajay Seth: That’s right. And they will be under different heads. Arvind Sharma: Sir, you said direct import content of around 4%, what would be the indirect import content, if you could share? Ajay Seth: Between 10% and 11%. Moderator: We have our next question from the line of Chirag Shah from Edelweiss. Please go ahead. MSIL Conference Call Transcript 28th October 2022 12 | P a g e Chirag Shah: Sir, one very specific question on the SUV strategy. are you looking to enter the compact car category because some of your peers for example Tata Punch are trying to address that category with a SUV type or SUV feel model. is there a part of your strategy the SUV launches that you’re indicating? And if you can elaborate on it, it would be helpful. Rahul Bharti: Sorry, you mentioned SUV or XEV? Chirag Shah: SUV type of a product like Tata Punch, is a compact SUV, addressing that Baleno kind of a range in terms of price point. Rahul Bharti: So, as Mr. Seth mentioned some time ago, let’s keep the excitement and let’s bring our products which deliver pleasant surprises to the customers. So, you’ll have to wait for sometime. Chirag Shah: And sir, second questions was on the strong hybrid, is there any restriction or technological limitation on the size of the vehicle to add strong hybrids? How much lower you can go in terms of technology today, in terms of price of the vehicle? Rahul Bharti: So, you are right. Strong hybrids at the moment we have solutions in slightly bigger cars which have room and the engine room to accommodate both the powetrains',
        'So, you’ll have to wait for sometime. Chirag Shah: And sir, second questions was on the strong hybrid, is there any restriction or technological limitation on the size of the vehicle to add strong hybrids? How much lower you can go in terms of technology today, in terms of price of the vehicle? Rahul Bharti: So, you are right. Strong hybrids at the moment we have solutions in slightly bigger cars which have room and the engine room to accommodate both the powetrains. It becomes a bit of a challenge to bring them in smaller cars. But that is what Suzuki’s competence is all about. So, we’ll watch the market and to reduce carbon we have to adopt a portfolio of technologies and each technology, each model will have its own context, its cost, its volume. So, it’s a complex equation that we keep working on all the time. we’ll keep watching how we can maximize hybrid volumes in the future. Moderator: Thank you. Ladies and gentlemen, that was the last question for today. And with this we conclude today’s conference call. On behalf of Maruti Suzuki India Limited, we thank you for joining us and you may now disconnect your lines. Ajay Seth: Thank you. Rahul Bharti: Thank you. MSIL Conference Call Transcript 28th October 2022 13 | P a g e'
    ]
    final_summary= get_final_output(maruti_chunks)
    print("Final summary")
    print(final_summary)
        

    main()
