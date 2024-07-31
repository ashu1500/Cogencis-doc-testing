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
    ntpc_chunks=[
        "He is the Government Nominee Director on the Board of NTPC. Shri Upadhyaya is an IAS Officer of 1989 batch and has served for more than 33 years in various capacities in the State and Central Government. As Joint Secretary (Coal), he was instrumental in developing systems by applying space technology to curb the menace of illegal mining. On left of Shri Upadhyaya, we have Mr. Ujjwal Kanti Bhattacharya, our Director (Projects), who joined NTPC in the year 1984. In his illustrious career, he has significantly contributed for NTPC's vertical and horizontal business diversification, as well as growth through inorganic root. As Director (Projects), he is responsible for commissioning the projects and adding to our growth multiple. On left of Shri Bhattacharya, we have Shri Shivam Srivastava, who has recently joined our Board as Director (Fuel). Shri Shivam Srivastava joined NTPC in the year 1988. He has over 34 years of experience with outstanding contribution in areas of fuel handling, fuel management, safety, plant operation and maintenance, and in coal mining projects. As Director (Fuel), he is responsible for ensuring fuel availability, affordability, and security for generating stations, along with development and safe operations of captive coal mines of NTPC. Page 1 of 23 NTPC Limited July 31, 2023 On the right of our CMD, we have Shri Ramesh Babu V, who is our Director (Operations). He joined NTPC in 1987. He has over 35 years of experience with outstanding contribution in management of large size plants in the area of power plant operations and maintenance, renovation and modernization of old units and in the area of efficiency and system improvement of thermal plants. As Director (Operations), he is responsible for overall planning for safe, reliable and efficient operation of all the power stations of NTPC while ensuring environmental and safety compliances",
        'He joined NTPC in 1987. He has over 35 years of experience with outstanding contribution in management of large size plants in the area of power plant operations and maintenance, renovation and modernization of old units and in the area of efficiency and system improvement of thermal plants. As Director (Operations), he is responsible for overall planning for safe, reliable and efficient operation of all the power stations of NTPC while ensuring environmental and safety compliances. On the right of Shri Ramesh Babu, we have Shri Jaikumar Srinivasan, our Director (Finance), who has more than three decades of illustrious career behind him in power and mining sectors in both state and central PSUs in the field of finance, accounts, taxation, commercial, electricity regulation, renewables, IT, and project development, with nine years of Board level experience. He is also responsible for the commercial function of the company. His relentless efforts and dynamism have resulted in further consolidation of our financial as well as commercial position. With this, I would now request our CMD to begin with his opening remarks. Thereafter, Director (Finance) will make a presentation about NTPC, and we will have an interactive session after that. CMD Sir. CMD: Good evening. All the colleagues of the Board including our Special Secretary, Shri Upadhyaya Ji, Shri Bhattacharya Ji, Shri Srivastava Ji, Mr. Ramesh Babu, Shri Jaikumar Srinivasan Ji and other senior colleagues present here. So, what I would like to do that I will be briefly touching on a few of the subjects and then there will be a presentation, which will be made in detail by Director (Finance) and then we can spend a little more time for some of the questions and answers. The presentations or the information as you will be knowing that it is mostly available across. So, I would request all of you to raise the some of the concerns or the issues what you believe is very important for the investors and analysts',
        "So, what I would like to do that I will be briefly touching on a few of the subjects and then there will be a presentation, which will be made in detail by Director (Finance) and then we can spend a little more time for some of the questions and answers. The presentations or the information as you will be knowing that it is mostly available across. So, I would request all of you to raise the some of the concerns or the issues what you believe is very important for the investors and analysts. With that let me just say a few things on this. So let me welcome to all of you on this 19th Annual Analysts and Investors Meet. We had been consistently doing this in end of July or beginning of August and break was during the COVID period. Fortunately, it is behind us, nobody is talking about that, and we all are back. I hope that everybody is healthy and happy and we are looking at something, which is now the growth trajectory and as you will also would have seen the financial year 22-23 has witnessed a very unprecedented growth and there was some kind of uncertainties, which was encountered specifically on the fuel side worldwide and you will be knowing that many countries including Europe and I don't even talk about our neighbors, they have gone through a very tough time but we as country has managed the energy quite well and in that context, the credit goes to the government. Your Company has played a very, very vital role in this time of very kind of unprecedented increase suddenly after this pent-up demand, which was coming from the post-COVID. The good point is that we are still maintaining and this month also, it is almost around close to double digit growth, which is a very, very kind of promising thing. I think, this is going to be really providing enough opportunities for the growth",
        'Your Company has played a very, very vital role in this time of very kind of unprecedented increase suddenly after this pent-up demand, which was coming from the post-COVID. The good point is that we are still maintaining and this month also, it is almost around close to double digit growth, which is a very, very kind of promising thing. I think, this is going to be really providing enough opportunities for the growth. Page 2 of 23 NTPC Limited July 31, 2023 Our diligent project management strategies and the operational efficiency, contributed to the healthy EPS, which registered a growth of approximately 5.6% over the previous year. We have been carefully monitoring the situation and taking corrective actions continuously to ensure the energy security for the country by ensuring reliable fuel supply and this will be coming from our captive mines or sourcing the coal from other sources. Currently all our coal stations have sufficient stock. We have more than 16 days stock and due to the diligent monitoring of the situation, none of our stations were starved because of the coal in the last year. Our stock has been performing quite well and it has hit 52 weeks high, and you people can tell far better than probably anybody else. For FY 22-23, the Board of Directors have recommended a final dividend of Rs.3 per share, subject to the approval of our shareholders at the upcoming AGM, this final dividend is in addition to the interim dividend already paid at Rs.4.25 per share. So, put together it will become Rs.7.25 per share. This year marks the 30th consecutive year of dividend distribution showcasing our steadfast commitment to providing value to our esteemed shareholders. Regarding NTPC performance and the way forward, there will be detailed presentation as I just mentioned. But let me add, as I said earlier, a few points. We have fast-tracked our energy transition plans and made significant progress on the renewables',
        "4.25 per share. So, put together it will become Rs.7.25 per share. This year marks the 30th consecutive year of dividend distribution showcasing our steadfast commitment to providing value to our esteemed shareholders. Regarding NTPC performance and the way forward, there will be detailed presentation as I just mentioned. But let me add, as I said earlier, a few points. We have fast-tracked our energy transition plans and made significant progress on the renewables. There may be some kind of questions, which keep coming that, how much we are going to add this year, how much we are going to add next year, and we have the CEO of our NTPC Green Energy Limited, Mr. Mohit Bhargava, who is present here. So maybe, we will give a chance to him if it is required and some of the questions are coming that straightway can be addressed. And let me assure you that the progress is very, very promising. We will get into that detail. I am happy to inform you that for the first time our organic non-fossil capacity addition has surpassed the fossil capacity addition, showcasing our commendable progress in the energy transition journey. We have also finalized several new renewable and storage contracts and as a result, our total renewable pipeline has reached 20 GW. Additionally, we have made several new partnerships in the renewable segment including the C&I consumers and that is going to be not a very small 5 MW, 10 MW, it is going to be on a gigawatt scale. We take great pleasure in sharing our continuous success in securing the renewable bids, cementing our position as a formidable player in the India's renewable market. In the Financial year 22-23, we have achieved the capacity addition of 3,292 MW with a commercial capacity addition of 3,952 MW including the acquisition of Jhabua, which was 600 MW. This included significant contribution of 1,352 MW from the renewable sources and the first ever overseas capacity of 660 MW in Bangladesh",
        "We take great pleasure in sharing our continuous success in securing the renewable bids, cementing our position as a formidable player in the India's renewable market. In the Financial year 22-23, we have achieved the capacity addition of 3,292 MW with a commercial capacity addition of 3,952 MW including the acquisition of Jhabua, which was 600 MW. This included significant contribution of 1,352 MW from the renewable sources and the first ever overseas capacity of 660 MW in Bangladesh. Further, we have added 770 MW in this quarter, Q1 FY24, elevating the NTPC group’s total installed capacity to 73,024 MW. We are declaring the commercial operation of Barh Unit-2, 660 MW from midnight today. As you will be knowing, that Barh was something where we had to terminate the contract in between and we had to really take the kind of Make in India and our own engineers have Page 3 of 23 NTPC Limited July 31, 2023 contributed to commission these units. So, the first unit is running successfully, second unit is getting commercial operation and the next one is going to be next year. We have recorded an all-time high-power generation of 399 billion units, that means in a closing distance from 400 billion units, registering a growth of 11%. Further, eight of our coal stations figured in the top 25 best performing stations of the country in terms of the PLF and our coal- based station registered a PLF of around 75% against the country's 64%. Let me take a second here and try to clarify, I have no hesitation to respond afterwards also, but going forward let us not harp on the PLF. PLF is not that important as the availability and the reliability is, because the PLF is something, which was the barometer for the shortage situation. The most important is that the power should be available on demand. So, during the daytime when the solar is going to be in plenty, most of the power stations will be getting back down, which is going to be a good thing",
        'PLF is not that important as the availability and the reliability is, because the PLF is something, which was the barometer for the shortage situation. The most important is that the power should be available on demand. So, during the daytime when the solar is going to be in plenty, most of the power stations will be getting back down, which is going to be a good thing. And we would also like to do that because of our blending policy of the Ministry of Power, so that we can also reduce our emissions in that. So, I just took this opportunity to explain that the PLF should not be the only criteria. The availability and the reliability are much more important in this. These numbers demonstrate our best-in-class asset management practices and capability of human resources, and we are completing almost 40 years for the first unit of Korba unit now. We have already completed almost 42 years for Singrauli. We are completing almost around 40 years, and these are operating at about 100% PLF anyway. We have successfully commissioned a blending project that incorporates the green hydrogen with PNG, piped natural gas. Further, the green hydrogen mobility in Leh and Delhi are slated to be commissioned in the current financial year. Just to let you know that this green hydrogen which is blended with the PNG is coming from the floating solar plant in Kawas. So, it is completely different. The floating solar plant is feeding the electrolyser to produce the hydrogen and that hydrogen is being blended. So that may be a very, very small, tiny kind of experiment but that will pave way for a lot of opportunities going forward. On the Leh-Ladakh, we were very close to have the commissioning or rather starting the operation of our hydrogen powered bus for the mobility, but floods have caused some problems in the transportation. So maybe, it will take another 15-20 days though the bus is going to be delivered on 10th of August',
        'So that may be a very, very small, tiny kind of experiment but that will pave way for a lot of opportunities going forward. On the Leh-Ladakh, we were very close to have the commissioning or rather starting the operation of our hydrogen powered bus for the mobility, but floods have caused some problems in the transportation. So maybe, it will take another 15-20 days though the bus is going to be delivered on 10th of August. Another important highlight is the exceptional growth in the coal production. So, one way is that what we are talking about this transition and mobility. On the other side, this fuel security is the coal production from captive mines, which reached 23.2 million tons during FY 22-23 making a growth of 65%. Further, the Q1 of the current financial year, we have registered a growth of 100% in the coal production as compared to previous financial year. So that is quite reassuring on the fuel security side that we are putting all efforts. We have set an ambitious coal production target of 34 million tons in this current financial year. Our bill realization of the financial year 22-23 reached Rs.1,54,356 crore and no doubt that, we have achieved the 100% realization on that. Page 4 of 23 NTPC Limited July 31, 2023 We have demonstrated MSW to green charcoal plant at Banaras and now we are taking in Noida and Bhopal and Hubli and maybe, that will be another one which will be for those who had the keen interest in the ESG that it is not only for our Company, but we are trying to help the society as a whole and the country in this aspect. So, coming to ESG, we are diligently following ESG principles, timely responding to any ESG related queries by all investors and the ESG rating analysis. We also regularly update our website with the ESG related disclosures. We are consistently expanding our scope by incorporating additional sustainability standards and the ESG frameworks',
        'So, coming to ESG, we are diligently following ESG principles, timely responding to any ESG related queries by all investors and the ESG rating analysis. We also regularly update our website with the ESG related disclosures. We are consistently expanding our scope by incorporating additional sustainability standards and the ESG frameworks. I am happy to share with you that, our ESG rating by Sustainalytics has improved by one band in the last fiscal. In financial year 22-23, we achieved a specific water consumption of 2.69 litre per kilowatt hour, which is a very good improvement. Additionally, the commencement of our first air-cooled condenser at North Karanpura plant is expected to save around 75% of the water compared to the conventional water-cooled condenser. Our ash utilization percentage has increased to 83% in financial year 22-23, reflecting our commitment to the sustainable waste management practices. We have also embraced the principle of life, lifestyle for environment, as promoted by our country on a global scale and have implemented numerous campaigns and the awareness program across our business units. On the CSR fronts, we have spent approximately Rs.353 crore on the CSR activities and we have another flagship project in this called Girl Empowerment Mission, where we bring the young girls, they are our guests at our townships for a month and that is a life-changing experience for all of them. Almost around 2,000 to 2,500 girls every year, we are able to help them in their life journey. We are actively pursuing the just transition and prioritizing the re-skilling of the workforce and ensuring that, the transition to the clean energy is accompanied by the opportunities for the career development, safeguarding the livelihood of all stakeholders. We recognize the importance of upholding the trust and the confidence placed in us by our stakeholders',
        "Almost around 2,000 to 2,500 girls every year, we are able to help them in their life journey. We are actively pursuing the just transition and prioritizing the re-skilling of the workforce and ensuring that, the transition to the clean energy is accompanied by the opportunities for the career development, safeguarding the livelihood of all stakeholders. We recognize the importance of upholding the trust and the confidence placed in us by our stakeholders. By practicing sound governance principles and consistently fostering a culture of integrity, we aim to ensure that, we operate in a manner that aligns with the best interest of all stakeholders. I am very happy to inform you that, our pioneering efforts in India's clean energy transition and emphasis on the training have garnered global recognition through the prestigious accolades including the highly esteemed global awards presented by S&P Platts and ATD Best. These coveted awards represent international global recognition of our sustainable practices and our relentless pursuit of excellence in the clean energy domain. Let me touch on something on the bright future ahead. The Indian power sector is undergoing a significant transformation driven by the government policies. Ongoing power sector reforms address the challenges and create the conducive environment for growth. With the economic activities gaining further momentum leading to increased energy demand, as the largest utility in the country, we will play a crucial role in meeting the growing power requirement. We would Page 5 of 23 NTPC Limited July 31, 2023 like to maintain our share in the power sector by supplying almost around one-fourth of the total electricity what we are doing. And we will try to see that if we can really move up on that. Accordingly, we have set ambitious target for the capacity addition to the tune of 6 GW in the fiscal year 23-24",
        'We would Page 5 of 23 NTPC Limited July 31, 2023 like to maintain our share in the power sector by supplying almost around one-fourth of the total electricity what we are doing. And we will try to see that if we can really move up on that. Accordingly, we have set ambitious target for the capacity addition to the tune of 6 GW in the fiscal year 23-24. As informed last year, while putting utmost thrust on the renewable capacity addition, we are also considering the construction of 7 GW of additional coal-based capacity as Brownfield projects. The commissioning of these capacities is planned in a phased manner with the target time frame set till 2030. As you will be knowing that we had already started the work on Talcher last year. This year we are going to start in Lara and further we will be working on Sipat. Our focus is on completing existing projects, fast-tracking new ones, and operating our fleet efficiently and reliably. As all remaining projects are either pithead or situated on the mine, with the commissioning of these projects our generation share shall see a further rise. Simultaneously, we are adding renewable capacity aggressively and exploring the opportunities in the green hydrogen, on the nuclear front, on the small modular reactors, green charcoal, carbon capture and utilization, green chemicals. Green chemical is basically the project what is going on at present and in it we will capture the carbon dioxide from the stack and utilizing the green hydrogen and making the green methanol. We are also working on the PMC contracts through the International Solar Alliance. Additionally, two nuclear power projects namely Chutka and Mahi Banswara through ASHVINI, a joint venture between NTPC and NPCIL with an aggregate capacity of 4,200 MW are being considered for implementation',
        'Green chemical is basically the project what is going on at present and in it we will capture the carbon dioxide from the stack and utilizing the green hydrogen and making the green methanol. We are also working on the PMC contracts through the International Solar Alliance. Additionally, two nuclear power projects namely Chutka and Mahi Banswara through ASHVINI, a joint venture between NTPC and NPCIL with an aggregate capacity of 4,200 MW are being considered for implementation. You would have seen some news item that we have already entered into joint venture agreement with NPCIL and these two projects which NPCIL was taking up will be transferred to this joint venture. We are also working on the various storage solutions including the large-scale PSPs. Two of our subsidiaries THDC and NEEPCO have been allocated almost around 7,000 to 8,000 MW, new Greenfield hydro projects in Arunachal and Assam. To facilitate our ambitious growth plan, we are actively collaborating with the start-ups, innovators, manufacturers, commercial and industrial consumers as well as the leading institutes. We are also collaborating with the various state governments for securing land patches. We are creating a future ready workforce through the rescaling, redeploying and hiring talent to meet the evolving business needs. These initiatives position us for the success in the dynamic energy landscape while contributing to the sustainable and the prosperous future for all. I would like to place on record that we have annulled the process of identifying a strategic investor for NTPC Green Energy Limited due to certain issues and are now working on the strategies for the IPO which will unlock further value. We have started working on the CERC regulations and we are engaging actively with the CERC at present for ongoing discussions on 24-29 Regulations',
        "I would like to place on record that we have annulled the process of identifying a strategic investor for NTPC Green Energy Limited due to certain issues and are now working on the strategies for the IPO which will unlock further value. We have started working on the CERC regulations and we are engaging actively with the CERC at present for ongoing discussions on 24-29 Regulations. So, I still would have missed many things, but I think after the presentation we will be more than happy to cover whatever would have been of your interest. With this I would like to assure all Page 6 of 23 of you that we will continue to put tireless efforts for maximizing profitability in a sustainable manner and bring value to the shareholders. Thank you. Let me request Director (Finance) to make the presentation. NTPC Limited July 31, 2023 Director (Finance): Thank you CMD Sir and good afternoon once again. I welcome all the investors and analysts to this annual meet of NTPC. NTPC, as all of you know, has been the largest power generator in the country and a crucial energy enabler, holds a pivotal position in India's growth story and energy transition. As I share the vision and mission statement on the screen, I also wish to outline the steps we are taking to achieve this vision. Our vision revolves around being the mainstay of fulfilling the energy needs of the country and being a key driver of the ambitious economic growth story of India and going on to transforming itself into a global energy company. This vision is founded on the recognition of emerging energy trends and the abundant opportunity they present. Currently, we stand as India's largest and the most efficient power company with a determined trajectory towards becoming a leading player in the global energy landscape. As an essential constituent in implementing of government of India's ambitious plans, we play a crucial role in creating robust and modern power infrastructure for the new India",
        "This vision is founded on the recognition of emerging energy trends and the abundant opportunity they present. Currently, we stand as India's largest and the most efficient power company with a determined trajectory towards becoming a leading player in the global energy landscape. As an essential constituent in implementing of government of India's ambitious plans, we play a crucial role in creating robust and modern power infrastructure for the new India. Our vision will be realized through close adherence to our core values which we refer to as the ICOMIT, together with our capabilities to produce & offer dependable power and related solutions in the most economical, efficient and environment friendly manner. The NTPC group presently operates projects spanning the length and the breadth of our country. Our wide presence throughout the nation's landscape enables us to spread and mitigate risk associated with operating within a limited geographical territory. Our operating stations comprise units of various sizes including those in collaboration with our JV partners and subsidiaries. A key advantage of our coal-based plants is their predominantly proximity to the fuel sources resulting in substantially lower cost and lower energy charges. The outline of the presentation, in our presentation I will give a broad overview of NTPC and then delve into the following key points. Our strategies, initiatives and roadmap towards energy transition, renewable energy and fostering sustainable practices. I will highlight the various ESG initiatives undertaken by NTPC as a responsible corporate citizen aimed at overall well-being of the people and environment, demonstrating our commitment to achieving growth in a sustainable and affordable manner. I will be discussing the factors and the key imperatives contributing to our growth and our operational and project execution excellence along with our robust financial performance",
        "I will highlight the various ESG initiatives undertaken by NTPC as a responsible corporate citizen aimed at overall well-being of the people and environment, demonstrating our commitment to achieving growth in a sustainable and affordable manner. I will be discussing the factors and the key imperatives contributing to our growth and our operational and project execution excellence along with our robust financial performance. As you can see on the screen, as the largest power generator of the country, NTPC plays a significant role generating an impressive 25% of the nation's electricity with a share of just 17% of the installed capacity. Our corporate plan outlines a clear growth path aiming to become a 130 GW plus company by 2032. During FY23 we added around 4 GW of commercial capacity. We are right on the path with over 17 GW capacity currently under construction and a further 18 GW capacity in the Page 7 of 23 NTPC Limited July 31, 2023 planning stage. We have maintained consistent operational excellence and a lead in terms of availability factor and PLF. Financial year 23 marked a year of milestone for us in the sense that we achieved the highest ever generation, profit, and revenue realization. As part of our energy transition, commitment, NTPC is aiming for 60 GW of RE capacity by 2032 and also exploring business opportunities throughout the clean energy value chain. With a clear focus on ESG, we are consistently progressing on all the defined KPIs. With our key strength and our project execution capabilities, we are spearheading the energy transition while powering the growth of new India. We have plans to add 10 GW of conventional and 16 GW of renewable energy in the next three years. As part of our overall energy security plans, we are actively considering awarding thermal capacity of 7.2 GW within the next year. Furthermore, to have a greater fuel security, we are enhancing our coal mining capacity as well",
        "With our key strength and our project execution capabilities, we are spearheading the energy transition while powering the growth of new India. We have plans to add 10 GW of conventional and 16 GW of renewable energy in the next three years. As part of our overall energy security plans, we are actively considering awarding thermal capacity of 7.2 GW within the next year. Furthermore, to have a greater fuel security, we are enhancing our coal mining capacity as well. Our concern and determination for a cleaner environment is evident in our plan to complete the implementation of the FGD, which is the flue gas desulphurization units in our entire operational and under construction capacity within the next three years. Our group has secured definitive tie-ups for 10 GW of renewable capacity with commercial and industrial consumers, and we are in discussion with many others. With these strong indicators of growth and expansion, we are confident that we will not only fulfill expectations, but also set new benchmarks in the industry. Going to sustainable energy transition, our vision for 2032 revolves around transforming our energy portfolio, moving away from being a predominantly power generating company to being a diversified energy major and portfolios of clean, green, affordable power for our beneficiary with presence across the value chain. To achieve this vision, we aim to contribute significantly to development of a green ecosystem and establish ourselves as the leader in this domain. NTPC Group is all set to diversify into newer areas of clean energy comprising nuclear power, green hydrogen and chemicals, carbon capture and utilization and waste to wealth initiatives. The culmination of these strategies will significantly broaden our company's revenue stream, through an integrated energy business with power generation at the core",
        "NTPC Group is all set to diversify into newer areas of clean energy comprising nuclear power, green hydrogen and chemicals, carbon capture and utilization and waste to wealth initiatives. The culmination of these strategies will significantly broaden our company's revenue stream, through an integrated energy business with power generation at the core. Our growth and diversification strategies will be backed by our operation and project management excellence attained over the last five decades, aggressive addition to RE capacity through ultra-mega renewable energy power parks and other organic and inorganic modes, gainfully utilizing existing land banks and infrastructure and power and allied industry with strong financials and ratings, and the ability to raise funds competitively, we are confident to fuel our growth. Coming to specifics of renewable energy, as you can see on the screen, in alignment with the Government of India's focus on renewable energy, we have set an ambitious target of achieving 60 GW of renewable energy by 2032. Currently our group has an installed renewable capacity of 3.3 GW with an additional 5.9 GW under construction. Furthermore, we have secured tenders Page 8 of 23 NTPC Limited July 31, 2023 and bilateral tie-ups for another 10.8 GW of renewable capacity, creating a visible pipeline of 20 GW in the near term. In FY23, we made a quantum jump, doubling the RE generation and adding the highest ever renewable energy capacity. The commissioning of RE capacity surpassed that of conventional capacity during this period, clearly demonstrating our commitment to the energy transition. To support our aggressive RE capacity addition strategy, we are actively planning and implementing a cumulative capacity of 36 GW in different states through the Ultra Mega Renewable Energy Park scheme",
        "In FY23, we made a quantum jump, doubling the RE generation and adding the highest ever renewable energy capacity. The commissioning of RE capacity surpassed that of conventional capacity during this period, clearly demonstrating our commitment to the energy transition. To support our aggressive RE capacity addition strategy, we are actively planning and implementing a cumulative capacity of 36 GW in different states through the Ultra Mega Renewable Energy Park scheme. Additionally, we are in discussion with state governments regarding the implementation of pump storage projects, further contributing to the expansion of RE infrastructure. With a clear RE project pipeline, we are well on track to achieve our ambitious goal. To realize the Government of India's efforts to a carbon neutral economy, NTPC is leading efforts in green hydrogen, green chemicals, carbon capture and utilization and other related fields in the entire clean energy value chain. In this regard, NTPC has commissioned first green hydrogen blending into piped natural gas and is building the pilot projects for synthesizing green methanol and ethanol and also setting up first green hydrogen mobility project. We have also entered into an MoU with Indian Army for setting up green hydrogen projects in its establishments. This agreement ushers in a new era in defense and power collaboration. We have also signed various agreements and MOUs for developing green chemicals and green fuel. Company has also conceptualized setting up a green hydrogen hub near Vishakhapatnam in Andhra Pradesh. As can be seen here, we are steadfast in our approach towards developing the entire green ecosystem. Coming to the sustainability initiatives. Sustainability has emerged both an imperative and a challenge for global energy companies balancing of huge energy demand with environmental concerns",
        'We have also signed various agreements and MOUs for developing green chemicals and green fuel. Company has also conceptualized setting up a green hydrogen hub near Vishakhapatnam in Andhra Pradesh. As can be seen here, we are steadfast in our approach towards developing the entire green ecosystem. Coming to the sustainability initiatives. Sustainability has emerged both an imperative and a challenge for global energy companies balancing of huge energy demand with environmental concerns. NTPC has formulated a comprehensive sustainable energy strategy called as Brighter Plan that includes well-defined key performance indicators and targets. To further our commitment to sustainability, we are actively working on developing a net zero roadmap for NTPC in collaboration with NITI Aayog. To enhance our ESG performance, we engage in regular dialogue with ESG rating agencies. Our environmental conservation efforts are exemplified by the plantation of 38 million trees in and around NTPC projects, creating a significant carbon sink to offset emissions. We are also developing a mega eco-park in the national capital on the site of our decommissioned thermal plant. Water conservation remains a priority for us as evidenced by our continuous reduction in specific water consumption over the last four years. Furthermore, we have commissioned the first air-cooled condenser in North Karanpura power plant, resulting in significant saving on water. Through our concerted efforts and relentless commitment to sustainability, NTPC will be serving as a model for responsible and cleaner energy production. Page 9 of 23 NTPC Limited July 31, 2023 “Going higher on generation and lowering greenhouse gas intensity” remains our motto for environment management and drives our efforts to comply with the new environmental norms. As the leader in the industry, we have taken significant steps to control SOx and NOx',
        "Through our concerted efforts and relentless commitment to sustainability, NTPC will be serving as a model for responsible and cleaner energy production. Page 9 of 23 NTPC Limited July 31, 2023 “Going higher on generation and lowering greenhouse gas intensity” remains our motto for environment management and drives our efforts to comply with the new environmental norms. As the leader in the industry, we have taken significant steps to control SOx and NOx. Over the next three years, we plan to commission FGD systems for our entire operational and under construction capacity, ensuring a substantial reduction in SOx emissions. Moreover, for NOx control, we have successfully implemented combustion modification in 19 GW capacity resulting in a remarkable 30% reduction in NOx emissions. Beyond emission control, we are actively undertaking various blue-sky initiatives such as desalination of seawater at Simhadri, biomass co-firing and waste to charcoal projects at Varanasi and other cities as depicted here. Moving to new energy technologies on the research and development side, we are among select energy utilities globally to have a dedicated technology development center called NETRA. NETRA's core focus area encompasses cutting edge technologies crucial for the energy sector. In addition to driving technology development, NETRA also provides scientific support to NTPC stations facilitating operational excellence and efficiency. To ensure high quality research, a research advisory council has been established comprising eminent scientists and experts from India and abroad, guiding NETRA's high-end research endeavors. Furthermore, NTPC has doubled its R&D expenditure in the last five years, underscoring its dedication to remain at forefront of technological advancement in the energy sector",
        "To ensure high quality research, a research advisory council has been established comprising eminent scientists and experts from India and abroad, guiding NETRA's high-end research endeavors. Furthermore, NTPC has doubled its R&D expenditure in the last five years, underscoring its dedication to remain at forefront of technological advancement in the energy sector. Coming to our CSR initiatives, we have been steadfast in our commitment to corporate social responsibilities by continuously allocating 2% or more of our net profit towards CSR activities since many years. During FY23 the company dedicated a sum of Rs.353 crore towards various CSR initiatives, focusing primarily on health, sanitation, safe drinking water and education, etc. Additionally, Company has a special focus on girl empowerment, striving to empower girls in the project vicinities and make them self-reliant and confident in all aspects of life. Furthermore, NTPC is contributing to the development of archery, a sport from the grass root level by providing support to the Archery Association of India. Through its CSR initiatives, NTPC has made significant impact on the lives of approximately 16 lakh people positively influencing human development in remote locations. Now, we turn to the key growth pointers for the power sector. India's GDP is expected to grow at a robust pace in the coming years and the energy demand is expected to move in tandem with the economy. India's demographic strength coupled with vast latent demand for electricity is poised to play a significant role in driving annual incremental growth in power sector. Projections by Central Electricity Authority further support the notion of substantial growth in the sector. One significant achievement in recent times is that every Indian now has access to electricity. This milestone signifies a major stride towards universal electrification and sets the stage for the power sector in India to thrive",
        "Projections by Central Electricity Authority further support the notion of substantial growth in the sector. One significant achievement in recent times is that every Indian now has access to electricity. This milestone signifies a major stride towards universal electrification and sets the stage for the power sector in India to thrive. On the screen, we have some key indicators that the power sector in India is poised for significant transformation driven by the Government's focus on achieving affordable and uninterrupted 24 by 7 power for all. Page 10 of 23 NTPC Limited July 31, 2023 This emphasis on providing reliable electricity to citizens has sparked changes across every aspect of power sector value chain. Key indicators in the power sector i.e., capacity generation and per capita projection are expected to experience substantial growth as can be seen on the screen. Share of renewable energy capacity in the country, total installed capacity is expected to increase substantially from 40% at present to more than 66% over the next decade. Comprehensive reforms and the emphasis on renewable energy are anticipated to drive substantial growth in the power sector. NTPC has operational capacity of 73 GW, and another 17 GW is under various stages of construction already. A further 39 GW is under planning and feasibility stage. NTPC's size and capabilities coupled with its operational excellence forms the major strength which will drive the company's growth engine to achieve the ambitious goal of becoming a 130 GW plus company by 2032. Furthermore, the strategic decision to have projects under construction at diverse locations plays a crucial role in reducing the overall execution risk. These key strategies and prudent measures demonstrate our commitment to efficient and sustainable growth. Coming to the financials, our multiple revenue stream is set to grow steadily, fueled by the substantial capacity addition we have undertaken",
        'Furthermore, the strategic decision to have projects under construction at diverse locations plays a crucial role in reducing the overall execution risk. These key strategies and prudent measures demonstrate our commitment to efficient and sustainable growth. Coming to the financials, our multiple revenue stream is set to grow steadily, fueled by the substantial capacity addition we have undertaken. Similarly, we expect our regulated equity to grow at a double-digit rate in near term, driven by projects already under implementation, both in conventional, mining, and RE business and also those which we have planned further. Under the guidance of an experienced and focused leadership team, we are confident in our ability to deliver up to the expectations. Coming to the coal mining sector, our coal mining group has an impressive asset portfolio, comprising eight coal mines with estimated geological reserves of 5 billion tons and an ultimate mining capacity of 77 million tons per annum. The company has demonstrated remarkable growth in coal production, producing 23.2 million tons of coal in financial year 23, which represents a steep 65% year-on-year growth. Looking ahead, we have set an ambitious target of producing 34 million ton of coal in financial year 24, indicating our commitment for fuel security. In line with this goal, the company has achieved a record high first quarter production of 8.59 million tons in Q1 of FY24. Some of our other business development endeavors, can be seen on screen. In response to the dynamic times and imperatives of diversification and adaptation, we are actively pursuing new business opportunities both domestically and globally, expanding into commercial and industrial market, we have formed joint ventures and signed MoU/ bilateral agreement to supply power to C&I customers. Additionally, we have ventured into nuclear power with 4.2 GW capacity projects under active consideration in Rajasthan and MP',
        "In response to the dynamic times and imperatives of diversification and adaptation, we are actively pursuing new business opportunities both domestically and globally, expanding into commercial and industrial market, we have formed joint ventures and signed MoU/ bilateral agreement to supply power to C&I customers. Additionally, we have ventured into nuclear power with 4.2 GW capacity projects under active consideration in Rajasthan and MP. We are also exploring small modular reactor nuclear technology, which showcase our commitment to innovation. Internationally, we have achieved significant milestone including the commissioning of our first overseas power unit in Bangladesh and in being appointed as project management consultant for 6.5 GW solar project in Latin America and Africa. Moreover, we are collaborating with various state governments in Page 11 of 23 NTPC Limited July 31, 2023 India for developing power projects, this will help us to position us in the global energy landscape. Turning to our operational excellence, NTPC's operational capabilities have been consistently proven through an unmatched track record of maximizing efficiency in the power sector. The Company's coal stations have demonstrated exceptional operation, clocking a plant load factor of 75.9% in financial year 2023, much above the All India PLF of 64.2%. Moreover, NTPC achieved its highest ever group generation of 399 billion units in financial year 23, reflecting a notable 11% growth in generation compared to the previous year. NTPC's operational efficiency is underpinned by robust systems and best maintenance practices, ensuring the smooth functioning of our power plants and the optimal utilization of resources. NTPC's proactive approach to safety has led to enhancing of safety standards at our power plants and the complete integration of Safety-First culture within the organization",
        "NTPC's operational efficiency is underpinned by robust systems and best maintenance practices, ensuring the smooth functioning of our power plants and the optimal utilization of resources. NTPC's proactive approach to safety has led to enhancing of safety standards at our power plants and the complete integration of Safety-First culture within the organization. This safety- conscious approach helps prevent accidents and ensure a secure working environment for our workforce. Coming to the long-term fuel security. We have taken proactive and successful initiatives to ensure fuel security for our current as well as future capacity requirements. Aggregation of annual contracted quantity on CIL subsidiary level basis has resulted in several benefits such as optimum utilization of coal, avoidance of fixed charges loss and efficient outage planning and stock management. We have signed long-term fuel supply agreements with both CIL and Singareni Collieries Company Limited for reliable supply of coal. We also source coal through bridge linkages, captive mines and e-auctions, further diversifying our sources of coal procurement. To address any shortage of domestic coal, NTPC has imported 15 million metric tons of coal during financial year 23, ensuring an uninterrupted and sufficient coal supply to meet our energy demands. Our consistent focus on being the low-cost power producer has enabled us to maintain a high merit order for our power plants. This advantageous position translates to better PLF and operational efficiency, ensuring that our power plant remains competitive and operate at optimal level. Although, as CMD sir was mentioning, PLF will be of less significance, whereas availability factor or the declared capacity will be the better hallmark or pointer of efficiency going ahead",
        "Our consistent focus on being the low-cost power producer has enabled us to maintain a high merit order for our power plants. This advantageous position translates to better PLF and operational efficiency, ensuring that our power plant remains competitive and operate at optimal level. Although, as CMD sir was mentioning, PLF will be of less significance, whereas availability factor or the declared capacity will be the better hallmark or pointer of efficiency going ahead. The company's elaborate payment security mechanism has proven highly effective in managing the receivables and ensuring timely and reliable payments from customer, thereby achieving highest ever realization of more than Rs.1.54 lakh crore during financial year 23. Furthermore, the fact that our trade receivables are back at pre-COVID levels indicates the company's resilience and ability to manage financial changes effectively even in face of adverse circumstances. Coming on to NTPC’s HR Vision, our Company has “People First” approach towards employees. We believe in continuous development of our employees through objective and open Page 12 of 23 NTPC Limited July 31, 2023 performance management system. We provide comprehensive training to familiarize our employee with technological advances and up to date operational and management practices. Our key employee performance metrics like sales per employee, value-added per employee, profit per employee and man megawatt ratio has shown consistent improvement. NTPC continues to win all round laurels in various fields in operational, quality, HR, CSR, safety, etc. We are proud of building a high trust, high performance culture. Turning to financials, as can be seen from the results projected on the screen, NTPC's financial performance has been remarkable, exhibiting sustained revenue growth and robust profits level over the years. In financial year 23, the company achieved its highest ever profit of Rs",
        "NTPC continues to win all round laurels in various fields in operational, quality, HR, CSR, safety, etc. We are proud of building a high trust, high performance culture. Turning to financials, as can be seen from the results projected on the screen, NTPC's financial performance has been remarkable, exhibiting sustained revenue growth and robust profits level over the years. In financial year 23, the company achieved its highest ever profit of Rs.17,197 crore, showcasing our strong financial management and operational efficiency. The momentum has continued in financial year 24. The company posted strong financial results for Q1FY24. The profit after tax for Q1 stood at Rs.4,066 crore, reflecting a substantial 9% growth compared to the Q1 of the last year. This financial achievement underscores our ability to adapt to changing market conditions, maintain operational excellence and leveraging our diverse portfolio of power generation assets to sustain growth. Our consolidated financials have exhibited consistent growth driven by strategic investments in value, accretive joint ventures and subsidiaries. The company's performance in FY23 saw significant growth in dividends from JVs and subsidiaries with a remarkable increase of 35% year-on-year. This indicates the success of our investment in these ventures which have contributed to the company's overall financial strength and performance. The NTPC Group's EBITDA also experienced a double digit-growth crossing the notable milestone of Rs.50,000 crore in financial year 23. This substantial EBITDA growth signifies our ability to generate strong operational earnings and manage our financials efficiently. With a proactive approach to investment, operational excellence and a strong financial foundation, we are well positioned to leverage our opportunities in power sector and other related domains, ensuring continued growth and value creation for our stakeholders",
        "50,000 crore in financial year 23. This substantial EBITDA growth signifies our ability to generate strong operational earnings and manage our financials efficiently. With a proactive approach to investment, operational excellence and a strong financial foundation, we are well positioned to leverage our opportunities in power sector and other related domains, ensuring continued growth and value creation for our stakeholders. Our balance sheet size has been growing bigger and stronger demonstrating the company's financial strength and stability. Over the last three years our gross fixed assets have increased by an impressive 43% to Rs.3,38,436 crore showcasing NTPC's commitment to expanding power generation capacity and infrastructure. Concurrently, the capital work-in-progress has decreased by 9% to Rs.89,133 crore, which signifies a successful effort in unlocking capital and turnover of investment into completed assets. Looking ahead, we anticipate continued growth. The turnaround from CWIP to completed assets is expected to be expedited further, particularly with a greater mix of renewable energy projects in the pipeline. Our ability to raise debt at competitive rates from the market enhances our financial flexibility and capacity to fund our expansion plans effectively. With strong financials, NTPC has consistently paid dividends to our shareholders for the past 30 years. The company maintains a dividend policy that balances dividend payouts with the deployment of funds for future growth initiatives. In conclusion, as the leading power generation company, we are well positioned to drive India's energy transition and contribute significantly to the nation's growth and development. Thank you all of you for your attention and patience too. I now hand over to Aditya Dar for further proceedings. Page 13 of 23 NTPC Limited July 31, 2023 ED (Finance): Thank you sir. We will now have an interaction with our Board. Atul Tiwari: Yes, sir",
        "In conclusion, as the leading power generation company, we are well positioned to drive India's energy transition and contribute significantly to the nation's growth and development. Thank you all of you for your attention and patience too. I now hand over to Aditya Dar for further proceedings. Page 13 of 23 NTPC Limited July 31, 2023 ED (Finance): Thank you sir. We will now have an interaction with our Board. Atul Tiwari: Yes, sir. I am Atul Tiwari from Citi Research. Just two questions. Sir, do you think that over the next four-five years, India will have peak power deficit again, given the dynamics in the sector today? And if that happens, will that increase the opportunity set for NTPC to set up more coal- based plants beyond the 7.2 GW that you have planned? That is my first question. I have one more. Okay, so let me ask the second one. So, you did refer to IPO of the green energy company. So, should we conclude that the strategic investor induction is off the table and now IPO is the only plan or are you exploring both these alternatives in parallel? CMD: Okay. Next. Analyst: Hello sir. What is the logic of separating out the coal business into a 100% subsidiary especially when it is entirely captive for us and there are no commercial sales outside the company? That's my question. CMD: Who was that? Just please, okay. Thank you. Analyst: You talked about IPO, and you say, this particular meeting is for stakeholders also. So, what I am looking here, those who hold shares in NTPC, they must get a right in the proportionate to their holding in green IPO. That is my first thing. And second thing, sir, in your speech you said that pump storage helps for renewable. That is absolutely not clear. If you can do that, that will be better. Another thing what I am seeing is here, in my whole career, hydrogen, ammonia, methane, all these are classified as toxic gases. Hydrogen was the most dangerous",
        'So, what I am looking here, those who hold shares in NTPC, they must get a right in the proportionate to their holding in green IPO. That is my first thing. And second thing, sir, in your speech you said that pump storage helps for renewable. That is absolutely not clear. If you can do that, that will be better. Another thing what I am seeing is here, in my whole career, hydrogen, ammonia, methane, all these are classified as toxic gases. Hydrogen was the most dangerous. And how suddenly, during the last five years, all these gases became safe to operate, use, that is not very clear, sir. Another thing, if we talk about saving in water and other things so that we can reduce the cost of the electricity. But at the same time, we are seeing that the demand is increasing. To reduce the demand, how we can say that, how we can use better equipment so that the energy can be saved. So, what I am looking here is BLDC, to what extent we can use BLDC technology, so that, because BLDC technology saves two-third percent of the energy. So, whether it is possible to use to the higher capacity MW, KW rating, something like that. That I would like to know. And another thing, a small question, and one thing, great thing about you is that you have got only 17% capacity, but you meet 25% requirement, so whether it is a fault of others or whether it is your efficiency, that depends. Thank you, sir. CMD: I think you had put multiple questions. Anyway, we will try to respond. Yes, please. Girish Achhipalia: Hi, sir. Girish from Morgan Stanley. Just three questions. One, on award timelines for the coal- based capacity, can you just suggest how this will get ordered in the next few quarters? Second one was this long-term 8 GW on hydro you mentioned THDC plus NEEPCO',
        "Thank you, sir. CMD: I think you had put multiple questions. Anyway, we will try to respond. Yes, please. Girish Achhipalia: Hi, sir. Girish from Morgan Stanley. Just three questions. One, on award timelines for the coal- based capacity, can you just suggest how this will get ordered in the next few quarters? Second one was this long-term 8 GW on hydro you mentioned THDC plus NEEPCO. Can you help us like what's the ballpark capex that we should assume and how much time does it normally take and how we should kind of think through in slightly longer term in terms of commissioning of this capacity? Page 14 of 23 NTPC Limited July 31, 2023 And finally, just the data point on C&I, out of the 20 GW right now, what is the portfolio in C&I right now? And are these firm commitments in terms of PPA? Rohit: So, this is Rohit from Antique Stock Broking. Sir, my first question is on that the country is planning 40 GW of solar and 10 GW of wind, so what is the solution from a macro standpoint that you see for evening base load and peak load demand? So, will you think gas based can see a revival or do you think lithium-ion battery could possibly be a solution or pumped hydro storage? What is the unitary economics that you have in mind which will be feasible? That's question number one. Two is to do with the SMRs, the modular reactor part. What is the capital outlay that you are planning on the nuclear front? Is there any concrete plan as such you have had in mind? Thank you. Analyst: Two questions. First is on the IPO. Do you have any target in mind which you want to achieve in terms of scale and size? Any MW term, and EBITDA terms? When you look for the IPO, is it FY24 or FY25 phenomena when you want to start the process? Sir, my second question is on the renewable side, the capital cost, and the cost of capital, both are equally important",
        "Analyst: Two questions. First is on the IPO. Do you have any target in mind which you want to achieve in terms of scale and size? Any MW term, and EBITDA terms? When you look for the IPO, is it FY24 or FY25 phenomena when you want to start the process? Sir, my second question is on the renewable side, the capital cost, and the cost of capital, both are equally important. So how are you going about ensuring that you are able to produce renewable energy at the lowest cost of capital in comparison with the private IPPs. My third question is on the NEP, the government is, so CEA has started targeting 27 GW of PSP, 47 GW of battery and 25 GW thermal. The cumulative number for the investment is Rs.3 trillion, right? And we are targeting only 60 GW. The numbers seem to be on the lower side. CMD: No, I mean it that you are thinking that this is on the lower side, so we are also a conservative company. So, we are trying to give the conservative figures. So, you can expect a little higher than that. Okay, let me try to attempt a few of the questions. Manish Bhandari: I have a question here. CMD: Yes, please go ahead. Manish Bhandari: Manish Bhandari from Vallum Capital. So, my question is related to the grid imbalance which has happened in many parts of the world because of different sources of energy getting pulled into the grid. So, is your renewed interest in the coal is because of the likelihood of the grid imbalance and this will be a thrust area, the revival of the coal by NTPC will be the thrust area? And maybe some more direction you can put on the grid imbalance, which is likely to come and which I've seen happening in China also? Thank you. Analyst: Sir, my question is we are planning a capex of or 3.3 GW of renewable capacity going to 60 GW by 2032",
        "So, is your renewed interest in the coal is because of the likelihood of the grid imbalance and this will be a thrust area, the revival of the coal by NTPC will be the thrust area? And maybe some more direction you can put on the grid imbalance, which is likely to come and which I've seen happening in China also? Thank you. Analyst: Sir, my question is we are planning a capex of or 3.3 GW of renewable capacity going to 60 GW by 2032. So, what is the cumulative capex that we have in mind from FY24 to FY32 in the renewable side? What I am interested in is the amount of money that we will spend over the next 8 to 10 years on the renewable side and what is the IRR that we have built in for this capex? Thank you. Page 15 of 23 NTPC Limited July 31, 2023 Analyst: Yes, sir. Sir, one question. The government and regulator wanted to implement this market- based economic dispatch from April 2022 with the NTPC's power projects as the first phase. So, what is the status there? When is it likely to be implemented? Thanks. CMD: Yes, please. You are the last? Okay, go ahead. Analyst: Sir, many years back I was in the BASF plant in Germany. They have a central dome like structure which they call an incinerator. So, the solid, liquid and gaseous wastes from all the 29 plants of BASF were sent into the incinerator. Will the usage of incinerator for emissions be helpful cost wise? Thank you. CMD: No. Atul Mehra: Sir, just one other question on NTPC Green IPO, just at the back. CMD: Who is that? Atul Mehra: Atul Mehra from Motilal Oswal Asset Management, sir. Would you be considering a full- fledged demerger of the business of NTPC Green, or it will continue to remain substantially held by NTPC Limited and there will be some minority which will be offloaded in the IPO? So what is the thought process about the IPO per se? Last year it was said that transportation of this hydrogen is not viable",
        'CMD: Who is that? Atul Mehra: Atul Mehra from Motilal Oswal Asset Management, sir. Would you be considering a full- fledged demerger of the business of NTPC Green, or it will continue to remain substantially held by NTPC Limited and there will be some minority which will be offloaded in the IPO? So what is the thought process about the IPO per se? Last year it was said that transportation of this hydrogen is not viable. So, are we in-house only developing those cylinders and all and developing or we are just going to pass on the hydrogen to those companies, sir? Thank you, sir. CMD: Okay, should I start attempting to? One more at the back. Koundinya: Yes, hi, thanks for the opportunity, sir. This is Koundinya from JPMorgan. So, sir, firstly on the renewable capacity addition, so you spoke about around 6 GW of capacity addition all put together for FY24 within which 4.6 GW is only the conventional. So just trying to understand, are there some bottlenecks that you are seeing with respect to capacity addition on the RE front at this point in time? And second thing, now that the CERC regulations are in each way being discussed for a next control period, and you did speak about lower PLFs with higher RE capacity addition. So, are there some discussions going on to compensate for loss of PLF incentives over there? And lastly, if you can speak about some the kind of returns that you are envisaging in the green initiatives or even on the nuclear front, if you can speak, provide some color on that. Thank you. CMD: Okay. Let me start attempting, and then I will request my colleagues to just respond to the few questions. I think the most prominent question is that this NGEL IPO, how it is, how much percentage, when it is, how much it is, and how the capex is going to be from the renewable energy? I think just simplicity and then I will ask Director (Finance) also to just chip in on that, whatever the missing link is, if there is any',
        'Thank you. CMD: Okay. Let me start attempting, and then I will request my colleagues to just respond to the few questions. I think the most prominent question is that this NGEL IPO, how it is, how much percentage, when it is, how much it is, and how the capex is going to be from the renewable energy? I think just simplicity and then I will ask Director (Finance) also to just chip in on that, whatever the missing link is, if there is any. So, the first thing is that we have just started the work. We will be very careful, and we will be watching the market before hitting the market. How much it is required? I think we will start with the minimum and then let us see that how much will be required because you all know that we are generating enough cash to fund our ongoing projects or our capex. So, it is not going to happen just tomorrow or day after. Unless Page 16 of 23 NTPC Limited July 31, 2023 the government is directing me to do it, then by this date it has to be done. So, I think this is what I hope that I am clarifying in a simplified way. So, I think, there can be n number of questions around that, but I think as of now this is what it is. So, there was a direction from the government for the monetization and it still exists. We had complied last year, and we will see that we will go on doing what is required. But as of now I can only say that we will try to realize the good value, otherwise it is not the time bound rather than it should be value based rather than the time bound. So, I hope that this all questions related to the NGEL IPO are more or less covered. The first question, who was the first one to just say four years, five years is going to be a peak shortage? Let me try to give my perspective and what, as NTPC, what we keep discussing. There had been quite a good amount of inventory which is I think getting exhausted and that is the reason that we have started just even looking at the coal-based assets in addition to our renewable',
        'So, I hope that this all questions related to the NGEL IPO are more or less covered. The first question, who was the first one to just say four years, five years is going to be a peak shortage? Let me try to give my perspective and what, as NTPC, what we keep discussing. There had been quite a good amount of inventory which is I think getting exhausted and that is the reason that we have started just even looking at the coal-based assets in addition to our renewable. You can rest assured that you keep consuming as much power as you can, provided you are paying for that, obviously. And rest assured that we will be taking care of those increase in the demand and how the projections are going to come, so without any hesitation. The 7 GW is not something which is new thing. This was already under kind of the planning or this kind of the drawing board and then what we are saying is these are the ongoing projects which we are taking. As per the CEA projections, I think we will be requiring around 250 - 252 GW of coal based thermal capacity on the outer side. There are different scenarios. This is the most stress scenario on that side. But nevertheless, if there is a more requirement, let us see our economy rather than getting at 7%, tomorrow we start growing at 10% and there is a further need, I think then we will go for the further capacity addition on that side. The idea is that power should be available, it should be available on demand, and it should be reliable, and it should be affordable. And it will be cleaner as we are going forward because of this SOx, NOx and other efficiency parameters, whatever it takes. So, whatever it takes, I think that is what we will have to meet the energy requirement, we have to meet the power requirement in the country. So, we are committed towards that, and the government takes the decision in the holistic way',
        'The idea is that power should be available, it should be available on demand, and it should be reliable, and it should be affordable. And it will be cleaner as we are going forward because of this SOx, NOx and other efficiency parameters, whatever it takes. So, whatever it takes, I think that is what we will have to meet the energy requirement, we have to meet the power requirement in the country. So, we are committed towards that, and the government takes the decision in the holistic way. It is not only we will be doing but we will be playing the leading role in whatever it comes as far as the power sector is concerned. I hope that this is there anything else what anybody would like to add on this side. Director (Finance): Talking about renewable business, now if you look at the overall picture which we said that 130 GW is the overall capacity and 60 GW, we are targeting in renewables. Of course, this target is with a little bit of redundancy considering the fact that, as we move ahead in the next 10 years, there could be surprises in one so that we always should have a latitude of pushing it on. But as far as the ballpark figure I would say is that an amount of Rs.40,000 to Rs.60,000 crore of equity would be required. And so, the IPO thing is not out of a compulsive need for money because the group’s cash flows are capable enough to take care of this equity thing. But it is more in terms of since the renewable side presents a tremendous opportunity for growth to capture this and Page 17 of 23 unlock value out of this growth initiative, we have this plan but however when exactly to do and you know in what way it is to be done is a matter of strategy as we go ahead. Thank you. CMD: Okay, so the next question was NML, NTPC Mining Limited. We are power generating NTPC Limited July 31, 2023 company mainly. We are transforming ourselves to energy, but mining is completely different than the power generation',
        "Thank you. CMD: Okay, so the next question was NML, NTPC Mining Limited. We are power generating NTPC Limited July 31, 2023 company mainly. We are transforming ourselves to energy, but mining is completely different than the power generation. So, the DNA of the company really suits to the requirement on that side. So, now what we are doing is that mines will go to NML and there will be a full-fledged subsidiary, which will be taking care of the mining. These will remain as captive coal blocks to us, there is no doubt. But just for information of all others, who might have missed, even today with the permission from the government, you can sell some of the coal from that after meeting your own requirement. NML is not the NCL. NML is NTPC Mining Limited. So, we will be also looking at some of the opportunities if it comes, on the other mining things. It can be tomorrow limestone, it can be tomorrow let's say lithium, tomorrow it can be anything, I am just saying from the kind of very low value items to the very high-cost items on that. So, and that's the reason, there is a lot of sense and the rationale to go for this mining company. And this will be working like a mining company, and it will have its own profit and loss and it will even have a few of the subsidiaries, like what Coal India has. Probably it will come somewhere in between. So this is what the idea is. So, though we had this work in progress, it has taken a little longer. We had finally convinced the government and thanks to the government that it has been agreed upon. First, it was recommended by power ministry and then it was with the coal ministry, and it has been now done deal on that side. So, now we have to do the actual transfer. So that is from the NML side. PSP, how it helps the renewable? I think this was the question to start with. PSP is pumped storage. So, it does not have its own kind of flowing water or reservoir, it is a kind of this thing. So, what is requirement? It's not necessary",
        "First, it was recommended by power ministry and then it was with the coal ministry, and it has been now done deal on that side. So, now we have to do the actual transfer. So that is from the NML side. PSP, how it helps the renewable? I think this was the question to start with. PSP is pumped storage. So, it does not have its own kind of flowing water or reservoir, it is a kind of this thing. So, what is requirement? It's not necessary. It's not that, it's not there. But it can be a kind of upper reservoir and lower reservoir and the water does not go anywhere. During the daytime, you can store, and you can utilize that excess energy and to pump the water from the lower reservoir to the higher reservoir and during the peak hours, you can extract that energy out of it. So basically, it is energy storage, and this is the best energy storage options for India as of today, I am just saying. And this is the least cost energy storage solution as of now. Then what else can it be? Hydrogen, methane, ammonia, these are the really future and we are aggressively working on this. Let me assure you that, we are not going to be left behind. Our teams are working on some of the demonstration projects at present, but they are also working aggressively for, it was mentioned some time back, that we will be also working on how to develop the hydrogen hub and our site at Pudimadaka, in case of Andhra Pradesh, where we have 1,200 acres of land. If you can imagine, we are trying to see that we can develop the hydrogen hub which will be a, which will be not only taking the hydrogen, green hydrogen and trying to convert into hydrogen and it will be taking the renewable energy and then converting it into the green hydrogen and green ammonia but also the other chemicals or maybe even the manufacturing of some of the electrolyzers and other things. We are not going to do on our own. By the way, we cannot get into manufacturing",
        'If you can imagine, we are trying to see that we can develop the hydrogen hub which will be a, which will be not only taking the hydrogen, green hydrogen and trying to convert into hydrogen and it will be taking the renewable energy and then converting it into the green hydrogen and green ammonia but also the other chemicals or maybe even the manufacturing of some of the electrolyzers and other things. We are not going to do on our own. By the way, we cannot get into manufacturing. This is a clear-cut decision as of today unless it is then there will be further things which is going to come on that. But hydrogen and ammonia Page 18 of 23 NTPC Limited July 31, 2023 are future, and we are working on that and there is a group, which is under NGEL which is working on that in all fronts. We have a general manager, which we appointed almost four years back to look after the green hydrogen and this is what are the developments, which are happening. We can have the further discussion whosoever is interested in this subject. And it may be also the green ammonia is one, but there is a green methanol which is also equivalent, it is equally important to take care of the green chemicals and green energy carrier on that side. I was in Goa, G20 last week. We had a very good meeting with the International Marine Organization, they are looking at both ammonia as well as green methanol. So, there are a number of opportunities, which are emerging. And on the lighter side, I think I might have told last time that NTPC someday will become national transport, power, and chemicals. So, we are not only thermal power company and there is a lot of synergies, which are coming and as the world moves, development happens. We will be having more and more opportunities to just work on those sides. Water saving, what was the question? Water saving is this is what we are going ahead with the air-cooled condenser',
        "And on the lighter side, I think I might have told last time that NTPC someday will become national transport, power, and chemicals. So, we are not only thermal power company and there is a lot of synergies, which are coming and as the world moves, development happens. We will be having more and more opportunities to just work on those sides. Water saving, what was the question? Water saving is this is what we are going ahead with the air-cooled condenser. Air cooled condenser does not require evaporation and that is how it is really saving a lot of water. And this is, what was mentioned, it really saves 75% compared to the normal conventional power. So, we have already commissioned in North Karanpura, and we will be doing that in our Patratu and going forward, wherever it is essential. Better equipment for energy saving, BLDCs, etc. this comes in the demand side, not the supply side. So, it's a power generation. But Bureau of Energy Efficiency is working a lot on that. Please visit that site and you will be coming to know that this LED, etc. this is started and there is an energy efficient fan. Now there are energy efficient air conditioner and what not on that side. There is a whole lot of kind of full-fledged scope on those side, how to conserve and how to reduce the energy consumption. The 17% capacity and 25% on the generation, we should give the credit to our engineers, starting from the designers to the operating personnel, that they are able to maintain the assets. As I just mentioned some time back, our Singrauli plant is 42 years old, and we are able to run it at 100%. Whereas there was some myth which was being talked about that coal-based power plant, the life is 25 years. So, and I just mentioned about our Korba, so then credit should go to how it is constructed, how it is maintained and how it is operated. Coal based capacity, Timeline, I just mentioned. I think this is up to 2030, which I think Director (Projects) will give",
        'As I just mentioned some time back, our Singrauli plant is 42 years old, and we are able to run it at 100%. Whereas there was some myth which was being talked about that coal-based power plant, the life is 25 years. So, and I just mentioned about our Korba, so then credit should go to how it is constructed, how it is maintained and how it is operated. Coal based capacity, Timeline, I just mentioned. I think this is up to 2030, which I think Director (Projects) will give. Director (Projects): Coal based capacity addition timeline, see in 2023-24 we will add 3600 MW. In 2024-25 we will be adding 3,580 MW and in 2025-26, we will be adding 1460 MW. Then TTPS Unit-1 660 MW will be commissioned in 2026-27. In 2027-28, we expect to have 2260 MW, Talcher Unit- 2, and Lara, both the units. In 2028-29, we expect 4,000 MW, which will be Sipat one unit, Darlipalli one unit, Singrauli, both the units, and Meja one unit. In 2029-30, will be completing the capacity addition with 1,600 MW capacity. Page 19 of 23 CMD: So, this is up to 2030, and what was suggested that, if economy goes further up and then there is NTPC Limited July 31, 2023 further requirement, we will have to start looking at something on that. Okay, the next question was the long-term or this something on the THDC and NEEPCO. I have the exact number with me that whatever we have got the letter for allotment, it is 6,291 MW, for THDC 2,950 and for NEEPCO 3,341. And there are further, this is in the kind of pipeline, so when I was saying around 8,000, so 6,291 is kind of, already the letter is there and maybe some kind of MOU is being signed this month itself. You would have seen that, there is something to SJVNL and NHPC also. So, it is similar to THDC and NEEPCO. THDC and NEEPCO are our subsidiaries. So, our hydro is coming mainly through these two arms, and we will make sure that, we are meeting the requirement of the capex for those companies also',
        'And there are further, this is in the kind of pipeline, so when I was saying around 8,000, so 6,291 is kind of, already the letter is there and maybe some kind of MOU is being signed this month itself. You would have seen that, there is something to SJVNL and NHPC also. So, it is similar to THDC and NEEPCO. THDC and NEEPCO are our subsidiaries. So, our hydro is coming mainly through these two arms, and we will make sure that, we are meeting the requirement of the capex for those companies also. And they are generating on their own also, but I think we will try to see that, we as the shareholder and we as the holding company will be able to take care of that. Director (Projects): We can give them a sense of the capex that we are planning. The group capex for the current year is Rs.27,104 crore. Next year, it will be Rs.38,692 crore, and then next year, that is 2025- 26, it will be Rs.49,000 crore and in 2026-27, it will be around Rs.58,000 crore. In 2027-28, it will be Rs.50,600 crore. In 2028-29, it will be Rs.40,000 crore and in 2029-30, it will be around Rs.38,000 crore. We have projected up to that. This includes our nuclear also. CMD: Okay. So, moving further. How much is the C&I in 20 GW? Okay. Go ahead. CEO (NGEL): C&I would actually add up to about 10 GW out of this 20 GW, in terms of the actual installed capacity. CMD: Okay, base load plus peak load gas, I think gas, whether it is going to be? Yes, I think gas will have some say, but I am afraid that it will be few hours in a day and few months in the year. So, I will request Director (Operations) to just take that question on that. Director (Operations): Gas stations are still being run primarily for grid control. They are being run by Grid India. No discounts are actually scheduled in that power. But from the grid security point of view, it is very much essential that we have the gas plants available. So, for that, government has taken a lot of initiatives',
        "So, I will request Director (Operations) to just take that question on that. Director (Operations): Gas stations are still being run primarily for grid control. They are being run by Grid India. No discounts are actually scheduled in that power. But from the grid security point of view, it is very much essential that we have the gas plants available. So, for that, government has taken a lot of initiatives. They have instructed us to ensure fuel supply and in fact, this was for a crunch period, it was done in this financial year in the month of June, and going forward, the government sees that gas is required in order to maintain the grid discipline. So compared to last year, we already consumed lot of gas and we are going in agreement with the GAIL, to ensure further availability of gas in the long run. We are going to sign an agreement for another 4 and a half years. So, gas is going to stay here for some time to maintain the grid stability. SS&FA, MOP: Just to add on to it, gas will remain the last resort. CMD: That is the reason I just mentioned as Special Secretary said, it is just saying that this is going to be the last resort because it is very costly and we should not be unnecessarily trying to utilize it only for the grid security purpose, it is being used. Let us hope that the gas prices come down. Now it is still in double digit, if it comes in the single digit that will be really good. Okay. Page 20 of 23 NTPC Limited July 31, 2023 SMR capital outlay, I think the first thing, first stage is the SMR development itself which will be required only very less kind of capital cost but going forward, it will be huge. It can vary anything between anybody's guess as of today. So, let's not, I will not even dare to do that. It can be as high as Rs.30 crore to Rs.40 crore per MW, it may even touch Rs.50 crore also on the smaller side but there is a lot of advantages of that. So, the first thing is about the development side",
        "It can vary anything between anybody's guess as of today. So, let's not, I will not even dare to do that. It can be as high as Rs.30 crore to Rs.40 crore per MW, it may even touch Rs.50 crore also on the smaller side but there is a lot of advantages of that. So, the first thing is about the development side. So, which will be kind of some about Rs.1,000 crore in the first phase and then when you start manufacturing then it is going to be. So, it is going to be a developmental activity as of now. But this is for future. Keep in mind like the solar was 10 years back, the SMRs we have to keep track on them. Capital cost and the cost of capital competitiveness with private IPPs. Who are the gentlemen? I think that we have already responded on that side. Yes, you are right. Thank you. The first thing is, we were never scared of the competition from the IPPs. You would have seen that. That has been the track record on that side and whether it is cost of capital, we have some edge over the IPPs or our project management or our operational practices, this is what we have demonstrated, that is our core strength, which is coming and the only, I don't have the numbers but till now, our old plants are doing quite well, whatever we have in the renewable also and technically, operationally and financially. So, this is it. You want to say anything on it? Director (Projects): Yes sir, I would like to give them a reference, so that they can have more confidence on NTPC. See there is a constant argument on this IPP is cheaper, NTPC is costly. That's the argument you have. See the cost of the capex depends on two fundamental things. One is how do you engineer it and what are the quality standards. Engineering includes what is the likely availability you would like to project and all these things, loading factor, and quality definitely, how long it is going to run with that 100% capability. We have demonstrated what we have done",
        "See there is a constant argument on this IPP is cheaper, NTPC is costly. That's the argument you have. See the cost of the capex depends on two fundamental things. One is how do you engineer it and what are the quality standards. Engineering includes what is the likely availability you would like to project and all these things, loading factor, and quality definitely, how long it is going to run with that 100% capability. We have demonstrated what we have done. Now, if you compare, I'm not telling all IPPs, some of the IPPs are also good. If you compare now the solar, you will understand the solar, the substantial portion of the generation comes in from a module. So, what I purchase, the IPP also purchases. But we are mostly competitive in solar. So, why there should be an argument that apple-to-apple basis if we compare the coal fire power station, we are costly. We are not. The comparison comes because our things are better engineered with higher quality. CMD: Okay, so the next question was on the NEP and there is something on that, how much it is on the lower side. I just responded that yes, if whatever is required, we will see those on that side. So, it is not something which is done on one year and then we have to stick to them forever. So, it can be on the higher side, I can agree on that. In the RE, what is the bottleneck? I don't know. Then it's Mohit, you wanted to respond to that on that, there was one question. What is this loss of PLF? I think then this. Yes. No, no, but why you call it the loss of PLF? Analyst: I was saying loss of PLF incentive, potential loss of PLF incentive. CMD: No, no, I think, okay, the incentives mainly were then only above 85% of the PLF. So, I think Pithead Power Station, yes, Pithead Power Station still runs and it's almost more than 85%, but Page 21 of 23 NTPC Limited July 31, 2023 at the same time if there is a backing down, we are compensated for that efficiency loss. So, this is included in the regulatory mechanism",
        "No, no, but why you call it the loss of PLF? Analyst: I was saying loss of PLF incentive, potential loss of PLF incentive. CMD: No, no, I think, okay, the incentives mainly were then only above 85% of the PLF. So, I think Pithead Power Station, yes, Pithead Power Station still runs and it's almost more than 85%, but Page 21 of 23 NTPC Limited July 31, 2023 at the same time if there is a backing down, we are compensated for that efficiency loss. So, this is included in the regulatory mechanism. Green initiatives and nuclear. I think this is what we have been covering on that part. Is there anything specific you wanted to ask? Analyst: No, I was asking you know that kind of returns that you are targeting both on the renewable front and also on the new initiatives, what is the kind of capex and also the returns that you are targeting here? CMD: Yes, that we had responded on that part. But okay, I am still not very clear on that. See, IRR, it cannot be just like there is -- it has to be on a benchmark, and it has to be a double digit anyway. So, we cannot go in the kind of, this is not a regulated sector as far as the renewable is concerned. But you cannot say that this is the exactly one IRR I have to take it. So, you have to be something on the higher side, something you can take little on the lower side, but more or less it will be at par with our regulated sector. Director (Finance): If I may add, see in a conventional safe business which you know on a cost-plus basis you have a return of 15.50% which is available by the regulator. Do you know the gestation period is almost 5 to 6 years, so that translates to something around 12.5% IRR. So, that would be our endeavor, in the sense that, there is a trade-off, like in order to, there is compulsion you need to add capacities and also go on the renewable side that is the agenda",
        'Director (Finance): If I may add, see in a conventional safe business which you know on a cost-plus basis you have a return of 15.50% which is available by the regulator. Do you know the gestation period is almost 5 to 6 years, so that translates to something around 12.5% IRR. So, that would be our endeavor, in the sense that, there is a trade-off, like in order to, there is compulsion you need to add capacities and also go on the renewable side that is the agenda. But in that process let us say if I am very aggressive, I am also pursuing some capacities on merchant, I am also pursuing some capacity on C&I. The endeavor would be to fit in closely with the IRR which we are doing. However, I mean more than an IRR I would say the kind of profitability because IRR can be financial wizardry also, you all understand. So, leave aside the IRR, we will have returns which would be adequate enough to maintain. Analyst: Sure, sir. That answers my question. Thank you. CMD: So, if I have taken then a few of the messages, I think mainly it was on the NGEL IPO, the capacity addition, the renewables, the hydrogen, and I think then another one is coming out to within what kind of capacity. I think just be reassured that the demand is increasing, the inventory is getting exhausted, so we will have to go on adding the capacity both on renewable side as well as on the conventional side. We have a lot of space in our power stations, existing power stations, so we are not required to go to the Greenfield. That is what our endeavor will be unless there is something. Incidentally, I have not mentioned that the Government of UP has already approved one joint venture. We will be going ahead with the 50-50% and one of the existing sites, Obra as of now, and followed by Anpara, that is another one which is going to be. There are other state governments who are talking to us. So, I think this is all round',
        "That is what our endeavor will be unless there is something. Incidentally, I have not mentioned that the Government of UP has already approved one joint venture. We will be going ahead with the 50-50% and one of the existing sites, Obra as of now, and followed by Anpara, that is another one which is going to be. There are other state governments who are talking to us. So, I think this is all round. It is not only one, it’s not only renewable, it’s not only green hydrogen, green ammonia, it’s as he mentioned the gas is also coming into that, but the last resort it was just rightly corrected. So, there is enough I think growth and we will be taking the leadership in this also and we will try to not only maintain our share, but I think we will try to increase that from 25%. I think that should be giving a kind of and we are not really out of cash, we are generating enough cash which can really support this kind of our capex requirement and the growth on that side. Page 22 of 23 So, I think with the team's capability, we will be able to really go on doing what we had been demonstrating in the past. So, before I conclude, I would request Special Secretary sir to see if there are a few things what he wanted to convey. NTPC Limited July 31, 2023 SS & FA, MOP: Well, there was a lot of discussion about asset monetization, when the IPO will come, how the IPO will come. As you know, we require huge capital investment in coming years, looking into the demand of energy sector and it is not only in this sector, but in almost in all sectors of government wherever PSUs are working, infrastructure works, and capital demand is growing. And when there is a huge demand, our PSUs go to the market. They leave very little space for private sector to come out or I will say the space for private sector gets limited",
        'As you know, we require huge capital investment in coming years, looking into the demand of energy sector and it is not only in this sector, but in almost in all sectors of government wherever PSUs are working, infrastructure works, and capital demand is growing. And when there is a huge demand, our PSUs go to the market. They leave very little space for private sector to come out or I will say the space for private sector gets limited. In order to remove this imbalance, government has taken this policy decision that some of the part of this capital requirement should be generated by the CPSEs themselves by asset monetization and this is how this whole picture fits that they should generate some of the capital required for their future growth from within their own assets. ED (Finance): Thank you Sir. We now conclude this 19th Annual Analysts and Investors Meet. I am grateful to our CMD, Special Secretary and Financial Advisor, Ministry of Power, our Directors and Senior Management present here for taking out time for this interaction. The address by CMD and the presentation by Director (Finance) highlighted the achievements of NTPC and showcased the potential that we have. I am thankful to all our analysts and investors who are present here today, whose faith has driven us to exceed our performance, both operationally and financially, year after year. Last but not the least, my heartfelt thanks to Regional Executive Director, Western Region and his team, for making arrangements for a successful meet. We hope to see you next year. I now invite all of you for refreshments. Thank you. ******************************* Page 23 of 23'
    ]
    final_summary= get_final_output(ntpc_chunks)
    print("Final summary")
    print(final_summary)
        
main()
