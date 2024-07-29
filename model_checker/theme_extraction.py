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
    ''' Identify top 10 themes from the given list'''
    try:
        template = """<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        <</SYS>>
        Identify 10 most important points relevant for financial information without any explanation and repetition from the given text below.
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
        Your summary should  consist of exactly 10 bullet points, each at least 20 words long. Blend factual information with insightful inferences.
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

def remove_headers(text):
    """Remove headers and generate as bullet points"""
    try:
        lines = text.strip().split("\n")
        processed_lines = []
        for line in lines:
            if line.startswith("•"):
                colon_pos = line.find(":")
                if colon_pos != -1:
                    processed_line = "• " + line[colon_pos + 1:].strip()
                else:
                    processed_line = line.strip()
                    processed_lines.append(processed_line)
        
            elif line!='' and line[0].isdigit():
                line= line.replace(line[0],"• ",1)
                line= line.replace('.','',1)
                processed_lines.append(line.strip())
            
            else:
                processed_lines.append(line.strip())
        processed_text = "\n".join(processed_lines)
        return processed_text
    except Exception as e:
        logging.error("Error removing headers: %s", e)
        raise e
    
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
        actual_list= [x.strip() for x in combined_summary.split('\n')]
        joined_summary= "".join(actual_list)
        summary_list= textwrap.wrap(joined_summary,12000)
        output_summary=""
        for summary in summary_list:
            generated_summary= get_final_summary(summary,llm_model)
            output_summary+=generated_summary
        if len(output_summary.strip().split("\n"))>10:
            concised_summary= get_final_summary(output_summary,llm_model)
            final_summary= remove_headers(concised_summary)
        else:
            final_summary= remove_headers(output_summary)
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
        return theme_based_summary
    except Exception as e:
        logging.error(e)
        raise e

def remove_similar_summary_points(embedding_model,theme_summary):
    ''' Check similarity between summary points'''
    try:
        summary_points= theme_summary.strip().split("\n")
        summary_embeddings= [generate_embeddings(embedding_model,summary) for summary in summary_points]
        for i in range(len(summary_embeddings)):
            for j in range(i+1,len(summary_embeddings)):
                if (cos_sim(summary_embeddings[i],summary_embeddings[j]).item())>0.89:
                    summary_points.remove(summary_points[j])
        final_theme_summary= "\n".join(summary_points)
        return final_theme_summary
    except Exception as ex:
        print(ex)
        raise ex


def compare_two_themes(embedding_model,theme1_summary,theme2_summary):
    ''' Check similarity between two themes'''
    try:
        similar_pairs=[]
        theme1_embeddings=[generate_embeddings(embedding_model,summary_point) for summary_point in theme1_summary]
        theme2_embeddings=[generate_embeddings(embedding_model,summary_point) for summary_point in theme2_summary]
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
        for x in range (len(themes_summary_list)):
            for y in range(x+1,len(themes_summary_list)):
                if compare_two_themes(embedding_model,themes_summary_list[x],themes_summary_list[y]):
                    del theme_based_summary[themes_list[y]]

        return theme_based_summary

    except Exception as ex:
        print(ex)
        raise ex

def get_refined_document_summary(chunk_dictionary,embedding_model):
    ''' Apply cosine similarity to remove similar data'''
    try:
        final_doc_summary={}
        document_summary= get_document_theme_summary(chunk_dictionary,llm_model)
        refined_summary= check_similar_theme_summaries(embedding_model,document_summary)
        for theme,summary in refined_summary:
            final_doc_summary[theme]= remove_similar_summary_points(embedding_model,summary)
        
        return final_doc_summary
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
    airtel_chunks=[
        'The participants may click this option during the management opening remarks itself to find a place in the queue. Upon announcement of name, participants may kindly click on “Unmute Myself” in the pop up on screen and start asking the question post introduction. With this, I would like to hand over to Mr. Gopal Vittal for the opening remarks. Gopal Vittal – Managing Director & Chief Executive Officer - Bharti Airtel Limited Good afternoon, ladies and gentlemen. Thank you for joining this webinar to discuss Bharti Airtel’s results for the Quarter ended 30th June 2022. Also present with me on this webinar are Soumen Ray, Harjeet Kohli and Arpan Goyal. I want to focus this quarter’s earnings call on three things. Some overall comments and an update on our performance, a deeper dive into our 5G approach now that the auction is behind us and why I believe we are so well positioned as Airtel to win the coming 5G game. Despite macro challenges, we have seen another steady quarter for Airtel. Our consolidated revenues for the Quarter grew sequentially by 4.1% to hit Rs 32,805 crores. In India our EBITDA margins improved to get to 51% We have now received the 700 million USD from Google for a stake of 1.2 percent. Our leverage at the end of the quarter stands at 2.52x and has been consistently getting better. This is due to our tight fiscal prudence and improved operating leverage that we are seeing. A quick word on ESG. To drive greater transparency around how our businesses are creating value by contributing towards a sustainable in economy, we are voluntarily making our BRSR disclosure with effect from the financial year ending March 2022. I am also delighted that Sustainalytics has upgraded the ESG rating of Airtel to Low risk with a significant enhancement in rank across all telecommunications services companies globally. During the quarter Nxtra was recognized as one of the Sustainable Organizations 2022',
        'To drive greater transparency around how our businesses are creating value by contributing towards a sustainable in economy, we are voluntarily making our BRSR disclosure with effect from the financial year ending March 2022. I am also delighted that Sustainalytics has upgraded the ESG rating of Airtel to Low risk with a significant enhancement in rank across all telecommunications services companies globally. During the quarter Nxtra was recognized as one of the Sustainable Organizations 2022. In order to bring world-class digital access to all and reduce the digital divide, we became the first ISP to launch broadband in Ladakh and the Andaman and Nicobar Islands. Finally, we continue to maintain the highest standards of corporate, financial and operational disclosure. Let me now briefly touch on each of our businesses. I will start with our smallest but most exciting businesses. Airtel Payments bank now has a monthly transacting user base of 44.4 million. This quarter we clocked an annual revenue run rate of Rs 1100 crores and a GMV of 5.4 billion Dollars. Our take rates of 0.66 percent are the highest in the industry making us the only profitable fintech player. In addition, our digital services have now reached an annualized run rate of Rs 850 crores. As I have mentioned before, our digital business is extremely capital light and leverages our underlying strengths. Now I want to turn to Airtel Business, a segment that I like to think of as a jewel in the portfolio. Here we reported strong 4.4% sequential revenue growth. We have out-performed the other listed companies in this space and accelerated our market share further. In fact, Airtel Business has reached a special milestone and is now the #1 B2B player in India. Our success can be attributed to a razor-sharp strategy of both going wide to penetrate more accounts as well as going deep to serve our largest customers with our full set of products and solutions',
        'Here we reported strong 4.4% sequential revenue growth. We have out-performed the other listed companies in this space and accelerated our market share further. In fact, Airtel Business has reached a special milestone and is now the #1 B2B player in India. Our success can be attributed to a razor-sharp strategy of both going wide to penetrate more accounts as well as going deep to serve our largest customers with our full set of products and solutions. Clearly, the focus on fast growing emerging products including CPaaS, data centres and cyber security are now yielding results. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 2 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript I also want to call out IoT – where we are outperforming the market by a margin. You should also know that all our IoT customers are postpaid customers but we include these connections as a part of the Airtel Business segment. So at the end of Q1, we had a customer base of about 29.2 Million on Postpaid – 18.1 million reported in Mobile services as a part of our Mobile services and another 11.1 million IoT connections. In effect Post-paid, therefore, as a segment is about 45% larger than our nearest competitor. Our broadband business has continued to see very strong customer additions driven by rapid roll outs and solid marketing. During the quarter, we added 1.7 Mn home passes on the back of accelerated rollouts through the local cable operator model. With this, we are now present in 983 cities. We are making solid investments in this segment to build a pole position in the broadband market. As a result, we added 310K customers and reported 5.7% sequential revenue growth. This was despite some offset of a one-off benefit in Q4FY22. The DTH business saw a decline of 0',
        'During the quarter, we added 1.7 Mn home passes on the back of accelerated rollouts through the local cable operator model. With this, we are now present in 983 cities. We are making solid investments in this segment to build a pole position in the broadband market. As a result, we added 310K customers and reported 5.7% sequential revenue growth. This was despite some offset of a one-off benefit in Q4FY22. The DTH business saw a decline of 0.9 percent but the silver lining is that we have consistently grown market share over the last few quarters. As I have said before, the entire industry has been brought to its knees by excessive regulation. The New Tariff order introduced a few years ago created mind boggling complexity for the customer with no benefit to any stake holder. Secondly, the same content, if provided through a different medium – broadband or wireless is not subject to this tariff order creating an arbitrage and uneven playing field. Finally, good content is being made available on Free to Air channels. This perfect storm has led to serious head winds in an industry where there is still a massive opportunity to grow from Cable. Given this backdrop, our approach has been to leverage the core strengths of the ongoing broadband explosion to put Airtel Black and Convergence at the heart of our strategy. We have seen substantial growth in the last month on our broadband and content bundling that includes linear TV as well as OTT content. We expect the business therefore to see some recovery in the coming quarters. Let me now turn to the Mobile segment. Here we have seen sequential revenue growth of 3.4%. We added 4.5 million 4G net adds in the quarter and about 250K postpaid customers. With the increase in prices of entry level smart phones, and the focus of OEM’s on higher value smart phones, the overall industry did see a nearly 15 percent reduction in upgradation. We expect this to normalize over the coming quarters',
        'We expect the business therefore to see some recovery in the coming quarters. Let me now turn to the Mobile segment. Here we have seen sequential revenue growth of 3.4%. We added 4.5 million 4G net adds in the quarter and about 250K postpaid customers. With the increase in prices of entry level smart phones, and the focus of OEM’s on higher value smart phones, the overall industry did see a nearly 15 percent reduction in upgradation. We expect this to normalize over the coming quarters. At Rs 183 ARPU, we continue to lead the industry. We are confident that we will see ARPU of Rs 200 and eventually Rs 300 arising out of tariff rises. In all, the quarter witnessed broad-based growth across all of our businesses and our focus on war on waste led to improvement in margins. Let me now turn to 5G. As you know, Airtel has now acquired 100 MHz of 3.5 Ghz across every circle in the country. 3.5 Ghz is the work horse layer for 5G and will give us a pan India foot print that can bring a true 5G experience for our customers. In addition, Airtel has acquired 800 Mhz of 26 Ghz spectrum across every circle in the country. This is a band that has limited propagation but gives 4 Gbps type speeds. Finally, we also bolstered our spectrum holdings in the mid band and low band (1800/2100/900 bands). As a result, the total commitment in this auction was 43040 crores. Let me now give you our approach on 5G. I want to do this in real layman terms so that I am hopefully able to demystify the incorrect narratives that are making the rounds in a few places. You may know there are two modes in which 5G operates. Stand Alone (or SA) mode and Non Stand Alone (or NSA mode). In the SA mode, 5G comes as a top up to an existing 4G radio layer. The 4G layer then operates independently. There are two issues here. The first is the lack of a well-developed eco system for SA devices. The second issue is propagation. The work horse layer 3.5 Ghz, has lesser propagation than even 2',
        'You may know there are two modes in which 5G operates. Stand Alone (or SA) mode and Non Stand Alone (or NSA mode). In the SA mode, 5G comes as a top up to an existing 4G radio layer. The 4G layer then operates independently. There are two issues here. The first is the lack of a well-developed eco system for SA devices. The second issue is propagation. The work horse layer 3.5 Ghz, has lesser propagation than even 2.3 Ghz and this impacts coverage in urban areas. As a result, SA can be effective only if there is a sub Ghz layer that is also offering SA and the two work in conjunction. The role of the sub Ghz layer is only for coverage not capacity or speed. So the sub Ghz is needed where 3.5 does not reach. In fact, we have seen in our trials that the sub Ghz layer on 5G SA gives only 8-10 Mbps speeds, no different from 4G. This is a very important point to understand given the misconceptions about the 700 band and why it is a panacea to everything. It is not. The band is absolutely no different from 850 or 900 in terms of propagation. All it does is provide coverage at the edge, deep indoor or in far flung rural areas. And gives at best 4G like speeds. Nothing more. The second mode that 5G operates is NSA. A word on the physics involved. The limitation of Radios and spectrum propagation is the uplink. It is the uplink that allows a user device to upload a photo, any other content or even have a conversation. The down link however can travel much further which as you can imagine, is of no help for user experience because without uplink capabilities there will be no coverage. In the NSA mode, the great advantage is that the 3.5 Ghz layer actually travels much further because it performs the down link – it does not need to do the uplink – because that is done by the 4G layer – this is invariably the 1800/2100 band, also called mid band. As a result, there are several advantages of the NSA mode. First, the 3',
        'In the NSA mode, the great advantage is that the 3.5 Ghz layer actually travels much further because it performs the down link – it does not need to do the uplink – because that is done by the 4G layer – this is invariably the 1800/2100 band, also called mid band. As a result, there are several advantages of the NSA mode. First, the 3.5 Ghz band extends at least 30 percent more on the down link, which implies a massive 100 meters extra. This gives you substantially more coverage in urban areas, when mid band is providing the uplink. At the edge when you need further extension of coverage and the mid band does not work, the fall back in terms of coverage is provided by the sub-ghz layer. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 3 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript The second advantage of NSA is that all devices work on this mode. It is the most widely available eco system in the world and as you know telecom is a game of eco systems. In the US and South Korea where both SA and NSA have been launched the traffic on SA is less than 10 percent of total 5G traffic. The third advantage of this mode is that it allows us to use existing 4G technology at no extra cost since we already have the radios and the spectrum that are live on our network. Finally, the last advantage of NSA is around experience – it will allow for a faster call connect time on Voice. In addition, it allows us to provide a faster uplink than anyone else given our massive spectrum holdings in the mid band. This is what leads me to the crux of the issue. To offer NSA you need solid mid band spectrum. Because that is what provides a great uplink experience',
        'Finally, the last advantage of NSA is around experience – it will allow for a faster call connect time on Voice. In addition, it allows us to provide a faster uplink than anyone else given our massive spectrum holdings in the mid band. This is what leads me to the crux of the issue. To offer NSA you need solid mid band spectrum. Because that is what provides a great uplink experience. Over the last few years we have strategically accumulated the largest pool of mid band spectrum – today we have 30 Mhz of mid band spectrum in 4 circles and 20 or above in the rest. Our competition does not have such large mid band spectrum. Do remember that if we did not have this large chunk of precious mid band spectrum we would have had no choice but to buy expensive 700 mhz spectrum. And once we had bought it we would have had to deploy large power guzzling radios on this band. Not only would the cost have been higher, it would have led to more carbon emissions and very very importantly given us no additional coverage than our existing 900 spectrum band. So in sum, what would we have got. More cost, more ESG unfriendly and no improvement in coverage. Therefore, this well thought out strategy for spectrum acquisition through auctions, M&A and trading over the last five years has allowed us to avoid the need for adding an expensive sub Ghz band and yet deliver a world class 5G experience. In sum, our consistent long term spectrum strategy, will help us meet all our objectives - the best 5G experience, 100x capacity enhancement, the most power efficient solution and a lower total cost of ownership. We believe this will give us an enduring competitive advantage for years to come. With this auction, we are fully confident that we will not be required to spend any material amount on spectrum for many years to come. Let me now briefly touch on our 5G readiness and rollout. We intend to launch 5G starting August and extending to a Pan India roll out very soon',
        'We believe this will give us an enduring competitive advantage for years to come. With this auction, we are fully confident that we will not be required to spend any material amount on spectrum for many years to come. Let me now briefly touch on our 5G readiness and rollout. We intend to launch 5G starting August and extending to a Pan India roll out very soon. By March 2024 we believe we will be able to cover every town and key rural areas as well with 5G. In fact, detailed network rollout plans for 5000 towns in India are completely in place. This will be one of the biggest roll outs in our history. While our 3-year capex will remain around the same levels this rapid roll out could see some advancing of capex on an in year basis. Every network domain is completely 5G ready. Our transport layer has been built over the years. With the combination of fiber to the tower, synergies from fiber to the home and the availability of E-band micro wave spectrum, every site we roll out will be backhaul ready to provide 5G experience in line with what is needed to deliver a world class experience. Our multi terabit MPLS and internet backbone is fully ready to take on the 5G led data growth. Even on the cloud network side, we have best of breed partners for the cloud platform that will help us in the deployment of our network in a smooth manner. At the same time, our core and our radio network are all future proofed and can operate either on NSA or SA mode. So over time, as more and more devices come in to India and we see disproportionate 4G traffic move on to the 5G network, we will be able release all of our spectrum on 4G band – 1800/2100/900 and even 2300 to SA and switch seamlessly from one mode to the other – all of this at the flick of a button through software. I now want to step back and share my confidence by giving you five powerful reasons on why we will win the 5G game. First, our whole business is squarely focused on quality customers',
        'I now want to step back and share my confidence by giving you five powerful reasons on why we will win the 5G game. First, our whole business is squarely focused on quality customers. Whether it is Mobile, Broadband or Converged products everything we do across channels, marketing or differentiation is singularly centered on Quality customers. We see the lowest level of churn amongst these customers and they will be the ones who will upgrade to a 5G device faster than anyone else. Second, we have one of the most powerful Enterprise Businesses where we have the relationships and enjoy the trust of our customers. This business has incredible momentum. All of 5G enterprise innovation will be brought to the market by us to serve these customers. I believe we can win even more strongly on 5G because of the Enterprise business. Third, we have compelling digital capabilities now to lead the 5G game. Whether it is our omni channel approach or indeed our digitization of the network, we are now ahead. A couple of examples here. We now have our own Airtel Self Optimising network, built by our Engineers. This is an AI-ML driven in-house, closed loop, self-healing platform. It was developed for automated network management, for proactively managing faults, for saving power on the fly and much else. Use cases are being written every fortnight. This powerful tool has won global awards as well. A second tool has been built to completely automate 5G network planning, configuration, on boarding and even for field operations. Our Security operating centre (SOC) for enhanced security and vulnerability management is also 5G ready. These capabilities will allow us to deliver the best 5G experience. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited',
        'A second tool has been built to completely automate 5G network planning, configuration, on boarding and even for field operations. Our Security operating centre (SOC) for enhanced security and vulnerability management is also 5G ready. These capabilities will allow us to deliver the best 5G experience. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 4 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Fourth, we have a set of digital services each of which plays a role in reducing churn and improving stickiness across both the consumer and Enterprise business. So whether it is our Payments bank, our Wynk music platform, our X stream video platform, our launch of Airtel IQ or even Airtel Ads each of these also creates stickiness. And this stickiness translates to lower churn and higher switching cost. Finally, the most important reason - our track record. We came into the market late on 4G in 2016 against a deep pocketed competitor that rolled out a larger and wider network than Airtel who was in the industry for many years. It had not happened anywhere else in the world. While that was happening, narratives were spun. A pure 4G network is so much better than a legacy network. Simple and clean 4G is so much better than multi technology networks. We have heard such illogical and fact free stories before. All I can point you to is our track record. We have the best quality customer in the industry. We deliver the best experience in the industry. We are constantly getting better every day. Our whole purpose is to serve our customers. We listen to them carefully. And the icing on the cake is that we have out- performed the industry by notching up consistent market share gains for many years. In my view the only thing that customers really care about is Experience',
        'All I can point you to is our track record. We have the best quality customer in the industry. We deliver the best experience in the industry. We are constantly getting better every day. Our whole purpose is to serve our customers. We listen to them carefully. And the icing on the cake is that we have out- performed the industry by notching up consistent market share gains for many years. In my view the only thing that customers really care about is Experience. On that front we will deliver. Again and Again. Better than anyone else. Because we have been very astute in our spectrum acquisition. Because we will continue to invest in the best technology. We will work with the best global partners. We will not try and do everything ourselves because we are not arrogant to even entertain a fleeting feeling that we can be better than our partners. These are the reasons for my confidence. And for you the proof of the pudding will be in the eating. With this let me open up the floor for Q&A. Rajyita - Moderator Thank you very much Sir. We will now begin the Q&A interactive session for all the participants. Please note that the Q&A session will be restricted to analysts and investor community. Due to time constraints, we would request if you could limit the number of questions to two per participants to enable more participation. Interested participants may click on Raise Hand option on Zoom application to join the Q&A queue. Upon announcement of name, participant to kindly click on “Unmute Myself” in the pop-up on screen and start asking the question post introduction. The first question comes from Mr. Sanjesh Jain from ICICI Securities. Mr. Jain you may please un-mute your side, introduce yourself and ask your question now. Sanjesh Jain - ICICI Securities Good afternoon, Gopal. Thanks for that elaborated understanding on 5G, that is really really useful',
        'Upon announcement of name, participant to kindly click on “Unmute Myself” in the pop-up on screen and start asking the question post introduction. The first question comes from Mr. Sanjesh Jain from ICICI Securities. Mr. Jain you may please un-mute your side, introduce yourself and ask your question now. Sanjesh Jain - ICICI Securities Good afternoon, Gopal. Thanks for that elaborated understanding on 5G, that is really really useful. Few questions from 5G, first on the capex, you did mention that your three year capex holistically will not change, so we assume that for three years we still are confident that we will be spending only Rs 75,000 Crores or Rs 750 billion, which we were anticipating earlier, that remains true? How much of it will be front loading because we are also very, very aggressively pushing 5G, now talking of reaching 5000 towns by the end of March 2024, that looks like an impressive plan, but how much will it consume in terms of cash flow in the near term say for the next two years, how should we look at? That is number one. Number two on the operating costs you did mention on the user experience and the capex side of SA and NSA, can you elaborate what will be the difference between the operating costs in SA and NSA now considering that NSA will also sweat the 5G asset while SA will be predominantly a standalone 5G stack. Will it have any material difference from the operating cost side of it? Third one, on the economics of 5G itself, now that we have got a such a large spectrum pool and now we are also rolling out additional network on it, how should one think on this incremental investment? What will be the IRR? How will an operator recover the investment? So these are my initial questions. Thanks. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited Thanks, Sanjesh',
        'Will it have any material difference from the operating cost side of it? Third one, on the economics of 5G itself, now that we have got a such a large spectrum pool and now we are also rolling out additional network on it, how should one think on this incremental investment? What will be the IRR? How will an operator recover the investment? So these are my initial questions. Thanks. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited Thanks, Sanjesh. I think on the capex we do not give guidance anymore, but if you look at our capex profile over the last couple of years, we expect if you take a three years type of view we will probably be in the same ballpark. Of course, we will advance the capex in this year itself, starting now. So to that extent there will be advancement of capex that will happen over the course of this year and I think in an 18 months period it will normalize. On the operating costs, I think if we did not have the mid band spectrum and we had to buy the 700 Mhz band, we would need to spend let us say for 10 megahertz of spectrum Rs 40,000 Crores. This would need more antennas, big radios, extra radios across the country in all the sites that we go in, extra power that will need to be deployed there and extra rentals of the radios. All this would have had a cost per GB disadvantage compared to where we are today of almost 50%. So, if we are at X today, we would be 50% higher cost per GB. That is our math which is the reason that over the last five years we have carefully acquired a large pool of mid band spectrum because it was very, very well thought through. We never went public about it because obviously it was highly Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited',
        'So, if we are at X today, we would be 50% higher cost per GB. That is our math which is the reason that over the last five years we have carefully acquired a large pool of mid band spectrum because it was very, very well thought through. We never went public about it because obviously it was highly Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 5 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript confidential. This was really the strategy for us to make sure that we get to 20-30 megahertz in all circles in mid band and that is really what we have done. The economics of 5G is a function of I would say the incremental tariff increases that industry sees. Remember over the period of time, 4G capex would be displaced with 5G, as bulk of the 4G rollout has happened. Today, already 8% to 9% of the devices are 5G ready. Shipments that are coming in are about 30% to 35% of devices and we expect this will change a lot. So by next year, almost 80% to 90% of the devices that will come in will be 5G devices. Which means if you project 8% to 9% to March 2023, it could be close to 13% to 14% and then by March 2024, it will be much higher. So a bulk of our rollouts are done, which means that the capex that goes onto the radio on 4G will get displaced by 5G and this is why I stated over a three year period our capex profile fundamentally stays the same. How the pricing will work between 5G and 4G is something that have not yet quite decided, so I am not going to comment on that right now',
        'Which means if you project 8% to 9% to March 2023, it could be close to 13% to 14% and then by March 2024, it will be much higher. So a bulk of our rollouts are done, which means that the capex that goes onto the radio on 4G will get displaced by 5G and this is why I stated over a three year period our capex profile fundamentally stays the same. How the pricing will work between 5G and 4G is something that have not yet quite decided, so I am not going to comment on that right now. What we have seeing globally is that 5G by itself is not yet giving incremental ARPU to any operator anywhere in the world, but in India as you know tariffs are still very low, we do expect tariffs to increase and with every increase in tariffs obviously the economics will change and the return on capital will get much-much better. Sanjesh Jain - ICICI Securities That is quite clear. Just one last question before come back in the queue. You have formed customer advisory board for your enterprise business, now what are they telling about the 5G adoption in India particularly for Airtel, what are their demand and is 5G capex is in sync with them, are they really excited about the 5G or they think it is too much for them to invest initially to justify their ROCE, what is their understanding on enterprise side of 5G? Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited On the enterprise side of 5G, the customer advisory board has given us a lot of ideas on new services to be launched. Airtel IQ, work from anywhere solution, a lot of the work that we have done on the SD-WAN and the startup acquisitions came out of these discussions. We are also having multiple conversations on private networks with a large number of companies. So we are looking at that. As of now, our rollout will happen over a period of time, we will see how the enterprise use cases play out. But there are ongoing conversations on standalone networks for large distributed enterprises',
        'Airtel IQ, work from anywhere solution, a lot of the work that we have done on the SD-WAN and the startup acquisitions came out of these discussions. We are also having multiple conversations on private networks with a large number of companies. So we are looking at that. As of now, our rollout will happen over a period of time, we will see how the enterprise use cases play out. But there are ongoing conversations on standalone networks for large distributed enterprises. Sanjesh Jain - ICICI Securities So, we see in FY2024 some contribution in enterprise coming from the 5G part of it or will it an ambitious one? Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited No, I think that the enterprise contribution has been going up in the overall profile of the company. The enterprise business has performed exceedingly well over the last decade. We see no reason why the enterprise business will not sustain its growth. Obviously as new technologies come, we also see adoption. So to that extent, I do not look at isolating what in the enterprises is coming out of 5G. I look at 5G, 4G and all of this as underlying connectivity layers, but it is also giving you the capability of adding new solutions. Ultimately, we are going to deliver solution for the customers which will lead to revenue, so whichever technology comes from is not the issue, the issue is how we are going to grow the enterprises business. Sanjesh Jain - ICICI Securities Got it, Gopal. Thanks for all the answers and will come back in the queue for others. Rajyita – Moderator Thank you, Thank you very much Mr. Jain. The next question comes from Mr. Pranav Kshatriya from Edelweiss Securities. Mr. Kshatriya, you may please un-mute yourself, introduce yourself and ask your question now. Pranav Kshatriya - Edelweiss Securities Thanks for the opportunity',
        'Sanjesh Jain - ICICI Securities Got it, Gopal. Thanks for all the answers and will come back in the queue for others. Rajyita – Moderator Thank you, Thank you very much Mr. Jain. The next question comes from Mr. Pranav Kshatriya from Edelweiss Securities. Mr. Kshatriya, you may please un-mute yourself, introduce yourself and ask your question now. Pranav Kshatriya - Edelweiss Securities Thanks for the opportunity. My first question is on this NSA and SA, if you look at this SA rollout has been very recent and to that extent as you rightly pointed the ecosystem is not mature, but do you think that at some point if the SA becomes more predominant and the ecosystem picks up, we have the flexibility to really acquire additional spectrum and make the network SA. That is my first question. Secondly, there has been a fairly strong acceleration in terms of number of cities where you are present in terms of the fiber for home broadband, so can we expect that to drive disproportionate growth in the coming quarters and lastly a bookkeeping Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 6 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript question, can you give some colour on, how much is the depreciation pertaining to the lease liabilities because there is a sharp increase in the depreciation in this quarter, so some colour on the breakdown of that, these are my three questions? Thank you. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited I already explained the SA versus NSA issue and let me just kind of repeat this probably for the question that you asked. Over a period of time, as more and more 5G devices come in, all the 4G traffic will shift on to the 5G networks',
        'Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited I already explained the SA versus NSA issue and let me just kind of repeat this probably for the question that you asked. Over a period of time, as more and more 5G devices come in, all the 4G traffic will shift on to the 5G networks. Spectrum that we have in the mid band in the 1800 and 2100 band, which is 20 to 30 megahertz per circle, will be the first band that will be refarmed for 5G, 2300 band will also get refarmed for 5G, ultimately even the sub-ghz band. Remember Sub-ghz only gives you a coverage, nothing more than coverage. If it is 10 megahertz sub-ghz, it is about 7 to 8 Mbps and 5 to 6 megahertz spectrum will give 3 to 4 Mbps. So all it does is give you coverage at 4G like speeds. Thus, will be the last band to get refarmed to 5G. As the refarming happens the NSA mode can seamlessly switch on to SA. Everything is ready - the core will be ready, the radios are ready, they are all software enabled, so it is the mode of operating and in a way, there is absolutely no difference in what we will ultimately deliver in terms of customer experience. On fiber broadband, we are seeing a sharp growth. We have added 310,000 users this quarter and we see that this growth will hopefully sustain. The demand for home broadband is exploding, coupled with convergence and proposition that we launched in Airtel Black where we are combining the content bundles in both OTT and linear. We expect this to continue to see a significant traction and hopefully this business, which is a very good business with very low levels of churn, will become very sizable business in years to come. I will leave the last question to Soumen',
        'The demand for home broadband is exploding, coupled with convergence and proposition that we launched in Airtel Black where we are combining the content bundles in both OTT and linear. We expect this to continue to see a significant traction and hopefully this business, which is a very good business with very low levels of churn, will become very sizable business in years to come. I will leave the last question to Soumen. Soumen Ray - Chief Financial Officer, India & South Asia - Bharti Airtel Limited Pranav, depreciation has gone up by about couple of percentage points of which there is a day extra in this quarter plus the number of towers increased with some renewals that happened, but nothing out of the ordinary. Pranav Kshatriya - Edelweiss Securities I just wanted to understand because you had this renewal, which you have had sort of one-off impact because the duration increases and that disproportionately increases the depreciation for those lease liabilities. So if you break it off, I think it will be useful? Soumen Ray - Chief Financial Officer, India & South Asia - Bharti Airtel Limited So, essentially the impact is more on interest because as you know in an equated monthly installment, in initial period the principal amount is much lesser and the interest amount is much more. So the impact is initially much more on interest and less on depreciation, but overall depreciation is up by above 2% point, 1% point is because of number of days and the other is just routine. Pranav Kshatriya - Edelweiss Securities Thank you. Rajyita - Moderator Thank you very much Mr. Kshatriya. The next question comes from Mr. Kunal Vora from BNP Paribas. Mr. Vora you may please un- mute yourself, introduce yourself and ask your question now. Kunal Vora - BNP Paribas Thanks for the opportunity. This is Kunal Vora from BNP Paribas. Question is on home broadband opportunity',
        'Pranav Kshatriya - Edelweiss Securities Thank you. Rajyita - Moderator Thank you very much Mr. Kshatriya. The next question comes from Mr. Kunal Vora from BNP Paribas. Mr. Vora you may please un- mute yourself, introduce yourself and ask your question now. Kunal Vora - BNP Paribas Thanks for the opportunity. This is Kunal Vora from BNP Paribas. Question is on home broadband opportunity. Can you talk about the fixed wireless broadband opportunity in India? What is the potential size of the overall home broadband market and whether we can expect fixed wireless broadband to reduce the size of FTTH opportunity? Also, home broadband this quarter looks a little soft I understand like you mentioned about from high base from last quarter, but have you seen any issues there with the economy opening up and the customers down trading with reducing work from home? Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited On home broadband, we have been adding about 300,000 plus customers every quarter, so we are sustaining that trajectory. If you are commenting on the sequential revenue growth, then yes. As the base inflates, we start to see some, not softening, but just the Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 7 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript sequential growth, not in absolute terms it is about the same, but in percentage terms it comes down, but we are still added 300,000 users. Like I said that I have confidence that this trajectory will get sustained. I think fixed wireless is an interesting opportunity and has been deployed in some markets in US and even Germany, but one of the things that we are seeing on fixed wireless is ultimately the cost per home pass',
        'Like I said that I have confidence that this trajectory will get sustained. I think fixed wireless is an interesting opportunity and has been deployed in some markets in US and even Germany, but one of the things that we are seeing on fixed wireless is ultimately the cost per home pass. It has to work competitively versus fibre in terms of cost per home pass. In India, if you look at our cost per home pass we have been able to bring it down substantially. In the cities where we roll ourselves, the cost per home pass is down to about $30 to $35. In the LCO cities it is the fraction of that cost, so on a blended basis it is substantially lower than $30. Now, typically if you rollout 100 home passes, we see the end of 18 to 24 months about 30% utilization, which means that if you are taking your blended cost let us say $15 to $18 then this is about $50 cost per home connected home. The cost of the router for fixed wireless today is about $200. In the case of Verizon in the US or DT, the cost for home pass remains almost $400, so for them the fixed wireless actually makes a lot of sense as you can put a router and spend that money. As the scale India develops, we will need to see the cost per router for the fixed wireless actually crash below $50 then it will start making some sense, but until such time that happens, our objective will be to really go very, very aggressive on fibre rollout itself as there is no better substitute for home broadband when compared to fiber. Fiber is always going to be better than wireless because it is a dedicated pipeline that you are delivering to the home and you have a full control over both the uplinks as well as the downlink with no constraints on traffic or population in that particular area. So I think it is down to economics the way we see. Kunal Vora - BNP Paribas Understood',
        'Fiber is always going to be better than wireless because it is a dedicated pipeline that you are delivering to the home and you have a full control over both the uplinks as well as the downlink with no constraints on traffic or population in that particular area. So I think it is down to economics the way we see. Kunal Vora - BNP Paribas Understood. The second question on the SG&A cost, we have seen a fairly sizable increase around 24 bps, we are almost 50% over the last two years and the churn has remained elevated over the last few quarters, it went to like 2% levels, but back to 3%, can you talk about like what is happening on the competitive intensity and when would you expect this to moderate? Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited Maybe, Soumen you can cover this. Soumen Ray - Chief Financial Officer, India & South Asia - Bharti Airtel Limited So the SG&A increase is in line with growth in business and the competitive activities in the market. Also there is a bit of deferment impact in this. It is not very different as now with the deferment, the cycle is getting caught up. But if you look at purely sales & marketing expenses, it has remained almost flat. If you are looking at other expenses inclusive then, there is the increase of about Rs 400 Crores, which is essentially around three items, the first being there is this FLO liability recasting which has been about Rs 200 Crores and then Africa there has been an increase of over Rs 150 Crores. Intrinsic India business there has been marginal increase of about 50 Crores. Kunal Vora - BNP Paribas Just one last question, if I look at the growth differential between Airtel and the number three players we are seeing some narrowing of the gap',
        'Intrinsic India business there has been marginal increase of about 50 Crores. Kunal Vora - BNP Paribas Just one last question, if I look at the growth differential between Airtel and the number three players we are seeing some narrowing of the gap. Airtel used to grow much faster compared to the number three player, we have seen that narrow despite larger investments in network can you share your thoughts on why did not happening and this larger 5G investment would we expect accelerated market will gain from the number three player? Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited I think that I do not want to comment on the number three player. I do believe that any operator that actually goes out and rolls a 5G network, you will start seeing the best quality customers first move to those networks and we are very confident that our postpaid business, our converged strategy, high value prepaid, all of this should see a substantial growth even with the rollout of 5G. Kunal Vora - BNP Paribas Thats it from my side. Thank you very much. Rajyita – Moderator Thank you very much Mr. Vora. The next question comes from Mr. Manish Adukia from Goldman Sachs. Mr. Adukia, you may please un-mute yourself, introduce yourself and ask your question now. It looks like Manish Adukia has dropped off, we will move to Vivekanand Subbaraman from Ambit Capital. Mr. Subbaraman, you may please un-mute yourself, introduce yourself and ask your question now. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 8 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Vivekanand Subbaraman - Ambit Capital This is Vivekanand Subbaraman from Ambit',
        'Mr. Subbaraman, you may please un-mute yourself, introduce yourself and ask your question now. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 8 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Vivekanand Subbaraman - Ambit Capital This is Vivekanand Subbaraman from Ambit. Couple of questions, first one is these circles where you have some administrative spectrum coming up for renewal in 2024, like UP-East, West Bengal, and others, do you envisage spending meaningful amounts of money in the next three years in subsequent spectrum auctions? That is question number one. Question number two is on the B2B growth acceleration, so Gopal, if you could just touch up on these factors that are responsible for the faster growth, I am referring to the year-on-year growth which has been on an upward trajectory and this is not just the current quarter obviously we have seen this play out in the last several quarters, so perhaps if you could help us understand whether it is because of the market growth itself, market share gains and if you can just give us a break for colour or an update on the digital products and the new products that you launched in the B2B side and there the revenue contribution is coming from those areas? Thank you. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited As I mentioned that we believe that there is very-very little amount of spectrum spend that will happen over the next few years. There are a few circles they are very small the number of circles now. It is only the ones that are coming up for renewal in 2024, which still have some admin spectrum but we have already bolstered through a substantial amount of spectrum even in the low band',
        'Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited As I mentioned that we believe that there is very-very little amount of spectrum spend that will happen over the next few years. There are a few circles they are very small the number of circles now. It is only the ones that are coming up for renewal in 2024, which still have some admin spectrum but we have already bolstered through a substantial amount of spectrum even in the low band. Like for example if you take Odisha, we bought liberalized spectrum even though we have admin spectrum, we did the same in West Bengal, we have done the same in UP East where 6.2 megahertz of 900Mhz will come up for renewal but we already have five megahertz of liberalized spectrum that we bought. So we have tried to bolster our holdings the same thing we did in North East & Assam so I am quite relaxed actually about the renewals component. In some cases in some places wherever there is some traffic or requirement or so on, we may bolster it but it is not essential since we are sort of running it on existing liberalized spectrum already. B2B has been a carefully crafted strategy as far as we are concerned and let me kind of comment in two parts. Number one is that we have really retooled our entire go to market on B2B, you know I mentioned before that 80% of our revenues come from 20% of our accounts. These are the most important accounts for us but I see as a glass half full so the 80% of revenue that comes from 20% of accounts we are working with them through the Customer Advisory Board to bring more solutions, more services, this is where you know Airtel IQ - the CPaaS part of it, Cybersecurity which is Airtel Secure, SD-WAN and so on and so forth are all coming in to actually create a greater share of wallet. We have seen some fabulous stories where we have been able to raise our average revenue per account quite substantially through a concerted effort',
        'We have seen some fabulous stories where we have been able to raise our average revenue per account quite substantially through a concerted effort. The second part that we have is the 20% of revenue which comes from 80% of accounts. This is where we have again retooled our GTM to really have a bunch of people looking at hunting for these accounts and this is now true for the large enterprises as well as the top end of the medium enterprises and I will come to that in a moment. So I think that part we are tracking very closely, how many accounts have we added, what is the average revenue per account for each account. So that is the second part of our strategy. The third part is really around the SME side, where we spend a lot of money and time to build a digital marketing tool to create leads, so we are now getting almost 30% to 35% of leads coming from online and digital. These leads go to actually drive the SME growth which is the third part of our approach. The fourth part of our approach is that we have in source or we have done away with the channel. The SME business used to be run by the channel and the channel would get commissioned. We have now done away with the channel and we have about 1200 people who we were brought into our own company who are responsible for the SME. The entire go-to-market architecture is something that has been a very important reason for our growth and while we have done this we have taken out one layer in the organization so that the people who are serving customers are closer to the leadership. The second part on B2B is really to develop more and more solutions and services and this is where we have spent a substantial amount of our time through building our own services as well as through partnering. Taking small bets and acquiring a few companies to create a stronger portfolio of services which can also build our B2B business. Airtel IQ is one, Airtel Secure is another',
        'The second part on B2B is really to develop more and more solutions and services and this is where we have spent a substantial amount of our time through building our own services as well as through partnering. Taking small bets and acquiring a few companies to create a stronger portfolio of services which can also build our B2B business. Airtel IQ is one, Airtel Secure is another. Airtel IQ has been building strong traction, our CPaaS business is looking very strong. So, it is the whole combination of all of this on the B2B side. Vivekanand Subbaraman - Ambit Capital Very helpful. Just one small followup as far as capex and spectrums purchases are concerned, we have not called the second call of rights issue despite participation in the 5G auction, so I am just trying to understand what are the instances in the next 12 to 24 months where you call and what would that be used for now? Harjeet Kohli - Joint Managing Director, Bharti Enterprises Vivek, this is Harjeet. I think the rights issue about Rs 15,000+ Crores of residual calls are pending and really in the coming few months, we need to evaluate basis the business cash flow profile, which by the way now is organically both significantly positive in India and Africa, at what point in time should we call for it. Gopal also mentioned the ongoing dynamic play of some bit of acceleration of capex, so we will stay agile to what this business cash flow profile is and accordingly take it all in the few months on the residual calls for rights. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 9 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Vivekanand Subbaraman - Ambit Capital Thank you and all the best. Rajyita – Moderator Thank you very much Mr. Subbaraman. The next question comes from Mr. Abhiram Iyer from Deutsche Bank',
        'Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 9 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Vivekanand Subbaraman - Ambit Capital Thank you and all the best. Rajyita – Moderator Thank you very much Mr. Subbaraman. The next question comes from Mr. Abhiram Iyer from Deutsche Bank. Mr. Iyer you may please un-mute yourself, introduce yourself and ask your question now. Abhiram Iyer - Deutsche Bank This is Abhiram Iyer from Deutsche Bank. Thank you for taking my question and congratulations on a good set of results. My query was pertaining to you mentioned that you are going to bring forward some of the capex cycle over the next three years to maybe in the next 12 to 24 months due to the rollout of 5G may I know whether this would change your debt profile and you know your leverage targets? Do you expect to end the year maybe you know sitting at a higher leverage than what it is currently or and the other thing is obviously there is significant amount of operational cash flow available to the company given the nature and the quantum of capex that might be required, do you envisage coming to the debt markets? Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited No. Firstly let me just kind of say that you know the operating cash flows of the business are strong enough for us to fund any requirement of capex whatever it is whatever advancing we need to do. Yes, the debt on paper will get larger because of this large spectrum payment that will be due to the government. But the way we look at spectrum payments is that this is the annual payment which is over 20 years so it is no different from an operating cost and ultimately all of the spectrum sort of translates to revenue',
        'Yes, the debt on paper will get larger because of this large spectrum payment that will be due to the government. But the way we look at spectrum payments is that this is the annual payment which is over 20 years so it is no different from an operating cost and ultimately all of the spectrum sort of translates to revenue. So if you were to take that operating cost of the repayments that are due out of the Rs 43,000 Crores, Rs 3,600 Crores is really to be paid every year and the 43000 Crores does not go into debt and you pull the EBITDA down by Rs 3,600 Crores, we serve out of the existing EBITDA pool then the debt profile is actually getting very, very healthy given the operating cash flows so the leverage position is getting better and better every year and has been so over the last couple of years. Abhiram Iyer - Deutsche Bank Got it and just a follow up question more in terms of bookkeeping, but given the fact that we have called some of foreign exchange bonds. Is this going to be some sort of strategy going forward given that you know the FX rate has not been favorable for Indian companies, do we see you looking more to call some of their foreign debt and try and get more onshore debt? Harjeet Kohli - Joint Managing Director, Bharti Enterprises If you want me, Gopal, I will just quickly pitch in and just linking back Abhiram to what Gopal mentioned in the last question, if you see the EBITDA less capex, at least till now keeping aside some bit of acceleration, globally we will have about five billion dollars of operating free cash flow and our interest costs are well within two billion dollars. Considering, that the taxes are still you know fairly minimal, so we have significant free cash flow profile that allows us to keep the leverage reasonably well in check and also reducing fundamentally',
        'Considering, that the taxes are still you know fairly minimal, so we have significant free cash flow profile that allows us to keep the leverage reasonably well in check and also reducing fundamentally. Within that leverage, our FX leverage is significantly low as you can clearly see bulk of the Indian debt is DoT liabilities, the past AGR liabilities, there is a significant amount of FLO liabilities and only a smaller amount is really external debt of which a much smaller amount in FX. So in general the FX is fairly well managed in terms of the percentage of the outstanding that are in foreign exchange. The foreign exchange bonds that we have been calling off a) till date have been in Africa which I think is very welcome. There are two three reasons; one of course Africa is creating the cash flow profiles that are able to keep calling the bonds back. Second as Africa is a listed company, it stands on its own increasingly the guaranteed bonds from Airtel India are going down. If you would go back about three four years ago you know maybe prior to listing of Airtel Africa we had more than 11 to 12 billion dollars of guarantees given from Airtel India for the bonds of five to six billion dollars that were outstanding in Airtel Africa. Now that has gone down significantly. So that is another reason why that buyback is important. Third is we have to time the interest rate situations in the market which essentially means when the bonds can be bought back at a reasonable economic savings for the company is I think the right time. So we would we would not go just to reduce the FX leverage, I think the economics of the situation, the refinance cost and including the cash flow generations from both the engines - Africa and India will govern the further buyback actions. But in general, the company maintains capital markets access and we at least in the rupee market have been fairly active and if required will be active in the dollar markets too. Republished with permission',
        'So we would we would not go just to reduce the FX leverage, I think the economics of the situation, the refinance cost and including the cash flow generations from both the engines - Africa and India will govern the further buyback actions. But in general, the company maintains capital markets access and we at least in the rupee market have been fairly active and if required will be active in the dollar markets too. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 10 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Abhiram Iyer - Deutsche Bank Thank a lot for the comprehensive answer. Rajyita – Moderator Thank you very much Mr. Iyer. The next question comes from Mr. Aliasgar Shakir from Motilal Oswal. Mr. Shakir you may please un-mute yourself, introduce yourself and ask your question now. Aliasgar Shakir - Motilal Oswal Thanks for the opportunity. This is Aliasgar from Motilal Oswal. A couple of questions, Gopal, first one the 5G, so you did mention that the ecosystem is yet to develop on the handset device side you know we would probably in the next couple of years still be somewhere about mid teens kind of penetration levels. Just trying to understand what is the need to accelerate a 5G rollout, I mean is it to do with you know kind of matching competition or you think that you know I mean with our experience in 4G we do not want to kind of you know stay behind the curve or do you think that you know there is some possibility of better monetization opportunity left',
        'Just trying to understand what is the need to accelerate a 5G rollout, I mean is it to do with you know kind of matching competition or you think that you know I mean with our experience in 4G we do not want to kind of you know stay behind the curve or do you think that you know there is some possibility of better monetization opportunity left. So I am just trying to think about you know, why cannot we kind of wait for sometime, let the ecosystem develop and then kind of accelerate, we have already demonstrated that in our you know 4G rollout where we did accelerate it quite significantly well? Second question is you mentioned that we have not thought about how we will monetize 5G but it has been six months since the last tariff hike. Now that we plan to do 5000 town 5G rollout, how should we think of tariff hikes. Should it be direct tariff hikes or we should we see change in the plans from these unlimited plans that we have to more kind of limited plans or 5G related kind of pricing? Those are my two questions. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited On the tariff front, as I have said that tariffs need to go up. When they will go up I cannot comment on right now. Secondly, to change the construct of price plans in industry for us on a unilateral basis is always a challenge competitively. We have to be competitive in the market, ultimately in this business if you grow market share you are in actually a much better long-term position for investors as well as for customers and employees. So to that extent, we will do the right thing to remain competitive. I think we have debated this. We can evaluate this a lot, we can sort of wait for the ecosystem but ultimately it is just putting the same capex that anyway would have gone. Nothing is getting wasted. It is just putting it there a little ahead of time so that we are present where we need to be',
        "So to that extent, we will do the right thing to remain competitive. I think we have debated this. We can evaluate this a lot, we can sort of wait for the ecosystem but ultimately it is just putting the same capex that anyway would have gone. Nothing is getting wasted. It is just putting it there a little ahead of time so that we are present where we need to be. Of course, having said that there is a lot of analysis that happens on where we should go and which sites we should go to and all that based on devices. All of that is happening but we did see, for example, in some of the C-category circles when we were a little late on 4G, we lost a little bit of share early on and then we have now recovered that market share over a period of time. But in the cities and A-category towns where we launched head-to-head with our competition, we had a much better outcome. So I would say that it is just a question of bringing forward capex without trying to finesse this too much and intellectualize this too much. I think that's the way we see it. You are right, if this market was uncompetitive and it was a benign kind of market with very little competitive intensity, we may have said we can even buy the spectrum next year because there is nothing going away. So to that extent the fact is that we believe that we need to be leading the narrative in terms of what we do with our customers, make sure that our best quality customers not just stay with us but we are able to attract good quality customers onto our portfolio. I think that is really what we are trying to do. Aliasgar Shakir - Motilal Oswal Understood, this is very helpful. Thank you. Rajyita – Moderator Thank you very much Mr. Shakir. Due to time constraints, I would now hand over the proceedings to Mr. Gopal Vittal for closing remarks. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited Well I do want to thank you for tuning in",
        'I think that is really what we are trying to do. Aliasgar Shakir - Motilal Oswal Understood, this is very helpful. Thank you. Rajyita – Moderator Thank you very much Mr. Shakir. Due to time constraints, I would now hand over the proceedings to Mr. Gopal Vittal for closing remarks. Gopal Vittal - Managing Director & Chief Executive Officer - Bharti Airtel Limited Well I do want to thank you for tuning in. I did want to spend a substantial amount of time on 5G and give you the confidence as to why we believe we are very-very well positioned. I hope I have been able to do that in the last hour and I hope also that we have been able to you know clarify to you what are overall approaches on 5G and why some of the narratives that are doing the rounds are probably not based on fact or not founded on fact and I hope that has given you the confidence. So with that let me sort of sign off and see you next quarter. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 11 - Transcript of Bharti Airtel Limited First Quarter Ended June 30, 2022 Earnings Conference Call Final Transcript Rajyita - Moderator Thank you everyone for joining us today. Recording of this webinar will also be available on our website for your reference. Republished with permission. No part of this publication may be reproduced or transmitted in any form or by any means without the prior written consent of Bharti Airtel Limited. - 12 -'
        ]
    final_summary= get_final_output(airtel_chunks)
    print("Final summary")
    print(final_summary)
    

main()
