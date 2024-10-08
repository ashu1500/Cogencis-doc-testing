import os
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import subprocess
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from accelerate import Accelerator
import re
import datetime
import torch
import numpy as np
from accelerate import Accelerator

accelerator = Accelerator()


def load_llama_model():
    ''' Load llama model from the local folder'''
    try:
        print("llama model loading")
        hf_token="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"
        subprocess.run(f'huggingface-cli login --token={hf_token}',shell=True)
        model_path= os.path.join("model")
        model_pipe = pipeline(task="text-generation", model = model_path,tokenizer= model_path,device_map="auto")
        model_pipe= accelerator.prepare(model_pipe)
        final_pipeline= HuggingFacePipeline(pipeline = model_pipe, model_kwargs = {'temperature':0})
        print("model loaded successfully")
        return final_pipeline
    except Exception as e:
        print(e)
        raise e

llm_model= load_llama_model()

# Overall document summary

def generate_embeddings(e5_model,chunk_text):
    ''' Generate embeddings for the document chunks'''
    try:
        chunk_embeddings= e5_model.encode(str(chunk_text), normalize_embeddings=True)
        return chunk_embeddings
    except Exception as ex:
        print(f"Error in generate embeddings. {ex.args}")
        raise ex

def merge_strings_optimized(strings):
    ''' Merge strings if string length is less than 300'''
    try:
        merged_list = []
        buffer = ""  

        for string in strings:
            if len(buffer) == 0:
                buffer = string
            else:
                if len(buffer) < 300:
                    buffer += ' ' + string
                else:
                    merged_list.append(buffer)
                    buffer = string

        if buffer:
            merged_list.append(buffer)

        return merged_list
    
    except Exception as ex:
        print(f"Error in merging optimized chunks: {str(ex)}")
        raise ex


def extract_summary_section_perchunk(text):
    """Post processing to extract summary section from the text."""
    try:
        keyword = "SUMMARY:"
        keyword_pos = text.find(keyword)
        if keyword_pos != -1:
            summary = text[keyword_pos + len(keyword):].strip()
            return summary
        else:
            print("Keyword 'SUMMARY' not found in the text.")
            return None
    except Exception as ex:
        print(f"Error in extract summary section per chunk. {ex.args}")
        raise ex


def clean_summary(summary):
    ''' Remove instruction-like content from the summary '''
    # Simple heuristic: remove any sentence that refers to writing a summary or following instructions
    instruction_phrases = [
        r"task of writing the summary",
        r"asked to write a concise summary",
        r"delimited by triple backquotes",
        r"avoid adding any information",
        r"must strictly contain",
        r"do not include any kind"
    ]
    
    for phrase in instruction_phrases:
        summary = re.sub(rf".*{phrase}.*", "", summary, flags=re.IGNORECASE)
    
    # Return cleaned summary
    return summary.strip()

def get_chunk_summary(llm, text):
    ''' Get summary of each chunk '''
    try:
        
        template = """
        Write a concise summary of the following text, which is delimited by triple backquotes:
        - The summary must strictly contain only factual information present in the text.
        - Avoid adding any information that is not explicitly mentioned in the text.
        - The summary must be in a **single, continuous paragraph without any line breaks**.
        - The summary must avoid bullet points, lists, or names.
        - Use third-person language (e.g., 'they', 'their') and avoid first-person pronouns like 'we', 'our', or 'us'.
        - Do not include any kind of headers, emojis, asterisks, symbols, requests, questions, or instructions in the summary.
        - Do not include any introductory or closing statements such as "I have written" or "let me know if it meets your requirements." Only output the summary itself.
        - Ensure the summary flows logically from start to end.
        - Do not interpret, analyze, or infer any content; only summarize the given text.
        - Do not include any note or instructions in the summary.
        - Avoid restating any part of these instructions in the summary.
        - Ensure the entire summary is output as one continuous paragraph.
        ```{text}```
        SUMMARY:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["text"])
        formatted_prompt = prompt.format(text=text)
        text_summary = llm.generate([formatted_prompt])

        chunk_summary = extract_summary_section_perchunk(text_summary.generations[0][0].text.strip())
        cleaned_summary = clean_summary(chunk_summary)
        return cleaned_summary

    except Exception as ex:
        print(f"Error generating summary: {ex}")
        raise ex

   
def get_overall_document_summary(llm_model, chunk_list):
    ''' Get overall summary of the document '''
    try:
        overall_summary = []
        refined_chunk_list= merge_strings_optimized(chunk_list)
        for text_chunk in refined_chunk_list:
            print("Chunk summary started")
            summary = get_chunk_summary(llm_model, text_chunk)
            overall_summary.append(summary.strip()) 
            print("Chunk summary generated")
        
        # Join all chunk summaries with a space separating them
        final_summary = " ".join(overall_summary)
        
        # Final cleaning step to remove any lingering instruction-like content
        final_summary = clean_summary(final_summary)
        
        return final_summary
    
    except Exception as e:
        print(f"Error generating overall summary: {e}")
        raise e

def split_paragraph(paragraph):
    ''' Split paragraph into summary points'''
    try:
        abbreviations = {
            'Mr.': 'Mr_placeholder',
            'Mrs.': 'Mrs_placeholder',
            'Ms.': 'Ms_placeholder',
            'Dr.': 'Dr_placeholder',
            'Prof.': 'Prof_placeholder',
            'St.': 'St_placeholder',
            'Mr. C': 'Mr_C_placeholder'  
        }

        for abbr, placeholder in abbreviations.items():
            paragraph = paragraph.replace(abbr, placeholder)
    
        split_regex = r'(?<!\d)[.](?!\d)'
        points = re.split(split_regex, paragraph)
        points = [point.strip() for point in points if point.strip()]
        for i, point in enumerate(points):
            for abbr,placeholder in abbreviations.items():
                points[i] = points[i].replace(placeholder, abbr)
        return points
    except Exception as ex:
        print(f"Error in split paragraph. {ex.args}")
        raise ex

def remove_unnecessary_emojis(text_data):
    ''' Remove unwanted emojis from the text'''
    try:
        emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
        
        final_text= re.sub(emoji_pattern,'',text_data)
        final_text = final_text.replace("\u230f", "").replace("\u2139", "")
        final_text = final_text.replace("\ud83e\udd14","")
        return final_text
    except Exception as ex:
        print(f"Error in remove unnecessary emojis. {ex.args}")
        raise ex

def reduce_overall_summary(text, limit=12000):
    ''' Reduce overall summary in a given word limit'''
    try:
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) == 1:
            words = text.split()
            reduced_text = []
            current_length = 0
            
            # Add words until the limit is reached
            for word in words:
                if current_length + len(word) + 1 > limit:
                    break
                reduced_text.append(word)
                current_length += len(word) + 1  # +1 for the space
            
            return ' '.join(reduced_text).strip()
        reduced_text = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > limit:
                break
            reduced_text.append(sentence)
            current_length += sentence_length
    
        return ' '.join(reduced_text).strip()
    except Exception as ex:
        print(f"Error in reducing overall summary. {ex.args}")
        raise ex

def remove_similar_mda_overall_summary(embedding_model,overall_summary):
    ''' Check similarity between MDA summary points'''
    try:
        print("Removing similar MDA summary points")
        indices_to_remove=set()
        processed_summary= overall_summary.replace('*','')
        final_summary= remove_unnecessary_emojis(processed_summary)
        summary_points= split_paragraph(final_summary)
        summary_embeddings= [generate_embeddings(embedding_model,summary) for summary in summary_points]
        for i in range(len(summary_embeddings)):
            for j in range(i+1,len(summary_embeddings)):
                if (cos_sim(summary_embeddings[i],summary_embeddings[j]).item())>0.89:
                  indices_to_remove.add(j)
        filtered_summary_points = [point for idx, point in enumerate(summary_points) if idx not in indices_to_remove]
        keywords = ["management team", "presentation", "website", "recording","q&a","representative","investor","greetings","discussion","date","upload","morning","afternoon","call"]
        filtered_summary = [summary_point for summary_point in filtered_summary_points if not any(keyword.lower() in summary_point.lower() for keyword in keywords)]
        refined_summary= ".".join(filtered_summary)
        final_summary= reduce_overall_summary(refined_summary)

        return final_summary
    except Exception as ex:
        print(f"Error in remove similar MDA overall summary points. {ex.args}")
        raise ex

















def theme_extraction_per_chunk(chunk_text, llm):
    ''' Extract themes for each chunk'''
    try:
        # template = """<s>[INST] <<SYS>>
        # You are a helpful assistant. Generate concise and relevant key headers based on the financial information in the text.
        # Ensure the key headers are specific, contextually complete, and avoid any ambiguous or overly broad statements
        # <</SYS>>
        # Generate exactly 2 short and concise key headers (maximum 3-4 words each) from the following text. No explanations needed and don't include company name, numbers,country names or person names in key headers.The key haeders must be complete and meaningful.Avoid long phrases or sentences, and ensure the key header is a complete concept or topic. Avoid overly specific details such as timeframes, numbers, or minor specifics. Focus on capturing the essence of the information.
        # text: {text}
        # key headers:
        # """
        template = """<s>[INST] <<SYS>>
        You are a helpful assistant. Generate concise, specific, and complete headers based on the financial information in the text. 
        headers should be suitable as stand-alone titles and reflect complete concepts without being overly broad or incomplete.
        <</SYS>>
        Generate exactly 2 headers from the following text. 
        - Headers must be 3-4 words long and concise, strictly not exceeding 4 words. They should be fully formed, meaningful, and complete.
        - Do not include company names, numbers, country names, or person names.
        - Avoid generating full sentences, explanations, or long phrases. Focus on concise, well-defined topics that can be used as titles.
        - Avoid generating incomplete or partial comparisons (e.g., "X vs Y") or any unfinished phrases (e.g., "accounted for").
        - Do not use overly simplistic terms such as "improves" or "built," or vague phrases that do not convey a complete topic.
        - Do not generate long sentences, explanations, or phrases that read like full sentences (e.g., "Our commitment to bringing...").
        - Focus on capturing the core essence of the topic without minor details or excessive specificity, such as dates or figures.
        - Headers should be specific and contextually complete, ensuring they can stand alone as a title and are not too broad or vague.
        text: {text}
        headers:
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        result = llm.generate([prompt.format(text=chunk_text)])
        return result
    except Exception as ex:
        print(f"Error in theme extraction per chunk. {ex.args}")
        raise ex

def extract_headers_from_themes(output_text):
    ''' Get headers list for themes'''
    try:
        start_index = output_text.find("headers:")
        themes_section = output_text[start_index:]
        themes_lines = themes_section.split('\n')
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
    except Exception as ex:
        print(f"Error in extract headers from themes. {ex.args}")
        raise ex
    
def extract_headers_from_question_themes(output_text):
    ''' Get headers list for themes'''
    try:
        start_index = output_text.find("header:")
        themes_section = output_text[start_index:]
        themes_lines = themes_section.split('\n')
        themes_lines = [line.strip() for line in themes_lines[1:] if line.strip()]
        headers_list = []
        theme_line= themes_lines[0]
        if theme_line.strip().startswith(tuple(f"{i}." for i in range(1, 11))):
          if ":" in theme_line:
            header = theme_line.split(":")[1].strip()
            headers_list.append(header)
          else:
            header = theme_line.split(".")[1].strip()
            headers_list.append(header)
        else:
          headers_list.append(theme_line)

        return headers_list
    except Exception as ex:
        print(f"Error in extract headers from question themes. {ex.args}")
        raise ex

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
        final_themes= set(list(map(lambda x: str(x).title(), actual_chunk_headers)))
        return final_themes
        
    except Exception as ex:
        print(f"Error in get final transcript themes. {ex.args}")
        raise ex


def question_theme_extraction_per_chunk(chunk_text, llm):
    ''' Extract themes for each chunk'''
    try:
        template = """<s>[INST] <<SYS>>
        You are a helpful assistant. Generate concise, specific, and complete header based on the financial information in the text. 
        header should be suitable as stand-alone titles and reflect complete concepts without being overly broad or incomplete.
        <</SYS>>
        Generate exactly one short and concise header from the following text. 
        - Header must be 3-4 words long and concise, strictly not exceeding 4 words. It should be fully formed, meaningful, and complete.
        - Do not include company names, numbers, country names, or person names.
        - Avoid generating full sentences, explanations, or long phrases. Focus on concise, well-defined topic that can be used as title.
        - Avoid generating incomplete or partial comparison (e.g., "X vs Y") or any unfinished phrases (e.g., "accounted for").
        - Do not use overly simplistic terms such as "improves" or "built," or vague phrases that do not convey a complete topic.
        - Do not generate long sentences, explanations, or phrases that read like full sentences (e.g., "Our commitment to bringing...").
        - Focus on capturing the core essence of the topic without minor details or excessive specificity, such as dates or figures.
        - Header should be specific and contextually complete, ensuring it can stand alone as a title and is not too broad or vague.
        text: {text}
        header:
        """

        prompt = PromptTemplate(template=template, input_variables=["text"])
        result = llm.generate([prompt.format(text=chunk_text)])
        return result
    except Exception as ex:
        print(f"Error in question theme extraction per chunk. {ex.args}")
        raise ex

def get_final_question_themes(llm,input_list):
    '''Get final themes for the transcript document'''
    try:
        chunk_headers_list=[]
        all_chunk_header=[]
        actual_chunk_headers=[]
        for items in input_list:
            print("Theme generation")
            chunk_txt= question_theme_extraction_per_chunk(items,llm)
            print("Chunk text generated")
            chunk_header= extract_headers_from_question_themes(chunk_txt.generations[0][0].text)
            chunk_headers_list.append(chunk_header)
            print("Chunk header generated")
        for header in chunk_headers_list:
            all_chunk_header+=header
        print("All themes generated")
        ls=[actual_chunk_headers.append(x) for x in all_chunk_header if x not in actual_chunk_headers]
        final_themes= set(list(map(lambda x: str(x).title(), actual_chunk_headers)))
        return final_themes
        
    except Exception as ex:
        print(f"Error in get final question themes. {ex.args}")
        raise ex


#Theme based summary

def extract_summary_section_perchunk(text):
    """Post processing to extract summary section from the text."""
    try:
        keyword = "SUMMARY:"
        keyword_pos = text.find(keyword)
        if keyword_pos != -1:
            summary = text[keyword_pos + len(keyword):].strip()
            return summary
        else:
            print("Keyword 'SUMMARY' not found in the text.")
            return None
    except Exception as ex:
        print(f"Error in extract summary section per chunk. {ex.args}")
        raise ex

def summary_generation_perchunk(theme, text, llm, num_points):
    """Generate summary for each chunk based on the keywords."""
    try:
        print("Entering summary generation per chunk")
        template = f"""
        Generate a summary consisting of exactly {num_points} bullet points based on the following text. 
        - Each bullet point must be at least 20 words long, complete, and relevant to the theme '{theme}'. If a point is less than 20 words, expand it with only relevant information.
        - Start each point with a standard bullet (•). Do not use any other symbols, emojis, or icons.
        - Do not change or substitute words in a way that alters the meaning of the bullet point.
        - The summary must be written in the third person, using third-person terms like 'they', 'their' or specific nouns and do not use words like 'We', 'Our','us'.
        - Maintain the original tense of the sentences in the text. If a sentence is in the present tense, the summary should also be in the present tense.
        - Avoid all headers, titles, or additional comments. Only the bullet points should be present.
        - Ensure that each bullet point is a complete sentence with proper context, strictly based on the provided text.
        - Do not omit any numerical values or specific details mentioned in the text.
        - Do not introduce any information, details, or numerical values that are not directly mentioned in the text. 
        - Avoid analysis, interpretations, or inferences not directly mentioned in the text.
        - Do not start points with ambiguous phrases without clearly specifying what is being referred to; ensure every point is understandable on its own.
        - Ensure that each bullet point clearly specifies what specific topic is being discussed so that it is understandable without needing additional context.
        - Points should be clear, logical, and sequenced coherently, strictly adhering to the content of the provided text.

        TEXT:
        {text}

        SUMMARY:
        """
        prompt = PromptTemplate(template=template, input_variables=["text", "theme"])
        result = llm.generate([prompt.format(text=text, theme=theme)])
        final = extract_summary_section_perchunk(result.generations[0][0].text)
        return final
    except Exception as ex:
        print(f"Error in summary generation per chunk. {ex.args}")
        raise ex
    finally:
        print("Exiting summary generation per chunk")

def remove_unnecessary_emojis(text_data):
    ''' Remove unwanted emojis from the text'''
    try:
        final_text=[]
        text_lines= text_data.split('\n')
        bullet_point = '•'
        emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
        for point in text_lines:
            final_point= re.sub(emoji_pattern, lambda match: match.group(0) if match.group(0) == bullet_point else '', point)
            final_text.append(final_point)
        processed_summary = '\n'.join(final_text)
        return processed_summary
    except Exception as ex:
        print(ex)
        raise ex

def remove_unwanted_headers(text):
    """Remove numbered headers and generate as bullet points"""
    try:
        lines = text.strip().split('\n')
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
                    processed_lines.append(processed_line)
                else:
                    processed_line = line.strip()
                    processed_lines.append(processed_line)
    
            else:
                processed_lines.append(line.strip())
        processed_text = '\n'.join(processed_lines)
        final_processed_text = re.sub(r'\n\n+', '\n', processed_text)
        return final_processed_text
    except Exception as ex:
        print(f"Error in remove unwanted headers. {ex.args}")
        raise ex


def generate_theme_summary(theme,chunk_list,llm):
    try:
        print("Entered theme summary generation")
        theme_summary=""
        for chunk in chunk_list:
            if len(chunk)>0 and len(chunk)<300:
                chunk_summary= summary_generation_perchunk(theme,chunk,llm,1)
            elif len(chunk)>=300 and len(chunk)<500:
                chunk_summary= summary_generation_perchunk(theme,chunk,llm,2)
            elif len(chunk)>=500 and len(chunk)<1000:
                chunk_summary= summary_generation_perchunk(theme,chunk,llm,3)
            elif len(chunk)>=1000:
                chunk_summary= summary_generation_perchunk(theme,chunk,llm,6)
            chunk_summary_list= chunk_summary.split('\n')[:5]
            chunk_summary_list = list(map(str.strip, chunk_summary_list))
            actual_chunk_summary= '\n'.join(chunk_summary_list)
            theme_summary+='\n'
            theme_summary+= actual_chunk_summary

        filtered_summary= theme_summary.replace('*','')
        # removed_summary= remove_unnecessary_emojis(filtered_summary)
        # processed_summary= remove_unwanted_headers(removed_summary)
        # new_processed_summary= processed_summary.replace('•','\n•')
        
        return filtered_summary
    except Exception as ex:
        print(f"Error in generate theme summary. {ex.args}")
        raise ex


def get_document_theme_summary(chunk_dictionary,llm):
    '''Get theme-based summary of document'''
    try:
        print("Entering document theme summary")
        theme_based_summary={}
        for theme,chunk in chunk_dictionary.items():
            if chunk:
                print("Theme summary started")
                theme_based_summary[theme]= generate_theme_summary(theme,chunk,llm)
                print("Theme summary generated")
                print(datetime.datetime.now())
            else:
                continue
        final_theme_based_summary = {k: v for k, v in theme_based_summary.items() if v is not None and v.strip() not in ('')}
        return final_theme_based_summary
    except Exception as ex:
        print(f"Error in get document theme summary. {ex.args}")
        raise ex
    
    finally:
        print("Exiting document theme summary")






def main():
    indigo_discussion_data= ['Good evening, everyone, and thank you for joining us for the first quarter of fiscal year 2025 earnings call. We have with us our Chief Executive Officer - Pieter Elbers and our Chief Financial Officer — Gaurav Negi to discuss the financial performance and are available for the Q&A session.',
                            'Please note that today’s discussion may contain certain statements on our business or financials which may be construed as forward-looking. Our actual results may be materially different from these forward-looking statements.',
                            'The information provided on this call is as of today’s date and we undertake no obligation to update the information subsequently. We will upload the transcript of prepared remarks by day end. The transcript of the Q&A session will be uploaded subsequently.',
                            'With this, let me hand over the call to Pieter Elbers. For the first quarter of the financial year 2025, we reported a quarterly total income of 202 billion rupees, which is an increase of 18% as compared to same period last year. In terms of profitability, we reported a profit after tax of 27.3 billion rupees — 2,729 crore rupees with a profit after tax margin of around 14%. With these results, we have reported seven consecutive quarters of profitability.',
                            'We proudly served around 28 million customers during the quarter. And I would like to express my gratitude to each customer for choosing to fly with us. Our customers are the reason we do what we do and we remain committed to making every journey with us a memorable and enjoyable experience. We take into consideration all the feedback that we receive and makes changes to our product & services accordingly. I would like to highlight some of the recent changes that we have made:First of all, we are in the process of launching a tailor-made business product for the nation’s busiest and business routes',
                            'We have introduced a web check-in seat selection feature, specifically for women travellers. We are in the process of revamping our website and mobile application for enhanced customer experience. And, we have added a hotel booking option for our customers on our website and appAdditionally, on the operations side we are constantly expanding the role of technology in order to serve our customers better as recently we became the first airline in India to get approval from the regulator for Electronic Flight Folder that will enable reduction in the time spent on pre-flight preparations, smoother operations and will add to the overall operational efficiency.',
                            'We will be unveiling a lot more details on some of these changes next week during our 18th Anniversary celebration. When we first took to the skies, we envisioned an airline that would not only transport passengers from one destination to another but would also connect the vast and diverse cultures across India. And over the past 18 years, we have grown to the world’s seventh largest airline, in terms of daily departures and have become India’s most preferred airline.',
                            'Since our inception, India’s and IndiGo’s growth story have been closely interlinked. And we believe as India gears up to become the third-largest economy in the world, it’s important that we devise our strategy to, not only capitalize the opportunity, but also stay ahead of the curve. The Indian market, as we shared earlier, is still largely underpenetrated when it comes to air travel. And even more so, when it comes to international air travel. So, there is a buoyant market out there, hand in hand with the growth of the Indian economy.',
                            'With IATA’s 81st AGM to be held in Delhi in 2025, India’s aviation industry potential is being recognized, also at a global stage. And we are proud to be the host airline for IATA AGM next year and look forward to welcoming the global aviation community to our home country.',
                            'Expanding our network is a key part of our growth strategy. And we have added 30 new routes on a year-over-year basis as we fly to more than 540 routes currently. Further, in line with growing demand we are further enhancing our international capacity deployment to Central Asia and as from mid of August we will be flying daily to the cities we added last year - Tashkent, Almaty and Tblisi.',
                            'As guided at the starting of the year, we will continue to add more destinations and frequencies in the coming quarters to grow by early double digits in the financial year 2025. We have the Airbus XLRs joining our fleet next year which will allow us to reach the southern parts of Europe and further into Asia as well. Then to further expand our range, we have the Airbus A350-900 coming in from the year 2027.',
                            'In addition to that, our codeshare networks have been a very important pillar of our international strategy. Our experience of partnerships with multiple global airlines has exposed us to the needs and preferences of international customers, preparing us for the next stage of operations as so that we are able to apply the learnings once we have widebodies. Recently, we have announced codeshare partnership with Japan Airlines, under which Japan Airlines’ customers will be able to seamlessly travel to 14 Indian cities through Delhi and Bengaluru. All of these elements are building blocks of the very same cohesive strategy towards internationalization.',
                            'Operationally, over the past few quarters, as the air traffic in India continues to grow and infrastructure enhancements are being implemented across various stations, we have been experiencing increased block times and congestion. We are working relentlessly and taking all possible internal actions such as adjusting schedules to reassure our customer promises.',
                            'Additionally, due to the unfortunate event of canopy collapse in Delhi’s Terminal 1, we had to shift our operations to other terminals in a very short period of time and very recently the massive IT outage globally, which was not limited to the airlines, impacted hundreds of our flights over a two-day timeframe. IndiGo teams immediately stood up and dealt with the situation to minimize the impact for our customers. And I would like to sincerely thank my 6E colleagues for going above and beyond to serve our customers in these extraordinary times.',
                            'While large global airlines are facing multiple headwinds currently due to various factors. Given the size of the opportunity that India presents, and our clear strategy combined with a strong execution by the IndiGo team I remain confident that we will continue to reach new heights. As we enter the next phase of our growth, we remain committed towards our goal of 2030 and are investing heavily to support our growth plans.',
                            'Let me now hand over the call to Gaurav to discuss the financial performance in more detail. Thank you, Pieter and good evening, everyone. For the quarter ended June 2024, we reported a net profit of 27.3 billion rupees with a net profit margin of 13.9% compared to a net profit of 30.9 billion rupees for the quarter ended June 2023.',
                            'We reported an EBITDAR of 58.1 billion rupees with an EBITDAR margin of around 30 percent compared to an EBITDAR of 52.1 billion rupees for the quarter ended June 2023. On the revenue side, we experienced a similar revenue environment as compared to same period last year as the passenger unit revenue, PRASK, came in at 4.54 rupees. The yields came in at5.24 rupees, an improvement of 1.3 percent as compared to the same period last year which was offset by a marginally lower load factors of around 87%. Our unit revenue, which is RASK, came in at 5.40 rupees, which is around 5.5 percent higher compared to the quarter ended June 2023 primarily driven by accruals of the compensation that we have finalized with the OEM for the AOG.',
                            'On the cost side, the fuel CASK increased by 10.5 percent primarily driven by increase in fuel costs we have witnessed year over yearWhen one reviews the CASK ex fuel we experienced a similar increase of around 11 percent in the June 2024 quarter as compared to same period last year.',
                            'Now excluding the impact of forex, the CASK ex fuel ex forex increased by around 9 percent, compared to the same period last year primarily driven by aircraft grounding related costs, annual contractual escalations, inflationary pressures, annual increments, and the investments we continue to make towards supporting the future growth in areas of digital technology, talent and new initiatives.',
                            'Moving to the grounding of aircraft, the current count of grounded aircraft remains range bound at mid-seventies. We are working with Pratt & Whitney towards constant supply of spare engines and basis the current estimates we expect the groundings to start reducing towards the start of the next year. Further as communicated earlier, during the quarter we have finalized an amendment to the existing agreement with Pratt & Whitney for providing us with a customized compensation in relation to the grounding of aircraft due to spare engine unavailability.',
                            'During the quarter, we inducted 15 aircraft of which 8 are from original order book and remaining 7 were inducted as part of our mitigation measures in the form of damp leases and secondary leases. As of June 30", we have a total of 382 aircraft of which 14 aircraft are on finance leases and 18 aircraft are on damp leases. In addition to this, subject to regulatory approvals we will induct 6 more aircraft from Qatar Airlines for the Doha route on wet or damp lease in the coming quarters.',
                            'As shared previously, in order to diversify our sources of financing we have added 14 aircraft on finance leases in the last three quarters. Going forward, we will continue to further diversify our sources and will add more aircraft on finance leases. The tenure of these finance leases is similar to the operating leases of 8-10 years, and we will have the right to own the assets at the end of the lease term at a nominal price.',
                            'In terms of recognition, in an operating lease we recognize the ROU and the lease liability at a present value of lease payments, which is then depreciated over the lease period and the interest is accrued on the lease liability. Supplementary Rentals & Maintenance is also accrued on the flying hours of the aircraft.',
                            'Whereas, when it comes to finance leases, the full value of the aircraft is capitalized in the ROU, which is then depreciated based on the component accounted in the aircraft value. Similarly, theinterest is also accrued on the full value of the aircraft, but over the lease term. As a result, one will witness a higher allocation of interest and depreciation in the earlier years which will taper down over the lease term and life of the asset. Additionally, in finance leases the maintenance costs are capitalized when the maintenance event is carried out.',
                            'We ended the quarter with a capitalized operating lease liability of 449.6 billion rupees and a total debt, including the capitalized operating lease liability of around 525.3 billion rupees. Our right to use assets at quarter end were 358.6 billion rupees.',
                            'Due to our strong financial performance, our liquidity has further improved as we ended the June quarter with a free cash of 220.9 billion rupees. We continue to utilize part of our free cash towards future growth and our goal to become a 600 plus aircraft airline by the end of the decade.',
                            'With our firm pending orderbook of 975 plus aircraft that will enable us to receive more than one aircraft per week and the mitigation measures we remain firm on our capacity guidance of early double digits for the financial year 2025. And for the seasonally weak second quarter, we are expecting to add around high single digits capacity as compared to the same period last year. Further, on the revenue side basis the early trends of July, we are estimating a stable revenue environment on a passenger unit revenue which is PRASK basis in the second quarter of financial year 2025 as compared to the same period last year.',
                            'We will be turning 18 soon, we are proud of the legacy that we have built, and we are carefully considering the changing needs of our customers and making product advancements accordingly. As we embark on the next chapter of our journey our robust strategy and our improved financial health will provide us with the resources to invest for the future, explore new frontiers and reach greater heights.',
                            'With this, let me hand it back to Richa.'
    ]
    indigo_questions_list= ["Now just to clarify on the kind of lease this is. What I understand is if you take a damp lease, that might be for, let's say, 6 months or 1 year or 1.5 years. And you have your normal leases, which are longer term, but generally, you tend to return your aircraft in 6 to 7 years. Now when you have taken these 2 planes, what do you mean by a shorter period? Is it like 3 years, 4 years or even shorter?",
                            'Could you repeat what you said about the recognition of operating lease versus finance lease? And how should we look at it from an accounting perspective impact on depreciation and finance line items?',
                            'This is clear. Gaurav, my second question is again on what the previous participant asked, so in terms of whatever you have accrued so far, since there are still 70 planes which are on lease, why should the amount come off, because the maintenance related to that should also be recognized every quarter, right? So, if you could just explain how this entire compensation would work going forward?',
                            "And just as a follow-up, have you heard anything from the regulators or the government regarding yields? Because there's been some noise in the political arena about the fares and all that. So, any follow-up from the government or the ministry concerned?",
                            "Two questions from my side. The first one on cost. If I see your cost ex of fuel ex of forex, it's gone up by about Rs.0.30 on a Y-o-Y basis. Would the Pratt & Whitney issue be accounting fora minority share of this increase, majority or dominant share of the increase?",
                            'Understood. So, the second question that I had was on the demand environment. The context is that maybe the top 3 airlines in India, the CEOs have basically suggested that air fares are low, both IndiGo, Akasa as well as Air India CEOs. Is the demand environment good enough for the sector to be taking price increases ahead of cost inflation in the next 12 months?',
                            'Sure. One related follow-up is a recognition on the compensation part, which happened. So, you mentioned that there was some recognition which happened in the other operating revenue and some bit of compensation also get adjusted into the supplementary rentals part. And if I remember right, in one of your earlier calls, you had also mentioned that the income gets adjusted in the other income line item. So, can you please explain, I mean, how does the attribution basically happen to each line item?',
                            "Right. And another bookkeeping question, your employee incentives that has been accounted for this quarter? Or is it -- it's not?",
                            "So, first of all, in terms of yield. So, you mentioned that the next quarter yields are going to be flat year-on-year. That confuses me a bit because I think the last year, second quarter, I think the capacity was up almost 20%. And this year, you are guiding much lower capacity growth. And then yet you're expecting a yield, a flattish yield. So, I mean, could you just break it down a bit and give us a bit more colour in terms of are you sort of reducing the fares to inflate the demand? Or is it like how much of it is like underlying yield strength and how much is because of the network mix? So, what exactly happened on the yield? So, if you could break that up and then give a bit more colour, that would be helpful.",
                            "Okay, fine. My second question is about more on the international side. So now, of course, given that you're not very far from taking XLRs' deliveries and then you are also preparing for wide bodies. So, I just want to understand in terms of taking the flight rights and especially the sixth freedom right, which is quite important since India is getting an international hub. So, how do you see them? Do you see any challenges at the moment? Have you not started preparation yet? Or are you finding it easy to get all the flying rights which you need for international operations? Could you please help on that?",
                            'Congrats on the good results. My first question is on compensation again. So past 3 quarters has seen total cumulative compensation of Rs.2,300 crores, Rs.2,400 crores. Is this attributable cost of Rs.2,400 crores for the same 3 quarters and net spread neutral for the company as such? And maybe now the quarterly impact or compensation would be Rs.800 crores -- to the tune of Rs.800 crores going forward?'
    ]
    # indigo_discussion_themes= get_final_transcript_themes(llm_model,indigo_discussion_data)
    # print("Discussion_themes: ",indigo_discussion_themes)
    # indigo_question_themes= get_final_question_themes(llm_model,indigo_questions_list)
    # print("Questions_themes: ",indigo_question_themes)
    
    # e5_embedding_model = SentenceTransformer('intfloat/e5-large')
    maruti_chunks_list= ["Management Discussion and Analysis.Overview.The year 2018-19 flagged off with promising economic outlook supported by benign inflation, favourable interest rates, close to normal rainfall forecast and strong global economic growth. In Q1, the Indian economy registered a robust growth of 8% which gave an indication of economic activities returning to near normal post the GST roll-out. However, the momentum gained at the start didn’t sustain during the rest of the year and the economy faced major challenges leading to a slowdown in domestic consumption in the later part of the year. The pace of global economy also slowed down and couldn’t provide meaningful support to the Indian economy. The Government and the RBI undertook a slew of measures to provide the necessary stimulus to the economy..FY'15 FY'16 FY'17 FY'18 FY'19 Source: CSO.India’s passenger vehicle market grew by 2.7% in 2018-19 against 7.9% in 2017-18. This is the lowest annual industry growth recorded in previous seven years. Among the three broad industry segments, the utility vehicles segment that accounts for about 28% of industry sales, grew by 2.1%. The other two segments, passenger cars and vans, grew by 2.0% and 13.1% respectively. The urban markets witnessed weak demand while the non-urban markets saw relatively better growth. The demand for diesel models continued its weakening trend, and its industry share declined from 40% to 36%..The Company posted a volume growth of 5.3% in passenger vehicles in the domestic market (against the industry growth of 2.7%). Including the Light Commercial Vehicle (LCV) segment, the Company’s domestic sales growth stood at 6.1%..Domestic Passenger Vehicle Industry Growth.Board’s Report Corporate Governance Report Management Discussion & Analysis Business Responsibility Report.Contrary to expectations, sales were also impacted in export markets due to country specific reasons..The year also witnessed adverse commodity prices and foreign exchange movement.",
                        'Due to weak market situation, the Company could not take adequate price increases to neutralise the increase in input costs..Higher expenditure on marketing and sales promotions did not generate proportionate volume increase as demand remained low, impacting the profit margins..However, the Company could partially off-set the impact of unfavourable factors by stepping up cost reduction measures..During the year, the business partnership between Suzuki Motor Corporation and Toyota Motor Corporation (TMC), Japan started taking shape..The Company is likely to benefit immensely from this partnership by gaining access to the new-age technologies and from the mutual supply of vehicles..The Company has always endeavoured to provide clean technology in its products. India is at a nascent stage of using clean automotive technologies and the Company aims to be a front-runner in providing clean technology to the mass market. Using hybrid technology is the first step in this direction..This partnership with TMC is helping the Company to gain access to the hybrid technology. If the Company were to develop this technology on its own, it would take considerable time and significant investments. Also, many emission and safety related regulations are coming in the near future requiring more resources..Sourcing hybrid technology from Toyota could free-up the Company resources to devote them on other priorities..Combining the global volume of Suzuki and Toyota will provide a significant scale and make technology more affordable specially for a price-sensitive market like India..This partnership is also bringing opportunity to increase sales volume of the Company’s models by selling through Toyota Kirloskar Motor India. The Company is offering Baleno, Vitara Brezza, Ciaz and Ertiga to Toyota. Automobile industry is highly capital intensive and requires lots of investments in products, technologies and facilities.',
                        'For realising adequate return on investments, increasing volume per model and per platform is the key. This arrangement will bring in incremental volume for the Company and help maximise volume per model/platform.',
                        'Domestic Sales.Passenger Vehicles.The Company strengthened its leadership position across all the three industry segments - passenger cars, utility vehicles and vans. The success of models launched in the past along with the positive response for all new model launches i.e. Ertiga and WagorR helped it enhance its sales performance..For the second year in a row, five best-selling models in India came from the Company..The shift in consumer demand towards petrol segment is now even more evident with this segment’s contribution to the passenger vehicle sales going up to 64% during the year. For the Company, the contribution of petrol segment in the domestic passenger vehicle sales, during the year, increased to 74.5%, an increase of 3.4%..Industry Petrol Diesel Mix.Source: SIAM.Board’s Report Corporate Governance Report Management Discussion & Analysis Business Responsibility Report.Amid weak market demand with fewer walk-in customers to the showrooms, the approach of reaching out to customers plays an important role. With an extensive know-how of varied geographies along with the support from all the stakeholders, the Company conducted relevant events across urban and non-urban markets. This significantly helped identify the potential customers..MSIL Sales Network',
                        'Light Commercial Vehicle (LCV).The Company entered the LCV segment in 2016-17 and in a very short span of time, its product Super Carry sold 23,874 units, registering a growth of 138% compared with that of the previous year. One of the fastest network expansion to 310 outlets spread across 230 cities is a testimony to the good market acceptance for the product..New thrust on pre-owned car sales.To make pre-owned car buying experience more appealing, the Company re-launched True Value with a new brand identity. To enable the new customer drive the pre-owned car with confidence, special emphasis is laid on buying and selling of good quality cars that are refurbished and certified by authorised True Value dealers. The consumers can select from an array of available cars through a user-friendly mobile application and visit the nearest outlet to conclude the final purchase. There are now a total of 247 independent True Value outlets across 150 cities..Spare Parts and Accessories.Since inception, the Company has laid special emphasis on the ease of availability of genuine parts and accessories to its valuable customers. With the ever-increasing product portfolio, the Company manages the distribution of over 70,000 unique parts across the country. Every year, the Company augments its forecasting capabilities to achieve optimum inventory levels. The commencement of Siliguri warehouse in the East India and the expansion of Bengaluru warehouse operations will help the Company better serve its customers..During the year, the Company extended round-the- clock support to customer vehicles affected by the unfortunate floods in Kerala. Meticulous planning along with all stakeholders helped us enhance our response time during crisis..Exports.During the year, many export markets faced economic and political uncertainties leading to a 13.7% decline in exports of the Company.',
                        'Particularly, an East Asian country, a large export market for the Company put a sudden embargo on theimport of cars. Shipments to some Latin American markets suffered due to the prevailing political environment. Restrictions on retail financing and market skew towards used cars continued to pose pricing challenge in some of the African and SAARC countries. The Company arrested the decline to a certain extent by enhancing exports to other markets and improving its service processes in some key markets..Service.The Company’s service network serves around 18.8 million customers annually. All efforts of the Company are focused around:.Accessibility.During the year, 211 dealer workshops were added to the network, the highest ever in a single year. The Company’s service network now has 3,614 service workshops covering 1,784 cities. Additionally, 1,490 Maruti Mobile Support vehicles (MMS) are providing doorstep service to ~80,000 customers per month. Quick Response Team (QRT) aims at providing emergency road assistance sought by customers in the event of vehicle breakdown. With presence in 250 cities, it has provided assistance to over 135,000 customers till date..Human capital development.Service quality depends on the skill level of workshop technicians. Relevant training is imparted to them through Suzuki Service Qualification System (SSQS) based on the global standards defined by Suzuki Group. During the year, the Company increased its training infrastructure ten-fold to 170 centers..Digitalisation.Digitalisation is helping enhance customer satisfaction. The introduction of Online Customer Approval System (OCAS) is one such initiative, wherein the customer is able to approve or reject the additional jobs recommended by the service advisor..Operations.The Company firmly believes that sustainability of the business depends on manufacturing defect-free and safer products.',
                        'To achieve this, deployment of advanced technology along with upskilling of man-power is essential specially at a time when the market demand is witnessing considerable fluctuations both in terms.of overall volume and choice of models. To meet this challenge, the manufacturing operations need to be flexible so that the production can be adjusted according to the demand scenario..The Company has increased the use of digital technologies in manufacturing processes, preventive maintenance of machines and installation of new manufacturing lines. This has led to significant improvements in eliminating defects and also in cost reduction. Simultaneously, skills of human capital are being upgraded to effectively manage the increased deployment of technology, sustain high operational efficiency and quality..Recently, the Company introduced a system that checks the weld quality of almost all the welding spots in a vehicle. When it comes to ensuring uninterrupted operations, the Company implemented a self-diagnostics system. This system pre-empts a possible machine breakdown, thereby aiding preventive maintenance measures for improved life and reduced downtime of machines. During the commissioning of new manufacturing lines, the Company uses Digital Mock Up (DMU) checks in which the manufacturability is checked with respect to ergonomics for enhanced comfort of workmen.',
                        'Conservation of Natural Resources and Environment Protection.The principle of 3R (Reduce, Reuse and Recycle) is a way of life for the Company. ‘Smaller, fewer, shorter, lighter and neater’ is the guiding principle based on which the Company’s operating processes are built. Be it continuous enhancement of efficiency in the operations or development of highly fuel-efficient cars, the Company’s commitment to preserve resources is persistently reinforced..In a bid to conserve natural resources used in energy generation, the Company uses heat recovery steam generator. The Company is already using solar power for lighting its manufacturing plants and office areas. Now, the Company has also started using solar power in manufacturing of cars. During the year, the Company commissioned a 312 kWp solar power plant at its Manesar facility. With this addition, the total solar power used in manufacturing of cars now stands at 1.3 MW. This has further strengthened Company’s power generation mix in favour of renewable energy..Towards environment protection, the Company not only conforms to laws and regulations but also strives to stay ahead. During the year, the Company voluntarily put in place a globally recognised mechanism for controlling hazardous substances in its vehicles. With the launch of new WagonR, the Company has adopted the globally acclaimed International Material Data Systems (IMDS) tool in controlling the use of Substance of Concern (SoC). Since long, the Company has done away with usage of SoC, and now with IMDS, it will be able to.quantify the recoverable and recyclable materials in its vehicles. The new WagonR is minimum 95% recoverable and 85% recyclable, ahead of regulations in India. Besides commitment to environment protection, this initiative reinforces the Company’s firm belief in the 3R practices..The Company’s Sustainability Report elaborates on the initiatives undertaken in this area.',
                        'Safety.The Company’s vision on Safety is ‘Zero Incident - Zero Human Injury and Zero Fire’. A 3-tier committee under direct supervision of the MD & CEO is making continual progress towards the improvement of safety systems and compliance to achieve the Company’s safety vision. With every near-miss or incident occurrence, root-cause analysis is carried out and accordingly theme-based safety improvements are suggested..Quality.With increasing use of technologies in the vehicle, scale of operations, product variants and customer expectation about product quality, the complexity of manufacturing components and vehicles is increasing. This makes Quality a continuous journey. In order to deliver defect free products, the Company not only focuses on excellence in its in-house production processes but also actively supports suppliers in manufacturing defect free components. Among the many thrust areas to achieve world class quality, following areas remain in sharp focus:.Supply Chain 1 Reinforcement of zero-defect methodology - going beyond six sigma The Company believes that for the utmost customer satisfaction, not a single product should have any defect. To achieve this, the Company has taken an initiative to establish zero defect lines at suppliers’ works. During the year, a series of sensitisation workshops by the Company helped workmen at 518 supplier plants enhance their knowledge and implement best practices to achieve zero defect production lines. 2 Strengthening quality management systems - adherence to rules Adherence to defined systems and processes by the suppliers continues to be the key focus area and the Company has moved from monitoring its suppliers to now helping them enhance their system compliance capabilities. Recurrence prevention - reinforcing the culture of identification of root cause The Company carefully reviews and assesses market and dealer feedback. Prompt and corrective actions are undertaken to prevent.recurrence of all issues.',
                        'The Company has established a defect-recurrence-prevention department to institutionalise the learning and take necessary countermeasures.(b) Capability Development.1 The Company is promoting the development of relevant workmen skills and capabilities through the setting up of DOJO training centre at suppliers’ works. The workmen are required to mandatorily go through off-line training programs in the centre and they are introduced to the shop floor only after acquiring the required skills The training is provided in simulated production conditions to equip workmen to do a high-quality job on the production lines.(c) Scale and Complexity.Quality defects start getting evident when sudden production ramp-up of a new model component occurs at the time of mass production. To prevent the defects, the Company adopted a process called Peak Production Verification Trial (PPVT) In this process, on a trial basis, the production is carried out of anew model component at full scale to see the kind of quality issues that can surface when mass production starts. Since there is a sufficient time gap between the trial production and the mass production, the suppliers get time to take countermeasures. Earlier, PPVT was done for selected components. Now it has been extended to all the components Consolidation of Tier-II Suppliers Adequate scale helps suppliers to invest in enablers which help produce consistent quality. Tier-Il suppliers have limitation to raise their scale and hence fall short in meeting the desired quality levels. Tier-Il supplier consolidation is one of the ways to provide sufficient scale and achieve consistency in quality..Human Resource.Our Philosophy.The Company always strives to promote a safe, healthy and happy workplace.',
                        'It creates and instills a culture of partnership attitude among its employees.Board’s Report Corporate Governance Report Management Discussion & Analysis Business Responsibility Repo!.The biggest strength of the Company lies in its healthy combination of top-down and bottom-up approach in decision making process towards empowering the employees. A major thrust is laid on the constant two-way communication, free flow of thoughts and mutual growth. Led by the MD & CEO, and enunciated across levels, communication has led to a marked change in the labour-management relations in recent years. A calendar for communication ensures two-way communication with employees across various levels.The unique strength of Employee-Employer Connect.The Company is a beneficiary of the unique strength it enjoys with its human resources. During the year, various channels, both digital and non-digital were improvised to interact better with the employees. A lot of thrust was given on digitisation of HR processes which led to enhanced employee experience owing to improved and faster responsiveness. Employees in turn, raise queries give feedback and participate in different policy and procedural decisions on these platforms.Along with industry-leading benefits the Company since 1989 has launched housing schemes to support workmen in their efforts to own a house. The Company’s relentless efforts have benefitted a large part of workforce.',
                        'In addition, under the Government’s recent flagship housing scheme program Deen Dayal Awas Yojana and Pradhan Mantri Awas Yojana, the Company has facilitated the entire process right from selecting land, negotiating price and appointing real estate firm for ensuring quality and timely construction of houses and also providing housing loan subsidies.During the — year, the Company and its three workmen-unions concluded a three year wage agreement in a congenial manner, and to the mutual satisfaction of all.The Company takes care of its employees’ health Periodic medical checkups, regular health talks from the experts in the field of medicine are organised. To create the awareness on health and promote well-being of the employees the Company not only encourages participation in various sporting events held in the country but also organises marathons exclusively for its employees. This initiative greatly helps the Company to engage positively with its employees.For larger connect and welfare of the families of employees, the Company has a calendar of events which includes expert career counselling for employees children, a gala family day, plant visits for family members and attractive rewards for innovators. In engaging the families of employees in communication, an in-house magazine and MD & CEO’s messages on special occasions play an important role.To address any grievance of its workmen including temporary workforce, the Company has a well-structured grievance _ redressal mechanism. Periodic grievance redressal camps are organised to hear the issues of workmen, if any, and take actions accordingly.',
                        'Talent Acquisition and People Development - Making the Workforce Future Ready.As the pace of business accelerates amidst an uncertain environment for the automobile industry led by technology and regulatory disruptions, the ability to react fast to changes and plan human capital needs well in advance is key to competitive advantage.In talent acquisition, the objective is to improve quality and consistency of hiring while making the process efficient robust and scalable at the same time Major thrust has been laid on diversification of sourcing channels by enhancing usage of social networking.platforms for recruitment. This helps in reaching out to a wider talent pool and also reduces hiring cost.During the year, anew digital recruitment system has been implemented to make the entire process more efficient transparent and metrics driven. Special emphasis was laid on improving the candidate experience through various initiatives such as chat-bots, video interviews and adopting intuitive and user-friendly processes.The Company is an equal opportunity employer promoting gender diversity and equality at the workplace. During the year under review, the Company celebrated ‘Gender Diversity and Inclusion’ week thereby emphasising its importance among the employees.The Company has a structured training, skill development and higher education program for career enhancement and personal growth for each employee. For this, the Company has tied up with some reputed institutions For workmen on the shop-floor, the Company provides opportunities to pursue a diploma engineering course at these institutions. Those with a diploma engineer qualification can pursue a degree in engineering The Company encourages employees by providing funds, time and better career opportunities to those pursuing a higher qualification.',
                        'During the year, 292 workmen and diploma engineer employees benefited from higher education programs.Number of people upskilled by Training Academy.Source: Company.An in-house educational and training infrastructure Maruti Suzuki Training Academy (MSTA) plays a pivotal role in facilitating the identification of skill gaps and preparing people for future business needs and challenges. The scope of training is extended to all relevant business partners as well. The investments in capacity and capability building of dealers, suppliers and transporter personnel go a long way in enhancing the quality of overall business and customer experience.During the year, a total of 70,914 people benefited from varied training initiatives of the Company.',
                        'Horizontal Implementation of Best HR Practices at Supplier Plants.The Company recognises the importance of sound human resource (HR) and industrial Labour relations (IR) practices to promote a safe and healthy work environment throughout the supply chain. The human.Board’s Report Corporate Governance Report Management Discussion & Analysis Business Responsibility Report.capital development of suppliers can only happen if the top management of the supplier feels the need to invest in the same. The Company conducted over 300 workshops to sensitise the top management on the need for human capital development and also shared the best practices. Subsequently, onsite assessments of their plants are conducted to evaluate and identify areas of improvements..Engineering, Research and Development.Since inception, the Company has internalised a customer-centric approach both in its strategy and organisational culture. Indian customers are very demanding and desire features of higher price segment cars at lower price bracket cars. To succeed in such a market, there is a necessity to design cost engineered products that not only meet customer expectations but also create new product and customer segments that help in garnering larger share of the market and better profitability for the Company. In all this, the role of R&D becomes crucial..In addition to rising customer expectations with regard to vehicle styling, features and technologies, regulations are changing rapidly. Both these factors are leading to significant increase in the intensity of R&D efforts. Moreover, as the Indian market is expected to expand in the near future, more number of products will be required at a faster pace further burdening the R&D. The Company has been able to remain ahead of the customer expectations and regulatory requirements and maintained its market leadership position in the Industry.',
                        'This was made possible with strong support and commitment of R&D centre of Suzuki Motor Corporation Japan which has several decades of experience in designing products and technologies. The Company’s in-house R&D function which is gaining design capability from SMC Japan, is complementing it in development efforts for some of the new models and also gaining capability to design vehicles on its own by obtaining core technologies and new age technologies from SMC. It is important to note that acquiring R&D capability to design a vehicle independently requires significant time and the Company continues to depend on SMC for a larger part of product design and development work..Key highlights of R&D efforts during FY’19 are discussed in detail in Annexure D of the Board’ Report..Cost Optimisation.The year was marked by adverse commodity price and foreign exchange rates impacting the profitability of the Company. In order to reduce the adverse impact of rise in input costs, several cost reduction programs continued throughout the year. These include localisation of direct and indirect imports, value engineering and value analysis, yield improvement and sharing of scale benefit with suppliers..The Company is also working with Indian steel makers for the development of local high tensile and galvanised steel material to improve indigenisation..The Company has foreign exchange exposure on account of import of components. In the last few years, a large portion of this exposure has been reduced with a focused approach by adopting various measures such as:.Project based approach for localisation of high technology parts Launching a new model with maximum possible localisation to realise benefits over alonger timeframe.',
                        'Localisation of critical functional parts with support of SMC and vendors’ overseas collaborators Enhanced _ procurement from the Japanese suppliers’ transplants in ASEAN region to reduce the dependence on Yen..The Company continued its localisation drive during the year as well..Foreign exchange fluctuation affects financial performance. The Company is preparing an ambitious plan to reduce the import content significantly and insulate the financial performance from such fluctuations..During the year, prices of commodities such as flat steel, plastics, aluminum, precious metals, lead and copper firmed up. The Company tried to limit the adverse impact of commodity price increase through better negotiation and hedging..Every year, the contribution of all employees in cost reduction drives and suggestion schemes result in significant cost savings. This participation process is a unique example to achieve organisational excellence. It demonstrates the oneness with which employees collectively work towards achieving organisational goals. During the year, with the help of suggestion scheme ‘Sujhaav Sangrehika’ and cost reduction drive ‘Sanchaika’, the Company was able to achieve cost savings to the tune of ¢ 1,118 million.',
                        'Financial Performance.The Company registered Net Sales of ¥ 830,265 million and Profit after Tax of ¥ 75,006 million, de-growth of 2.9% over the previous year..Treasury Operations.Abridged Profit and Loss Account for 2018-19.Board’s Report Corporate Governance Report Management Discussion & Analysis Business Responsibility Report.(® in million).Foreign Exchange Risk Management.The Company is exposed to the risks associated with fluctuations in foreign exchange rates mainly on import of components, raw materials, royalty payments and export of vehicles. The Company has a well-structured exchange risk management policy. The Company manages its exchange risk by using appropriate hedge instruments depending on market conditions and the view on Currency..Internal Controls and Adequacy.The Company has a proper and adequate system of internal control to ensure that all assets are safeguarded and protected against loss from unauthorised use or disposition, and that all transactions are authorised, recorded and reported correctly. The internal control system is designed to ensure that financial and other records are reliable for preparing financial information and other data, and for maintaining accountability of assets. The internal control system is supplemented by an extensive program of internal audits, reviews by management, and documented policies, guidelines and procedures..Information Technology.To harness the full potential of ever-increasing digital adoption, the Company has embarked on a digital transformation journey. The objective is to rethink every aspect of the organisation and come up with tailor-made solutions to further improve customer experience and process efficiencies. It started with CRM system modernisation that will, once completed, set new benchmarks in customer experience. During the year,.the Company digitised a host of internal processes for employees with an intent to further improve engagement, productivity and satisfaction.',
                        'Institutionalising a framework to capture organisational learning was another important initiative taken during the year that will help the Company build on its experience in a much more efficient manner..The Company is also actively pursuing the use of new age technologies. Deploying platform solutions based on Internet of Things (loT) technology is one such example. The platform is capable of capturing real-time machine parameters and production data. It is helping the Company with increased overall equipment efficiency, improved predictive maintenance, and enhanced safety. By using advanced data analytics, the Company is empowering business leaders with quick and efficient decision making. At the same time predictive and artificial intelligence is making the Company find ways to improve business efficiencies..Logistics.Amidst increasing spread and volume of logistics operations, the Company’s focus is on ensuring fleet movement to be safe, fast, cost-effective and environment friendly. Key initiatives, helping us meet the stated objective are:.Telematics Solutions.Most of the car carriers have been integrated with GPS devices for real time tracking. In 2018-19, the Company saved = 194.6 million on account of route optimisation which also resulted in faster delivery of vehicles. Each GPS device comes with a voice-box to give proactive and timely alerts to the driver before entering accident prone zones. It is also an extremely effective tool to give real-time feedback to the driver in the event of harsh manoeuvring or over-speeding, thereby, helping inculcate safe driving habits among the driver community..Integrated Digital Platform.The Company closely tracks the Key Performance Indicators (KPIs) of all service providers. Some of the standardised KPls include in-transit delay, en-route stoppages other than pit-stops, route deviation, daily running of trucks, loading and unloading time among others.',
                        'On observing any deviation against the set KPIs, an exception ticket is generated and the team gets in touch with the relevant stakeholder to immediately resolve the issues..Regional Stockyard.During the year, the Company started operations at its third regional stockyard in Siliguri that now handles dispatches for North-Eastern states. The two other regional stockyards are in Bengaluru and Nagpur to timely serve the southern and central region respectively..Enhancing Multi-modal Dispatches.Rail continues to be among the fastest and the most economical modes of transport along with benefit of reduced carbon footprint and delivery time. During the year more rakes and destinations were added. The year witnessed a growth of 40.7% in vehicles dispatch using rail model. As a result, CO2 emissions reduced by 1,258 MT. The Company is actively pursuing opportunities to come up with in-plant sidings at Manesar and Gujarat plant to further enhance volumes and operational efficiency of despatches..Risk Management.Over the past several years the Company has made conscious and concerted efforts to counter the threat of cyber security to its business. It has invested in Security Operations Center (SOC) to detect any IT security incident. Sandboxing technology has been put in place to ensure proactive malware detection and containment. As a measure against rapidly emerging cyber threats, the frequency of the Vulnerability Assessment/Penetration Testing (VA/PT) has been increased from once to twice a year. Periodic trainings are conducted for internal IT teams to equip them with knowledge and techniques to identify and respond against any cyber security incident. Regular user awareness programs are also organised to sensitise users on phishing attacks..The Company continued efforts to identify and improve on potential sources of risk that could disrupt business continuity.',
                        'Among the various potential sources of disruptions identified, greater emphasis was given to address the issues pertaining to fire safety..In the recent past, the Company has implemented various preventive measures including some of the best practices of SMC, Japan related to fire safety. The scope of fire safety has been extended to the suppliers’ operations also. The Company has started safety assessment of its suppliers with a special focus on fire safety assessment. Suppliers are re-audited to judge their preparedness and provided guidance on improvements to prevent fire accidents..After disruption at a supplier’s facility due to water logging, the Company has increased its focus on such issues and identified suppliers vulnerable to such problems. Adequate risk mitigation measures are being undertaken..The Company also carries out a comprehensive supplier assessment to identify the weak areas with an objective to improve suppliers’ capability. Risks get identified during assessments and appropriate mitigation measures are then taken with a time bound action plan..In order to tap the growth opportunity going ahead, it is imperative for the Company to groom its employees and create a talent pool. Accordingly, the Company has put a systematic succession planning process in place to create a talent pipeline.',
                        'Outlook.Auto industry will witness several regulations in the year 2019-20. Introduction of Anti-lock-braking system (ABS) and implementation of second phase of safety regulations are among major ones. Though BS-VI regulation is coming into effect from 1st April 2020, it will be applicable on registration of vehicles and not on production. This means, BS-IV spec vehicles cannot be sold from 1% April 2020 and any unsold inventory beyond 1st April would be of no use. A careful volume planning needs to be done in such a scenario..Further, all three major regulations will come into effective simultaneously in the financial year 2019-20. These regulations would lead to increase in prices may affect the demand specially of price sensitive entry level cars..On the economic growth for 2019-20, most of the credible national and international research agencies have revised down their growth forecast. Instead of any sharp recovery, the economy is expected to gather a gradual momentum from the current state. Concerns over a global economic slowdown are growing. On the positive side, however, easing of interest rate, public spending in rural areas and increase in disposable incomes of households due to tax benefits augur well for the economy.',
                        "Disclaimer.Statements in this Management Discussions and Analysis describing the Company's objectives, projections, estimates and expectations are categorised as ‘forward looking statements' within the meaning of applicable laws and regulations. Actual results may differ substantially or materially from those expressed or implied. Important developments that could affect the Company's operations include trends in the domestic auto industry, competition, rise in input costs, exchange rate fluctuations, and significant changes in the political and economic environment in India, environmental standards, tax laws, litigation and labour relations."
    
    ]

    adani_chunk_list= ["Indian Economic Overview.The Indian economy grew at 6.8% during FY19 buoyed by a strong recovery in investment and robust consumption along with improvement in monetary and fiscal policy. Further, the index of industrial production grew firmly, owing to sturdy demand for capital equipment, construction goods, and consumer durables. Initiatives taken by the Government towards reduction of GST rates for some real estate activities also provided a boost to the industry. However, a slight contraction in crude oil and natural gas production offset the strong growth in the coal output, resulting in mining sector growing by a meagre 1.2%..The gross capital formation demonstrated a growth of 10% and was sustained by a strong growth in Central Government capital expenditure, which rose by 20.30% as investment in roads, railways, and urban infrastructure remained strong. In addition, the capital spending by public sector enterprises posted a growth of 05.50%..The series of actions and initiatives have been taken to improve the business climate and to liberalize the Foreign Direct Investment (FDI). Reflecting the cumulative actions, India jumped 30 spots on the World Bank's Ease of Doing Business rankings brightening medium-term growth prospects. This has been validated in the form of a sovereign ratings upgrade, the first in 14 years. The major focus area for policy maker would remain stabilizing the GST, completing the twin balance sheet actions and staving off threats to macro-economic stability.",
                       'Further, revived rural consumption, persistent growth in private investment as a result of improved bank and corporate balance sheets, more competitive domestic firms and less strain from net exports will lead to a significant growth in the Indian economy..Annexure to the Directors’ Report MANAGEMENT DISCUSSION AND ANALYSIS REPORT.Industry Overview.Coal Business.The coal production in India increased by 8% reaching 730 Million Tonnes ("MT") during April to Mar 2019, from 675 MT in corresponding period of the previous year. Further, the country’s coal imports also jumped by ~9% to 233.6 MT in FY19 against 214.6 MT in FY18. The thermal coal imports also grew by 19% reaching 171.9 MT in FY2018, the highest since 2014. This growth in coal imports is largely attributed to constraints on petroleum coke consumption, a cheaper substitute of coal off-set by rising demand from cement and small and medium-scale industries in India..India is one of the world’s largest consumers of coal and the rising import of fuel is resulting in significant increase in trade deficit, urging the Government to invest in developing more domestic resources. In order to boost the growth of the industry, Government also increased the allocation towards exploration of coal and lignite from 7.82 billion in the previous budget to & 8.2 billion in interim budget 2019-20. During the year, the Government allocated 85 new coal mines in the country with a view to increase the production of coal in the country. This is expected to yield result in the form of higher output in the next year. Further, the Government also plan to add 10 more mines in the coming year.',
                        "(Source: Economic Times).Airports.In 2018, India was the fastest growing domestic air travel market witnessing a growth in number of people flying within the country from 117 million in 2017 to 139 million in 2018, reflecting a growth of 18.6%. Further, domestic air passenger.Source: MoSPI, Gol.volume also increased 14.27% to 126.8 million in FY19 over the previous year. The cumulative traffic of domestic carriers stood at 171.3 million for FY19, posting a growth of 14.25%. If the trend continues, India would become one of the top aviation hubs by 2040. The passenger traffic is expected to grow six-fold to around 1.1 billion. India has one of the largest aircraft order books currently with pending deliveries of over 1000 aircraft..This growth is being driven by a growing economy, rising incomes, intense competition among airlines and a supportive policy environment. The National Civil Aviation Policy (NCAP 2016) signaled the Government's intent to radically alter the sector's growth trajectory. NCAP's flagship program - Regional Connectivity Scheme (RCS or UDAN) is taking flying to the masses by offering subsidised fares as low as USD 35 for a one hour flight. Initiatives like Nabh Nirman (for airport capacity augmentation), Digi Yatra (for paperless travel) and AirSewa (for online passenger grievance redressal) etc. are bringing in radical changes. India may have around 190-200 operational airports in 2040. Its top 31 cities may have two airports and the cities of Delhi and Mumbai three each. The incremental land requirement is expected to be around 150,000 acres and the capital investment (not including cost of acquiring land) is expected to be around USD 40-50 billion..and 2018.",
                        'Further, the policy and regulatory environment for Public Private Partnership (PPP) in India also experienced continuous improvement on account of revival of stuck projects, faster dispute resolution, availability of funding (through multilateral agencies and bonds), large scale programmes like Bharatmala, UDAY, UJALA, UDAN etc. and schemes like INViTs and REiTs..In the coming years, roads and railways are expected to hold a larger share of the total investment pie. Construction sector is expected to ramp up to 32 km per day by 2020 on account of NHAI’s sharp focus on award of projects under the Bharatmala programme. For railways, electrification, station development and port connectivity projects are expected to continue offering large opportunities. Metro will also continue to offer EPC opportunities to various construction players, as more than 25 cities in India are anticipated to have metro rail networks in the coming years. (Source: Economic Times, Interim Budget 2019-20)..Average construction of highways during 2018-19.Allocation to village road construction, under the Pradhan Mantri Gram Sadak Yojana for FY 2019-20.Allocation to the highways sector under the Interim Budget for FY 2019-20',
                        'Road and Infrastructure.India witnessed significant growth across roadways, highways, railways and airways infrastructure during the year. Of this, highways sector remained one of the best performing segments with the Government expenditure rising from % 34,345.2 crore in 2014-15 to % 78,625.5 crore in 2018-19 towards development and construction of highways. Moreover, nearly 39% projects were started in FY19 under the highways sector out of over 600 infrastructure related projects which started between 2014.On the highway to growth.Road projects have been on the fast-track to execution from the second half of FY19, but may still fall short of initial Government targets.Water.As per the United Nations World Water Development Report 2019 (WWDR), nearly 2.1 billion people still do not have access to clean and readily available drinking water and that up to 4.3 billion are without access to safe sanitation. It has underlined that fulfilling the human rights to safe drinking water & sanitation for all can also significantly contribute to achievement of the broad set of goals of the 2030 Agenda for Sustainable Development. The report found that by the year 2050, 45% of global gross domestic product and 40% of global grain production will be threatened by environmental damage and lack of water resources..As compared to the global context, scenario in India is not too different. India has 4% of the world’s water resources and a disproportionate 17% of world’s population. Water leads to cultural, social, economic development of India. More than 60% of India’s irrigated agriculture land and 85% of drinking water supplies are dependent on groundwater supply. Agriculture in India is the prime user of freshwater with a share of 80% followed by industry & domestic applications. As per United Nation, any region with annual water availability of 1700 cubic meter per person is water stressed region.',
                        'In India, about 13 states spanning around 300 districts face water stress..Foreseeing the need, Indian Government increased 2019-20 budget allocation to water and sanitation to around = 21,000 Crore including Swachh Bharat Mission (SBM) & National Rural Drinking Water Programme (NRDWP). So far, as many as 30 projects in 17 packages have been identified to be awarded under the HAM scheme with total value of the projects is to the tune of = 101,115 crore..Defence Industry.The defence sector is one of the most important sector in India. It has been doubling the exports since three years and is estimated to have crossed the value of ~ 10,000 crore by March 2019. This growth in export from % 1,500 crore in FY2017 to = 10,000 in FY19 has been possible due to Government reforms which has made it easier for the private players to enter the industry. Further, since the liberalisation of FDI in the sector, there has been nearly = 4000 crore of investment in the sector. Also, since Launch of Make in India initiative in 2014, the defence ministry has undertaken a number of initiatives in order to promote indigenous defence manufacturing. This includes defence procurement process, industrial licensing, foreign direct investment, exports and innovation. Further, in the previous year, 2 new defence corridors were also set up with the aim to increasing defence production to <= 1,70,000 crore by 2025..Allocated towards defence sector in the Interim Budget 2019-20.Share of Defence Services in Interim Defence Budget 2019-20.Solar Panel Manufacturing.India has witnessed a_ substantial growth in power generation in the past few years. Out of electricity produced in Q1 2019, solar power accounted for over 11.4 BUs which reflects a growth of 34% YoY from 8.5 BUs generated in Q1 2018. During the year, about 39.2 BUs of solar power was produced, recording a growth of 52% from production in FY18.',
                        "The sector imported solar modules and cells worth nearly = 184.6 billion during 2018, a decline of 37% from = 269 billion in 2017. This decline is largely attributed to demand slowdown due to withdrawal of tax incentives and imposition of safeguard duty..During the year, the sector faced several headwinds such as imposition of safeguard duty on imported PV modules, falling tariffs and continuous fall of the Indian rupee against the US dollar. The depreciation in the Indian currency is also likely to augment the capital cost of solar power projects by 20- 25%, although its impact has been partly set off by decline in the PV module prices by about 20% since June 2018. Also, implementation of recent policy initiatives such as rise in share of renewable purchase obligation for states and waiving of inter-state transmission charges for solar projects contributes to faster capacity addition in the sector..The sector is anticipated to grow in the coming fiscal owing to capacity addition, favourable policy push and increasing demand for fossil fuel based energy. Further, the sector is also anticipated to witness a heightened technology-led growth. Technological innovation, being at the centre of.solar power development in India, is also anticipated to boost the sector with the help of several important tools such as floating solar, energy storage and flexible modules. (Source: Mercom India Research, Economic Times, The Hindu, Energy World).Gol's target of generating solar power by 2022.Solar capacity addition during 2018-19",
                        "Edible Oil Industry.India Edible Oil Import (mn tons).HIndian Edible Oil Import —— Refined Oil as % of Total Oil Imported.(Source: The Soybean Processors Association of India).Company Overview.About Us.Adani Enterprises Ltd. (the company or AEL) is one of the fastest growing diversified conglomerates providing a range of products and services together with its subsidiaries. The Company is engaged in coal mining & services, coal logistics, solar module manufacturing and edible oil & FMCG food businesses in India and internationally. Besides this, the Company is an incubator focusing on establishing new businesses in infrastructure and energy sector. It has done this consistently since 1994, when it was first established and listed. Post which, Adani Ports, Adani Power, Adani Transmissions and such businesses were demerged from the Company and/or independently listed on the stock exchanges. In the last two years, consistent with the same model, we have demerged Adani Green Energy Limited and Adani Gas Limited from Adani Enterprises which were respectively listed in June 2018 and November 2018..Financial performance.The Company's continued focus on infrastructure and energy verticals is expected to continue to drive its performance and the company remain committed to maintaining high operating standards. The Company has registered improved financial performance on the back of its strong operational performance across key businesses..Key Highlights of the Company’s consolidated performance for the year are as under..Consolidated Income from Operations increased by 12% to ¥ 40,379 Crores in FY 19 vs % 35,924 Crores in FY 18 Consolidated EBIDTA was = 2,541 Crores in FY 19 vs % 2,626 Crores in FY 18 Consolidated PAT for FY 19 was % 717 Crores vs = 757 Crores in FY 18.The Company has demonstrated strong performance across Integrated Coal Management, Mining & Services, Solar Manufacturing and Agro vertical in spite of rising exchange rates and competition.",
                        'Operational Performance.The Company remains committed to play an enhanced role in Nation Building. As an incubator, it focuses on establishing new businesses in infrastructure in the energy sector. The Company has achieved this consistently since 1994 when it was listed. Post which, various businesses were demerged from the Company and/or independently listed on the Indian stock exchanges..During the year under review, the performance of the Company is encouraging. The Company has been leading across all the fronts and maintained better than industry performance. We remain focused on executing our strategy and increasing momentum of our businesses across the key sectors for long term, sustainable growth..Key highlights of the Company\'s consolidated operational performance for the year is as under -.Integrated Coal Management volumes stood at 67.45 Million Metric Tons ("MMT"). Mining and Services business dispatch volumes stood at 12.13 MMT. Solar Module manufacturing volumes was 637 Mega Watt (“MW”).',
                        'Key business segments.Integrated Coal Management (ICM).Adani group (“group”) has evolved as a_ diversified conglomerate based in India having global operations with primary interests in energy sector while the Company continues to operate as the flagship company of the group. Group was primarily involved in coal logistics business and gradually it has backward integrated its operations in domestic and overseas coal mining through the Company along with forward integration in ports, logistics, power generation and transmission through various other group companies..The Company has maintained the status of being the largest Trader and Importer of Thermal Coal in India during the financial year 2018-19 and maintained its market share. However, the business saw a decline in the volume pertaining to the supplies made to various States or Central owned Electricity Boards. This segment continues to struggle amidst increased domestic production and power generation scenario in the country..During the last quarter of FY 19, there were some improvement in the supplies to various States or Central owned Electricity Boards. Most of the Central and State owned Electricity Boards have come up with the tenders for the procurement of imported coal for their respective plants. Furthermore, the Company is expanding efforts in capturing higher market share in steel, cement and other sector by venturing into the retail segment to cater specific local market in different geographies. In addition, the company has started to provide logistic solutions for coastal movement of domestic coal under the ambit of SAGARMALA..The Company with its established business relations with coal suppliers of Indonesia, Australia and South Africa has evolved as India’s largest coal importer for non-coking coal catering to the requirement of both private and PSU clients.',
                        'The Company has consolidated its position in coal trading business during the last decade and has developed strong business relationships with miners in Indonesia, Australia and South Africa for procurement of imported coal. The Company continues to look at opportunities to develop business relations with the new miners, which will lead to timely delivery of coal..The Company has developed business relationship with diversified Customers across various end-user industries. It enjoys major share in domestic PSU tendering business. It imports coal through all the major ports of India, which saves the logistic cost and ensures timely delivery to its customers.',
                        'Mining and Services.Domestic Mine Development and Operations (“MDO").Our coal mining business involves mining, processing, acquisition, exploration and development of mining assets..In India, as a part of the public private partnership model, Government / Public sector companies including State Gencos (State Electricity Boards), which are allotted Coal Blocks, appoint a Mine Developer and Operator (“MDO") to undertake all activities relating to the development and operations of a Coal Block allotted. After Hon\'ble Supreme Court’s 2014 order leading to cancellation of earlier Coal Block allotment, Ministry of Coal passed and notified The Coal Mines (Special Provisions) Act, 2015. As per new Act, coal mines are being auctioned and allotted. Many of the Government / Public sector companies who were allotted coal blocks have published tenders for selection of MDO and are at various stages of bid processes and subsequent award of tender. The Company has participated widely in such tenders to secure long term MDO contracts in the last financial year. In FY 2018-19, AEL has successfully entered into long term MDO contracts of Suliyari Coal Block and Bailadila Deposit 13 Iron Ore Mine allocated to Andhra Pradesh Mineral Development Corporation Ltd. (APMDC) and NCL (NMDC-CMDC Ltd.) respectively through competitive bidding process. Further, many of the other tenders are at advanced stage of getting concluded..Moreover, Ministry of Coal is also in process of opening up commercial coal mining for private sector in phased manner, which could be further opportunity for the Company to leverage its mining capabilities and coal trading experience..The Company has been appointed as MDO and is undertaking activities relating to the development and operations of certain Coal Blocks in India.',
                        'The outlook for the sector remains positive..Parsa East and Kanta Basan Coal Block Rajasthan Rajya Vidyut Utpadan Nigam Limited ("“RRVUNL\'’) has been allocated the Parsa East and Kanta Basan Coal Blocks (PEKB) in Chhattisgarh. RRVUNL has entered into a Coal Mining and Delivery Agreement with Parsa Kente Collieries Limited (PKCL) [a Joint Venture Company of RVUNL and the Company] appointing PKCL as Sole Mining Contractor. PKCL as Mine Developer and Operator of PEKB is undertaking development, mining, beneficiation of coal, arranging transportation and delivery of washed coal to end use power projects of RRVUNL. The project commenced Mining Operations and dispatch of coal to Thermal Power stations of RRVUNL in March 2013. For Financial Year 2018-19, Raw coal Production was 15 MMT, Washed coal Production was 12.0 MMT and Washed coal dispatch to Thermal Power Plants of RRVUNL was 12.1 MMT. Kente Extension Coal Block RRVUNL has been allocated the Kente Extension Coal Block at Chhattisgarh. RRVUNL has entered into a Coal Mining and Delivery Agreement with Rajasthan Collieries Limited (RCL) [a Joint Venture Company of RVUNL and the Company] appointing RCL as Sole Mining Contractor. RCL as Mine Developer & Operator of Kente Extension Coal Block will be undertaking development of the Coal Block, mining, beneficiation of coal and arranging for transportation and delivery of coal to end use power projects of RRVUNL. The Coal Block is under development stage. Parsa Coal Block RRVUNL has been allocated the Parsa Coal Block at Chhattisgarh. RRVUNL has entered into a Coal Mining and Delivery Agreement with Rajasthan Collieries Limited (RCL) [a Joint Venture Company of RVUNL and Adani Enterprises Limited] appointing RCL as Sole Mining Contractor. RCL as Mine Developer & Operator of Parsa coal block will be undertaking development of the Coal Block, mining, beneficiation of coal and arranging for transportation and delivery of coal to end use power projects of RRVUNL.',
                        'The Coal Block is under development stage. Gare Pelma Sector-lll Coal Block Chhattisgarh State Power Generation Company Ltd. (CSPGCL) has been allocated the Gare Pelma Sector- Ill Coal Block at Chhattisgarh for captive use in their.appointed (GPIIICL), a wholly owned subsidiary of the Company, as Mine Developer and Operator (MDO) for Development, Operation, Mining and delivery of coal to end use power project of CSPGCL. CSPGCL has entered into a Coal Mine Services Agreement with GPIIICL on 16° November, 2017. GPIIICL as Mine Development & Operator of Gare Pelma Sector Ill Coal Block will be undertaking development of the Coal Block, mining and arranging for transportation and delivery of coal to end use power projects of CSPGCL. The Mine Opening Permission of the Coal Block was obtained on 26" March, 2019 and overburden removal commenced on 28" March, 2019. Talabira I! & Ill Coal Block NLC India Limited (NLCIL) has been allocated the Talabira Il & Ill Coal Block at Odisha for captive use in their Thermal Power Plant. NLCIL has appointed Talabira (Odisha) Mining Private Limited (TOMPL), a subsidiary of the Company, as Mine Developer and Operator (MDO) for Development, Operation, Mining and delivery of coal to NLCIL. NLCIL has entered into a Coal Mining Agreement with TOMPL on 23% March, 2018. TOMPL as Mine Development & Operator of Talabira II & Ill Coal Block will be undertaking development of the Coal Block, mining, loading, transportation and delivery of coal to delivery points. The Mine Opening Permission of the Coal Block was obtained on 29" March, 2019 and overburden removal commenced on 31* March, 2019. Suliyari Coal Block Andhra Pradesh Mineral Development Corporation Limited (APMDC) has been allocated the Suliyari Coal Block at Madhya Pradesh for commercial mining of coal. APMDC has appointed the Company as Mine Developer and Operator (MDO) for Development, Operation, Mining and delivery of coal to APMDC.',
                        'APMDC has entered into a Coal Mining Agreement with the Company on 8" March, 2018. The Company as a Mine Development & Operator of Suliyari Coal Block will be undertaking development of the Coal Block, mining, loading, transportation and delivery of coal to delivery points. The Coal Block is under development stage. Bailadila Deposit - 13 Iron Ore Mine NCL (NMDC-CMDC Limited) is the Mining Lease holder of Bailadila Deposit-13 Iron Ore Mine in the State of Chhattisgarh. NCL has appointed the Company, as.Operation, Mining and delivery of iron ore to NCL. NCL has entered into an Iron Ore Mining Services Agreement with the Company on 6 December, 2018. The Company as Mine Development & Operator of Bailadila Deposit-13 Iron Ore Mine will be undertaking development of the Iron Ore Block, mining, loading, transportation and delivery of iron ore to delivery point. The Iron Ore mine is under development stage. Gare Palma Sector | Coal Block Gujarat State Electricity Corporation Limited (GSECL) has been allocated the Gare Pelma Sector - | Coal Block at Chhattisgarh for captive use in their Thermal Power Plants in the State of Gujarat. GSECL has issued conditional Letter of Acceptance (LoA) to Consortium of the Company and Sainik Mining and Allied Services Limited having 74% and 26% stake respectively on 15% December, 2018 for Development, Operation, Mining and delivery of coal to end use power projects of GSECL. Coal Mine Services Agreement between the AEL-SMASL Consortium and GSECL is yet to be signed. Gare Palma Sector II Coal Block Maharashtra State Power Generation Co. Ltd, (MAHAGENCO) has been allocated the Gare Pelma Sector-ll Coal Block at Chhattisgarh for captive use in their Thermal Power Plants in the State of Maharashtra. MAHAGENCO has issued conditional Letter of Acceptance (LoA) to the Company on 25" February, 2019 for Development, Operation, Mining and delivery of coal to end use power projects of MAHAGENCO.',
                        'Coal Mine Services Agreement between the Company and MAHAGENCO is yet to be signed. Gidhmuri Paturia Coal Block Chhattisgarh State Power Generation Company Ltd. (CSPGCL) has been allocated the Gidhmuri Paturia Coal Block at Chhattisgarh for captive use in their Thermal Power Plants in the State of Chattisgarh. CSPGCL has appointed the Company as Mine Developer and Operator (MDO) for Development, Operation, Mining and delivery of coal to CSPGCL. CSPGCL has entered into a Coal Mining Agreement with AEL-SMASL Consortium on 2% May, 2019. AEL-SMASL Consortium as Mine Development & Operator (MDO) of Gidhmuri Paturia Coal Block will be undertaking development of the Coal Block, mining and arranging for transportation and delivery of coal. The Coal Block is under development stage.',
                        'Coal Mining in Indonesia.PT Adani Global, Indonesia a wholly-owned step down subsidiary of the Company, has been awarded coal mining concessions in PT Lamindo Inter Multikon and PT Mitra Niaga Mulia (step down subsidiaries) in Bunyu island, Indonesia from which coal is used for the captive consumption in power projects..The Bunyu Mines has Joint Ore Reserves Committee (JORC) compliant resource of 269 Million Metric Tonnes (MMT) for both the mines (i.e. combined). Production from both the mines (combined) during the year 2018-19 has been at 4.9 Million Metric Tonnes (MMT)..Coal Mining and related Infrastructure in Australia.Our wholly owned step down subsidiaries in Australia have 100% interest in the Carmichael coal mine in the Galilee Basin in Queensland, Australia..During the year ended 31% March, 2019, the Group has been working on the negotiation of key contracts and redesign of project delivery strategies. Further, apart from working through the approval of management plans and other similar approvals, the Group has been responding to legal challenges brought with respect to decisions already made by relevant authorities..Road, Metro and Rail.To contribute towards Nation Building and infrastructure development, the Company intends to tap the opportunities in the road, metro & rail sector by developing national highways, expressways, tunnels, metro-rail, railways, etc. Adani group has a successful track record of nurturing businesses in the transport infrastructure and is confident of positioning itself as dominant player in the road, metro and rail sector. The group has developed several railway lines in India and abroad. Adani owns the longest private railway lines of about 300 km in India. These private rail lines are connected to our ports, mines and other business hubs to ensure seamless cargo movement..e The Company will focus on projects across pan-india initiated by National Highways Authority of India (NHAI) under Bharatmala Pariyojana, etc.',
                        'and Ministry of Road Transport and Highways (MORTH), Ministry of Railways, Metro Corporations of the various States and any other projects under the purview of the Central or State Authorities and Agencies. As a developer, the Company will primarily target PPP projects structured in Build-Operate-Transfer (BOT), Toll-Operate-Transfer (TOT) & Hybrid-Annuity Mode (HAM) models..The Company will also focus on select EPC projects which can offer scale and complexity in terms of the nature of work and technology requirement and which requires the developer to leverage its project execution Capabilities to create a differentiated value in the industry. Having multiple infrastructure businesses established across different states in India, we would like to leverage our local presence and expertise in project management to build synergies for our Road, Metro & Rail Infrastructure development. In addition, the Company would be focusing on in-organic growth through Mergers and Acquisition, where we will look out for good assets which offer clear visibility of cash flows and are available at attractive valuations. The Company and its wholly owned subsidiary, Adani Transport Ltd. have already bagged three Hybrid Annuity Road Projects from NHAI which are under various stages of development/execution..Water.Water, a basic necessity, is a scarce resource. India which accounts for 17% of world population has access to only 4% of fresh water resources. Lack of holistic policy for water resource management has resulted in increasingly acute crisis being faced in various parts of the country..Realizing the above, Government of India has taken a path breaking step forward by amalgamating various Government department & ministries into a centralized Jal Shakti Ministry..Defence.In continuation of its vision of nation building, the Company ventured into Defence & Aerospace in 2017 with its commitment towards the national security agenda.',
                        "The intent has been to play an instrumental role in helping.transform India into a destination for world class defence and aerospace manufacturing, aligned to the Make in India initiative; thus helping India become self-reliant in its defence and security needs..Modernization of capital equipment has been a key agenda of the Indian Armed Forces with the total planned spend over the next 10-15 years estimated at around US$255bn across the three services. With the Make in India initiative, the Government has placed a key emphasis on reducing the dependence on imports and build indigenous capabilities to facilitate job creation and high end skills development. In addition, the supply which was predominantly met by the Defence Public Sector Undertakings (DPSU's) has been insufficient to meet the modernization needs and hence, there has been a call to the private sector to support the design, development and manufacturing of defence equipment and develop a robust ecosystem where the public and private sector can co-exist..Adani Group has a strong track record of delivering mega projects and has been working effectively with the global partners. In each of the businesses across ports, power generation, power transmission, airports, it was a culmination of the national agenda, long term opportunity untapped by the private sector and the inherent inefficiencies of the public sector enterprises; a story playing in defence and aerospace in India as well. the company is well-positioned to build the long term capabilities especially at the platform level, working along with global partners in delivering defence equipment for the Indian Armed Forces. the Company has established a robust ecosystem of capabilities which will help deliver projects of critical importance for the Indian Armed Forces as well as in the civil sector.",
                        'Some of the highlights of the business are mentioned below :.The Company shall focus on building design and development Capabilities for large and complex platforms like fighter aircraft, helicopters, unmanned aerial vehicles etc. The focus shall stay on playing the role of a System Integrator and nurture the MSMEs ecosystem in the country through strategic investments and acquisitions for tier 1 and tier 2 capabilities. The Company is a compelling candidate for being declared a strategic partner in the Naval Utility Helicopters project for the Indian Navy under the Strategic Partnership Model. The Company has developed critical Tier | and Tier Il capabilities through its investments in MSMEs like Alpha Design Technologies Limited, Comprotech and Autotech across avionics, structures, system design and precision machining..The Company inaugurated a state-of-the-art Aerospace Park in Telangana, Hyderabad in December 2018. The 20 acre sprawling park which houses the Adani-Elbit UAV facility shall be a plug and play facility for MSMEs to set up Tier -I and Tier — II facilities catering to Indian as well as Global OEMs. The Company inaugurated a 50,000 sq. ft. state-of-the- art UAV manufacturing facility in Telangana, Hyderabad in partnership with Elbit Systems of Israel in December 2018. This is India’s first VAV manufacturing facility and the first outside Israel to manufacture the Hermes 900 Medium Altitude Long Endurance UAV. The factory has started operations with the manufacturing of complete carbon composite aero-structures for Hermes 900, followed by Hermes 450, catering to the global markets and will be further ramped up for the assembly and integration of complete UAVs.',
                        "Solar Manufacturing.The Company has set up a vertically integrated Solar Photovoltaic Manufacturing facility of 1.2 GW Capacity along with Research and Development (R&D) facilities within an Electronic Manufacturing Cluster (EMC) facility in Mundra Special Economic Zone (SEZ). The state-of- the art large-scale integrated manufacturing plant to produce Silicon Ingots/wafers, Silicon Solar Cells, Modules and support manufacturing facilities that includes EVA, Back-sheet, Glass, Junction box and Solar cell and string interconnect ribbon..At 1.2 GW of production, this plant is the largest vertically integrated producer of Solar Cells and Modules in India and well supported by manufacturing units of critical components designed to achieve maximum efficiency in the Indian market. This Solar PV manufacturing facility within EMC facility is the first to be located in an SEZ under the M-SIPs scheme under which the investment by MSPVL has been approved..The state-of-the-art manufacturing facility with multilevel infrastructure is optimized for scaling up to 3 GW of modules and cells under a single roof. The unit is located in one of the world's largest Special Economic Zone at Mundra, Gujarat and hence plays host to the entire solar manufacturing ecosystem from Polysilicon to modules, including ancillaries and supporting utilities. MSPVL is facilitating the thrust of Government of India’s “Make in India” concept through its various measures of 12 GW CPSU scheme, KUSUM scheme etc. to achieve its target of 100 GW by 2022..The cutting-edge technology, with machines and equipment sourced from the best in class producers, aim to help in cost leadership, scale of operations and reliability standards as per global benchmarks..Total experience of the team engaged in solar manufacturing business.Working on product development and research in the segment.EBITDA as on 31% March, 2019.Size of order book as on 31% March, 2019.Agro",
                        "Adani Wilmar Limited.The Company entered the edible oil refining business through a 50:50 joint venture company, Adani Wilmar Limited (AWL) with Singapore's Wilmar group. In edible oil and agro commodities business, the Company has continued to maintain its leadership position with its “Fortune” brand and contributes to lead the refined edible oil market..AWL takes pride in being one of India’s fastest growing food FMCG companies. With a 19.2% market share and growth of 7% in Refined Oil Consumer Pack (ROCP) category (Source: Nielsen Retail Monthly Index March 2018 report), “Fortune” continued to be the undisputed leader among edible oil brands in India with largest variety of oils under a single brand name..To strengthen its foothold in the food business, AWL is leaving no stone unturned in coming up with new products giving boost to its already flourishing product basket. The company is determined to reach more & more households in the country with its quality products. ‘Fortune Chakki Fresh Atta’, which was launched last year in NCR and Uttar.Pradesh is now getting launched at multiple locations throughout the country. Adding to its pulses and besan basket, AWL has also launched ‘Fortune Arhar Dal’ and ‘Fortune Khaman Dhokla Besan’ in the selected regions and has started receiving good response. AWL has spent heavily during the year on advertising and promotion for Biryani Classic Basmati Rice by coming up with a new commercial featuring prominent actors- Akshay Kumar & Twinkle Khanna..As a socially responsible organisation, AWL pays attention in safeguarding of environment and has taken a step forward by launching India’s 1% recyclable packaging of edible oil pouches. It has also started the process to collect plastic waste of its consumer products in the state of Maharashtra and is rapidly moving towards other states of the country.",
                        "AWL, in association with Adani Foundation, the CSR arm of Adani group had launched SuPoshan program, a step towards eradicating malnutrition and anemia from the country. This project received prestigious CSR Award during the 53° SKOCH Summit, Dainik Jagran CSR Awards for contribution in health category and was also bestowed with Silver Rank by ACEF Asian Leadership Awards under Best Public Health Initiative category..“Fortune” being India’s No. 1 brand has once again been awarded the prestigious Superbrands award 2018 and has also been voted as the winner of “Reader's Digest Trusted Brand Award” in Gold category. Further, AWL has also been recognized as “India’s Best Companies to work for” by the Great Place to Work Institute, India.",
                        'Adani Agri Fresh Limited.Adani Agri Fresh Limited (AAFL), a wholly owned subsidiary of the company has pioneered the establishment of integrated storage, handling and transportation infrastructure for Apple in Himachal Pradesh. It has set up modern Controlled Atmosphere storage facilities at three locations, Rewali, Sainj, and Rohru in Shimla District. The Company has also set up a marketing network in major towns across India to cater to the needs of wholesale, retail and organized retail chain stores. The Company which is marketing Indian fruits under the brand name FARM-PIK, has expanded its footprint in the branded fruit segment. The Company also imports Apple, Pear, Kiwi, Orange, Grapes etc. from various countries for sale in India..The production of apple during the financial year 2018-19 was impacted due to huge hail storm across the growing belts in Himachal Pradesh. Hence, the availability of good quality apple for CA storage was limited. On the other hand, apple production in Washington State and European.The Company is subject to risks arising from interest rate fluctuations. The Company maintains its accounts and reports its financial results in rupees. As such, the Company is exposed to risks relating to exchange rate fluctuations. The Corporate Risk Management Cell works with the businesses to establish and monitor the specific profiles including strategic, financial and operational risks..We believe that our multi-location operations also allow us to leverage the competitive advantages of each location to enhance our competitiveness and reduce geographic and political risks in our businesses..countries was good. The Government of India had banned the import of apple from China in 2017.',
                        'Due to all the above factors, there was heavy competition from the trade to purchase apple for CA storage and hence the price of apple was high during the procurement period..During F.Y 2018-19, the Company bought 15776 MT of Indian apple valued ~ 82 Crores and Imported 3653 MT of various fruits, valued at = 40 Crores. The Company sold 17,798 MT of domestic apples and 3653 MT of imported fruits Total valued at = 207 Crores.',
                        'Details of Significant Changes in the Key Financial Ratios & Return on Net Worth.Pursuant to amendment made in Schedule V to the SEBI Listing Regulations, details of significant changes (i.e. change of 25% or more as compared to the immediately previous financial year) in Key Financial Ratios and any changes in Return on Net Worth of the Company (on standalone basis) including explanations therefore are given below:.Risk Mitigation.The Company is exposed to business risks which may be internal as well as external. The Company has a comprehensive risk management system in place, which is tailored to the specific requirements of its diversified businesses, is deployed, taking into account various factors, such as the size and nature of the inherent risks and the regulatory environment of the individual business segment or operating company. The risk management system enables it to recognize and analyze risks early and to take the appropriate action. The senior management of the Company regularly reviews the risk management processes of the Company for effective risk management..Services Transformation.The objective of the Services Transformation program is to strengthen the delivery capabilities and governance effectiveness across all corporate services, to enable Services to support Group and Business growth and sustainability agenda. This underlines the need for Services to continually evolve and transform themselves, to be able to deliver on ever growing expectations..The program includes capacity building for services, greater empowerment and accountability at Sites and with expertise leverage across group as guiding principles. Key services have focused on operating models, keeping the service peculiarities, industry practices and delivery expectations, besides the overarching principles..As a follow-up, services have strengthened organization structure, KRAs of key roles, operational processes and delegation of authority.',
                        'Besides, the governance framework for Services has been also strengthened to sharpen focus on agreed priorities and monitoring progress..Integral part of the service transformation program is competency development in each of the service. Accordingly Services are in the process of refining the competency frameworks and designing competency development programs based on baselines created through assessments. Group is engaged with Academic institutes of repute to design and deliver programs to employees working at different competency levels..Towards leveraging the power of networked organization, several collaboration platforms have been created including Service Function Councils and All-Service Councils. These councils would provide platforms for deliberating common evolution agenda, debate specific solutions, and explore options of expertise and resource sharing across boundaries..Service Transformation Program is a multi-year mission, wherein the foundational elements for next stage of evolution have been put in place, while the design and roll- out various across services and shall be tracked to effective execution in coming years.',
                        'Internal Control.The Company has put in place strong internal control systems and best in class processes commensurate with its size and scale of operations..There is a well-established multidisciplinary Management Audit & Assurance Services (MA&AS) that consists of professionally qualified accountants, engineers and SAP experienced executives who carry out extensive audit throughout the year, across all functional areas and submit reports to Management and Audit Committee about the compliance with internal controls and efficiency and effectiveness of operations and key processes risks..Some Key Features of the Company’s internal controls system are:.Adequate documentation of Policies & Guidelines. Preparation & monitoring of Annual Budgets through monthly review for all operating & service functions. MA®&AS department prepares Risk Based Internal Audit scope with the frequency of audit being decided by risk ratings of areas / functions. Risk based scope is discussed amongst MA&AS team, functional heads / process owners / CEO & CFO. The audit plan is formally reviewed and approved by Audit Committee of the Board. The entire internal audit processes are web enabled and managed on-line by Audit Management System. The Company has a strong compliance management system which runs on an online monitoring system. The Company has a well-defined delegation of power with authority limits for approving revenue & capex expenditure which is reviewed and suitably amended on an annual basis The Company uses ERP system (SAP) to record data for accounting, consolidation and management information purposes and connects to different locations for efficient exchange of information. Apart from having all policies, procedures and internal audit mechanism in place, Company periodically engages outside experts to carry out an independent review of the effectiveness of various business processes.',
                        "Internal Audit is carried out in accordance with auditing standards to review design effectiveness of internal control system & procedures to manage risks, operation of monitoring control, compliance with relevant policies & procedure and recommend improvement in processes and procedure..The Audit Committee of the Board of Directors regularly reviews execution of Audit Plan, the adequacy & effectiveness of internal audit systems, and monitors implementation of internal audit recommendations including those relating to strengthening of company's risk management policies & systems.",
                        'Human Resource Strategy.As an organisation, the Company strongly believes that Human Resources are the principal drivers of change. They push the levers that take futuristic businesses to the next level of excellence and achievement. The Company focuses on providing individual development and growth in a professional work culture that enables innovation, ensures high performance and remains empowering. Our lot of focus has been given to HR Transformation activities to revamp the HR organisation structure and processes. The new human resource management systems and processes are designed to enhance organisational effectiveness and employee alignment. The result is that the Company is able to work towards creating leadership in all the businesses that it operates. During the year, several initiatives, such as performance management systems, Learning & Development system, and Talent Management system were put in place to efficient & effective organisation. A lot of focus is being given to enhance people capability through e-learning management system. The broad categories of learning & development include Behavioural, Functional / Domain and Business related..Many other programs for employee rejuvenation and creating stronger inter-personnel relations, team building as well as aimed at further strengthening the bonding across all divisions and locations of the company were organized in the year. These programs help employees significantly in leading a balanced work life in the organization. The HR function is committed to improve all its processes based on the results and feedback and ensure that its manpower will remain its greatest asset.',
                        "Cautionary Notice.Statements in the Management Discussion and Analysis describing the Company's objectives, projections, estimates, expectations and others may constitute “forward-looking statements” within the meaning of applicable securities laws and regulations. Actual results may differ from those expressed or implied. Several factors that could significantly impact the Company’s operations include economic conditions affecting demand, supply and price conditions in the domestic and overseas markets, changes in the Government regulations, tax laws and other statutes, climatic conditions and such incidental factors over which the Company does not have any direct control..The Company undertakes no obligation to publicly update or revise any forward-looking statements, whether as a result of new information, future events, or otherwise."
    ]

    titan_mda_relevant_dict={
        'Performance during the year 2020-21': ['The financial year 2020-21 was the most challenging year for the corporate world in living memory. In one sense, it was even more challenging for the Company, given the kind of products it makes, which are by and large discretionary in nature. However, some other factors kicked in and helped deliver an exceptional sales and financial performance for the year.',
                                                "The store-of-value aspect of the jewellery category as well as its share of wallet gained from within the wedding spends and other categories like travel The substantial innovation capability of Titan that accelerated significantly and drove up desire for all products even in these times The PhyGital capability of Titan that combined deep and extensive understanding of customers, their needs and preferences and used the relationships with the Company's 15,000+ sales staff to connect one-on-one The obsessive focus on ‘total-safety-in-the-store’ that eliminated customer anxiety about shopping A slew of digital initiatives that helped the Company leapfrog The impressive war-on-waste programme and the asset management effort that helped exceed profit and cash targets handsomely The deep commitment of all Titanians and all retail, distribution and vendor partners",
                                                'As a result of all these, all the businesses did better than plan and also laid the foundations for a very good Financial Year 2021-22. Unfortunately, the second wave of COVID-19 struck in April 2021 and created a temporary setback to the operations of the Company.'],
        # 'WATCHES & WEARABLES DIVISION Watches': ['The Management is convinced that the opportunity for “Watch as an Accessory” is timeless and is committed to capitalize on it. While the investment in design and new product development has happened consistently over the last 3-4 years, considerable focus is now being given to re- imagining the World of Titan channel, transformation of the Multi-Brand watch outlet, marketing communication investments and ratcheting up the omni-channel play.',
        #                                          'The very good work done through the War-on-Waste programme is also being continued.'],
        # 'Wearables': ['The Company significantly increased its capability for this domain through the acquisition of HUG Innovations in Financial Year 2019-20, while simultaneously improving its product and app design capabilities as well. The result is a steady pipeline of exciting new products, services and ecosystems starting as early as in 2021. Reflex 3, the Fastrack wrist band launched in February 2021 on the Company’s own proprietary platform, is a sign of things to come.',
        #               'The Management is confident that Titan will establish itself as a prominent player in the Wearables market within the next 18 months through its product/service/ecosystem efforts as well as the distribution expansion into mobile phone outlets and considerable marketing investments.',
        #               "The recovery of the Watches category in Titan’s international market has been a bit slow. But, the newly created business division (now 2 years old) is focusing on the intrinsic opportunity for the Company's brands and is currently building a customer-up strategy for growing in sales, profits and prestige in the next 5 years and a good addition to the domestic Watches and Wearables business."],
        # 'JEWELLERY DIVISION': ['The opportunity for the Jewellery Division during Financial Year 2022-23 as well as in the medium term is excellent. Apart from the low market share and the increasing competitive advantage and brand preference, the Division is continuing to push many levers for growth. Multi-pronged efforts within the wedding market (including a new Engagement Rings focus), a “Many Indias” programme to increase state-level relevance, keeping the momentum behind the Gold Exchange and Golden Harvest opportunities, keeping the “Middle India” network expansion effort going.',
        #                        'The Zoya and Mia brands will also be riding firm on the momentum generated during Financial Year 2021-22 towards their exciting future.',
        #                        'The Division has also created a well-oiled process to keep the total inventory and capital employed in check, which has become embedded and sustainable.',
        #                        'CaratLane has had a scorching growth in Financial Year 202 1- 22 and reported a profit for the first time. The powerful omni- channel approach, very high technology capability, innovative product lines and the new-age employee culture have combined exceptionally well. The Company is very confident that CaratLane has all the ingredients for creating history and substantial stakeholder value in the next few years.',
        #                        'Responsible Sourcing - Progressing definitively towards a sustainable future:',
        #                        'a) A robust rollout of a formal ‘Responsible Sourcing’ program to all vendors, to upgrade their units to the “Standard” level (Cottage, Basic, Standard and World Class, being the 4 increasing levels of evolution) across People, Process, Place and Planet parameters. b) Very good progress achieved on the integrated 3-year program for diamond sourcing, ensuring pipeline integrity by eliminating any mixing up of synthetics, and sourcing stones from sight holders / international mines having the right labour practices and conditions in their supply chains. c) 100% of fresh gold procured from banks is London Bullion Market Association (LBMA) certified ensuring highest purity, quality and mined from ethical sources. The rest of the gold is recycled exchanged gold from customers.',
        #                        'A revised and formal Titan Supplier Engagement Protocol (TSEP ver 2.0) was also rolled out to all suppliers on these counts.'],
        # 'International Business': ['The much-awaited launch of Tanishq in Dubai has been very successful, with customers giving a big thumbs-up for the excellent products and collections, the exquisite store and the superlative customer experience. The expansion into more stores in the UAE and the GCC countries is on the cards. The NRI/PIO jewellery opportunity is very large and the Company is committed to making this a meaningful part of its portfolio in the next 5 years.'],
        # 'EYEWEAR DIVISION': ['The Eyewear Division transformed itself during Financial Year 2021-22 by unlocking substantial value from its operations: discount reduction, channel-mix improvement, product-mix improvement, in-house production enhancement, store and lab closures, other fixed cost management, all of these have contributed to a new platform of profitability that is quite sustainable. The Division is now all poised to grow in sales in the medium term and leverage this foundation.',
        #                      "The Division has also worked substantially on product innovations in lenses (Antifog, Clearsight, Neo progressives) and frames (Indifit, Signature, Glam, etc.) as well as channels (Ecolight low cost stores), all of which significantly improved the Division's competitive advantage.",
        #                      'The overall opportunity in this market is vast on account of three things. The low market share that the Division has, the millions of people in the country with unaddressed refractive error and the new lifestyle (excessive screen time) that is accelerating the need for vision correction among the youth.'],
        # 'FRAGRANCE DIVISION': ["The Division has succeeded in creating a wide range of “Exceptional Quality, Affordable Price” Eau de Parfum variants: ‘Tales’ at ¥ 1595 to ‘Amalfi Bleu’ at 2495 and many in between, and through this created a broad aspirational category for millions of aspiring Indians who find international choices out of their reach. This platform will come in very handy in the next few years as India’s per capita GDP jumps significantly making this business start contributing meaningfully to the Company's revenues and profits."],
 
        # 'Ethnic Wear': ["The market for sarees is very large. While lifestyle changes are influencing the frequency of saree wearing, the opportunity for special occasion sarees, where the Company has chosen to play, is significant and growing. The Division has succeeded in establishing a very strong customer value proposition, a systematic product and store assortment creation process, a healthy gross contribution profile and a unique and special store experience. The next couple of years will see the Division scale up its efforts in network expansion and franchising as well as building a good sourcing and supply chain foundation. The huge advantage that this Division has is the ready member base of the Company's Encircle loyalty program and the franchise network."],
        # 'DESIGN EXCELLENCE CENTRE': ["Without doubt, Design is at the centre of the Company's competitive advantage. The manufacturing backbone then takes over and delivers finely engineered products with exceptional detailing with very high performance parameters and durability. This overall approach has helped the Company become a category expert in every business it is into.",
        #                              'The Design strategy at Titan systematically and seamlessly blends consumer insights, deep design stories, the right aesthetics and fine ergonomics to convert the best of materials into lifestyle products of sophisticated style and exquisite beauty.'],
        # 'DIGITAL': ['The Company started investing in digital about five years back, inCRM, Analytics, e-commerce and m-commerce technologies. Through Financial Year 2020-21, a transformation took place resulting in a wholesale embracing of all these technologies by all employees and the willingness of customers to try new ways of engagement with the Company. The combined effect of these was an explosive growth in sales through new ways: video calling, try-at-home, appointment shopping, omni channel, chatbots, endless aisle, etc.',
        #             'Apart from these, multiple Company processes have seen the effect of various Digital interventions that have ended up reducing costs, improving speed, and increasing accuracy: Artificial Intelligence in Design, Industrial Internet of Things in gold plating, Robotic Process Automation in invoice processing, Persuasive Technology in sales force management and many more.'],
        # 'HUMAN RESOURCES (HR)': ['The Company had 7,235 employees on rolls of which 1,917 were women as on 31% March 2021. Of the total head-count, 3,036 employees were engaged in manufacturing, 3,213 in retail and 986 in corporate and support functions. Of the total base, 150 employees are differently abled.',
        #                          "Employee Safety: The Company's approach to dealing with the unprecedented change the pandemic brought was steeped in the core value of Unconditional Positive Regard for People. The first and foremost step to deal with the pandemic also was to ensure that all employees in every part of the organization were safe and healthy. People over business is the approach that was adopted, be it in regards to resumption of services, opening of stores or people coming back to work. Utmost care and precaution was taken when resuming services. Leading by example is the other principle followed. From working from home when it was prescribed to following the social distancing norms once offices resumed, leaders walked the talk.",
        #                          'Employee Development: The last year has accelerated the adoption of technology and automation by leaps and bounds. In adopting digital media to attract and converting new consumers as well as connect with loyal customers, the Company has made a big shift. On the employees’ front, this has translated to developing new capabilities on digital quotient such as online marketing, phygital selling, virtual customer connect. For non-retail and manufacturing employees, plethora of programs were introduced on digital platforms enabling self-paced and bite-sized learning.'],
        # 'Employee Connect & Employee Wellness': ['A big area of impact for HR in the year has been the positive shift created in employee experience. Holistic approach to Wellness: Will It, Well It, a wellness program, was introduced in Financial Year 2020-21; which saw organization-wide participation in the offerings of the program. The programs offered covered all aspects of wellness: physical, mental and financial.'],
        # 'RISKS AND CONCERNS': ['The Company has a robust process for managing the top risks, overseen by the Risk Management Committee (RMC) of the Board. As part of this process, the Company has identified the risks with the highest impact and then assigned them a likely probability of occurrence. Mitigation plans for each risk have also been put in place and are reviewed by the Management every six months before presenting to the RMC.',
        #                        'These risk types span Technological, Geo-political and Regulatory. To illustrate:',
        #                        'Technological: Disruption in operations due to cyber-attacks or hardware/software failure; Impact of Wearables technology on the Watch category',
        #                        'Geo-political: Dependence of the Watches & Wearables division on China as a source',
        #                        'Regulatory: Compliance needs and challenges in the Jewellery industry'],
        # 'INTERNAL CONTROL SYSTEMS AND THEIR ADEQUACY': ['The Company during the year has reviewed its Internal Financial Control (IFC) systems and has continually contributed to establishment of a more robust and effective IFC framework, prescribed under the ambit of Section 134(5) of Companies Act, 2013. The preparation and presentation of the financial statements is pursuant to the control criteria'],
        # 'SEGMENT WISE PERFORMANCE': ['defined considering the essential components of Internal Control — as stated in the “Guidance Note on Audit of Internal Financial Controls over Financial Reporting” issued by the Institute of Chartered Accountants of India (ICAI).',
        #                              "The control criteria ensures the orderly and efficient conduct of the Company's business, including adherence to its policies, the safeguarding of its assets, the prevention and detection of frauds and errors, the accuracy and completeness of the accounting records and the timely preparation of reliable financial information.",
        #                              'Based on the assessment carried out by the Management and the evaluation of the results of the assessment, the Board of Directors are of the opinion that the Company has an adequate Internal Financial Controls system that is operating effectively as at 31% March 2021.',
        #                              'The Company has a robust internal audit function consisting of professionally qualified chartered accountants who cover the business operations as well as support functions and provide quarterly reports to the Audit Committee.'],
        # 'SIGNIFICANT CHANGES IN KEY FINANCIAL RATIOS': ['During the year, following are the key financial ratios of the Company where there was a change of 25% or more as compared to the immediate previous financial year'],
        # 'CHANGE IN RETURN ON NET WORTH': ['The details of change in Return on Net Worth of the Company as compared to the previous year is given below:',
        #                                   '* Note: With the declaration of the COVID-19 as a pandemic in mid-March 2020, the performance of various Divisions were affected due to store closures consequent upon declaration of national lockdown by the Government. This has resulted in the profit before tax being lower by 41%, which in turn impacted the respective ratios having a variance of more than 25%'],
        # 'DISCLOSURE OF ACCOUNTING TREATMENT': ['The financial statements of the Company have been prepared in accordance with the Indian Accounting Standards (Ind-AS) notified under the Companies (Indian Accounting Standards) Rules, 2015 and Companies (Indian Accounting Standards) (Amendment) Rules, 2016 read with Section 133 of the Companies Act, 2013.'],
        # 'OUTLOOK FOR FY 2021-22': ['At an overall level, the circumstances of FY 2021-22 are somewhat better than FY 2020-21.',
        #                            "e The economic circumstances of the Company's customers (the Upper Middle Class) are expected to be as good as in FY 2021-22 or perhaps even better, given that virtually all companies are giving out raises to their employees, with its cascading effects on other segments of the population The rural economy is also likely to be good, given the rains that we have had, and it will also have its cascading positive effect on other consuming segments During FY 2020-21, the brands of the Company have improved their competitive positions in each of the categories they operate in The Management of the Company has emerged intellectually and emotionally stronger and is going into FY 2021-22 with a set of proven initiatives for customer acquisition, cost and cash management as well as a well- developed agility",
        #                            'It is based on this understanding that the business plans for FY 2021-22 were made with a high level of ambition and substantial excitement and passion. The second wave of COVID-19 has come and caused a setback to those plans, but without taking away the medium-term opportunities and the advantages that those plans represented. Also, by September 2021, much of the country is likely to be vaccinated, paving the way for some kind of normalcy.',
        #                            'The Management is approaching the new FY 2021-22 with the same calmness and composure like in FY 2020-21 and is confident that it will be able to overcome all the challenges that come its way.']
    }




    # overall_doc_summary= get_overall_document_summary(llm_model,adani_chunk_list)

    # final_overall_doc_summary= remove_similar_mda_overall_summary(e5_embedding_model,overall_doc_summary)
    # print(overall_doc_summary)
    theme_based_summary= get_document_theme_summary(titan_mda_relevant_dict,llm_model)
    print(theme_based_summary)

    print("Completed")

main()