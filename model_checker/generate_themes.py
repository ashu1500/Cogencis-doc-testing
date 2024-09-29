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
        reduced_text = ''
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > limit:
                break
            reduced_text += sentence + ' '
            current_length += len(sentence) + 1
    
        return reduced_text.strip()
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
    
    e5_embedding_model = SentenceTransformer('intfloat/e5-large')
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
    overall_doc_summary= get_overall_document_summary(llm_model,maruti_chunks_list)
    final_overall_doc_summary= remove_similar_mda_overall_summary(e5_embedding_model,overall_doc_summary)
    print(final_overall_doc_summary)

    print("Completed")

main()