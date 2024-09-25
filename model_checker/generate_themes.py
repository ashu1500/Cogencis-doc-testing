import os
import logging
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import subprocess
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
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


#OVERALL DOCUMENT SUMMARY
def get_chunk_summary(llm, text):
    ''' Get summary of each chunk '''
    try:
        # Updated prompt template with clear rules about generating only a factual, single-paragraph summary
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
        
        # Create the prompt using the template
        prompt = PromptTemplate(template=template, input_variables=["text"])
        formatted_prompt = prompt.format(text=text)
        
        # Get the response from the language model
        text_summary = llm.generate([formatted_prompt])
        
        # Extract the summary
        chunk_summary = extract_summary_section_perchunk(text_summary.generations[0][0].text.strip())
        
        # Clean any instruction-like content from the summary
        cleaned_summary = clean_summary(chunk_summary)
        
        return cleaned_summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        raise e

   
def get_overall_document_summary(llm_model, chunk_list):
    ''' Get overall summary of the document '''
    try:
        overall_summary = []
        
        for text_chunk in chunk_list:
            print("Chunk summary started")
            summary = get_chunk_summary(llm_model, text_chunk)
            overall_summary.append(summary.strip())  # Append cleaned summary
            print("Chunk summary generated")
        
        # Join all chunk summaries with a space separating them
        final_summary = " ".join(overall_summary)
        
        # Final cleaning step to remove any lingering instruction-like content
        final_summary = clean_summary(final_summary)
        
        return final_summary
    
    except Exception as e:
        print(f"Error generating overall summary: {e}")
        raise e

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
    indigo_discussion_themes= get_final_transcript_themes(llm_model,indigo_discussion_data)
    print("Discussion_themes: ",indigo_discussion_themes)
    indigo_question_themes= get_final_question_themes(llm_model,indigo_questions_list)
    print("Questions_themes: ",indigo_question_themes)
    print("Completed")

main()