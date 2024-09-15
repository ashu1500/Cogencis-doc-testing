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
        template = """
        You are tasked with summarizing the following text, which is delimited by triple backquotes:
        - Do not include any kind of headers, emojis, asterisks, symbols, requests, questions, instructions, or explanations in the summary.
        - The summary must only include factual information from the text.
        - Avoid restating any part of these instructions in the summary.
        ```{text}```
        SUMMARY:
        """
        
        # Create the prompt
        prompt = PromptTemplate(template=template, input_variables=["text"])
        formatted_prompt = prompt.format(text=text)
        
        # Get the response from the language model
        text_summary = llm.generate([formatted_prompt])
        
        # Extract the summary
        chunk_summary = extract_summary_section_perchunk(text_summary.generations[0][0].text)
        
        # Clean any instruction-like content from the summary
        chunk_summary = clean_summary(chunk_summary)
        
        return chunk_summary

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
    
    indigo_discussion_summary= get_overall_document_summary(llm_model,indigo_discussion_data)
    print("Completed")
    print(indigo_discussion_summary)

main()