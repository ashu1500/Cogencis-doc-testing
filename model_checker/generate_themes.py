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
        Generate only 1 most important key header relevant for financial information in maximum 3-4 words from the given text.Please do not include any explaination for the key header.
        text: {text}
        key headers:
        """

        prompt = PromptTemplate(template=template, input_variables=["text"])
        result = llm.generate([prompt.format(text=chunk_text)])
        return result
    except Exception as e:
        print(e)
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
        print(e)
        raise e
    

def main():
    adani_discussion_points=[
        "Thank you so much. Hi Good Morning all. This is Robbie Singh, CFO of Adani Enterprise. I welcome you all to the earnings call to discuss Q1 FY23 results. AEL continues to create value for its shareholders as a successful incubator for the past two-and-a-half decades. This incubation model has created leaders in the respective sectors like Adani Ports, Adani Transmission, Adani Green Energy, Adani Total Gas, and Adani Wilmar and has delivered returns at a compound annual growth rate of 36% to shareholders. AEL holds a portfolio of businesses - both established and incubating - which are spread across different verticals in energy and utility, transport and logistics, direct to consumer and primary industries. Within primary industries it has established businesses of mining services and integrated resource management along with the developing vertical of metals. As our established business continue to sustain long term growth, we are making significant progress in our attractive incubation pipeline comprising of energy and utility which is Adani New Industries - it is a green hydrogen ecosystem and full service data center business AdaniConneX. In the transport and logistics we have Adani Airport Holdings and Adani Road Transport Limited businesses which will further accelerate value creation for Adani Enterprise shareholders. We are happy to inform that AEL has completed primary equity transaction of Rs.7700 Crores with Abu Dhabi based International Holding Company for 3.5% stake. This validates our strong capital management philosophy of equity funded growth and conservative leverage targets.",
        "Let me give you a quick update of our incubating businesses. In Adani New Industry portfolio as all of you would know we have announced investment of USD 50 billion over the next decade in developing green hydrogen ecosystem. This will be housed under Adani New Industry Limited. ANIL will have three business streams — (i) Manufacturing ecosystem to include module, cell, ingots, wafers and wind turbines, electrolyzers and associated ancillary equipment ecosystem. (ii) The green hydrogen generation include development of solar and wind power plants to produce green hydrogen (iii) Downstream products depending on the usage for ammonia, urea, methanol, etc. During the quarter we announced our partnership with TotalEnergies to develop the world’s largest green H2 ecosystem. TotalEnergies will acquire 25% stake in ANIL. While thetransaction will follow customary approval process, it takes the company one step ahead to produce the world’s least expensive electrons which will drive our ability to produce the world’s least expensive green hydrogen. Following are some of the updates on development: Existing capacity of 1.5 GW at Mundra is increasing to 3.5 GW and this additional 2 GW will be completed by September this year. With this the overall capacity will reach to 3.5 GW. Wind turbine erection for the first 5.2 MW wind turbine has been completed and testing and certification is underway. We expect completion in the next 6 months. We have identified three trial sites for initial testing of electrolyzers and we expect the testing to commence by end of this calendar year or early next year. From operational point of view, module sales from our manufacturing ecosystem within ANIL stood at 264 MW. EBITDA from these sales was at Rs. 42 Crores.",
        "In Adani Airport Holdings portfolio, passengers movement at the airports increased by 35% and it is now at 16.6 million which is approximately 85% of the pre-COVID numbers. Construction at Navi Mumbai International Airport has started and approximately 26% of the work is completed. In Adani Road Transport portfolio, we have signed concession agreement in May for Kagal-Satara road project of 65 KMs in Maharashtra under BOT basis. We have also received provisional COD for Bilaspur road project and construction activity is progressing well on other 8 road projects. The current road portfolio is now approximately Rs.38000 Crores of both operating and under development projects.",
        "A quick update of primary industries before I handover to my colleague Vinay. We achieved financial closure for our first metals business copper. This was led by SBI and it has been further down sold to various other banks. Financial performance of AEL for this quarter in terms of revenue number has increased 223% and now is at Rs.41066 Crores. Consolidated EBITDA increased by over 100% and is at Rs.1965 Crores with strong performance from both established and incubating businesses. Now I invite my colleague Vinay to take you through mining services and IRM business highlights. Vinay over to you!",
        "Thanks Robbie. Good Morning to all. In fact as far as the mining services business is concerned Adani Enterprise Limited is the pioneer of MDO concept in India with an integrated business model that spans across developing mines as well as the entire upstream and downstream activities. It provides the full service range - right from seeking various approvals, land acquisition, R&R, developing required infrastructure, mining, beneficiation and transportation to designated consumption plants. The company is also MDO for nine coal blocks and two iron ore blocks with combined peak capacity of 120 MMT per annum. These 11 projects are located in the state of Chhattisgarh, MP and Odisha. The mining production volume increased by 72% to 8.1 MMT on year-on-year basis and further dispatch increased by 58% to 7.2 MMT on year-on- year basis. The revenue from mining services increased by 18% to Rs. 677 Crores and EBITDA stood at Rs. 268 Crores versus Rs. 307 Cores on year-on-year basis on account of high operating costs. As Robbie told about copper business apart from financial closure, operational activities are progressing well and it is as per schedule. Additionally, we have also received PLI scheme approval for copper tube value added scheme for our copper business. As far as the IRM business is concerned, in terms of IRM business we have continued to develop business relationship withdiversified customers across various end-user industries. We retained number one player position in India and having the endeavor to maintain this position going forward. The volume in Q1 FY23 increased by 52% to 26.7 MMT, the EBITDA has increased by 72% to Rs.950 Crores on account of higher volumes. Thank you."
    ]
    chunk_headers_list=[]
    for items in adani_discussion_points:
        print("Theme generation")
        chunk_txt= theme_extraction_per_chunk(items,llm_model)
        chunk_header= extract_headers_from_themes(chunk_txt.generations[0][0].text)
        chunk_headers_list.append(chunk_header)
    print(chunk_headers_list)

main()