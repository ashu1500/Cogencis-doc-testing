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
accelerator = Accelerator()

def load_llama_model():
    try:
        logging.info("llama model loading")
        hf_token="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"
        subprocess.run(f'huggingface-cli login --token={hf_token}',shell=True)
        model_path= os.path.join("model")
        model_pipe = pipeline(task="text-generation", model = model_path,tokenizer= model_path,device_map="auto")
        model_pipe = accelerator.prepare(model_pipe)
        final_pipeline= HuggingFacePipeline(pipeline = model_pipe, model_kwargs = {'temperature':0})
        logging.info("model loaded successfully")
        return final_pipeline
    except Exception as e:
        logging.error(e)


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
        fixed_themes=["Financial Performance","Merger and Acquisition","Risks and Challenges","Market Trends and Outlook","Competitive Positioning"]
        final_themes= set(fixed_themes+generated_themes)
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
            if cos_sim(theme_embedding,chunk_embedding).item()>0.75:
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
            Your summary should consist of exactly 10 points, each point having at least 20 words long.Include a mix of direct observations and inferences drawn from the text.
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
        Your summary should  consist of exactly 10 points, each at least 20 words long. Blend factual information with insightful inferences.
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
            else:
                processed_lines.append(line.strip())
        processed_text = "\n".join(processed_lines)
        return processed_text
    except Exception as e:
        logging.error("Error removing headers: %s", e)
        raise e
    

def generate_theme_summary(theme, chunk_data, llm):
    ''' Generate summary for a theme'''
    try:
        combined_summary= ""
        for chunk in chunk_data:
            keywords_list = keywords_theme_extraction(theme, chunk, llm)
            print("Keywords generated")
            chunk_summary = summary_generation_perchunk(keywords_list, chunk, llm)
            print("Summary per chunk generated")
            combined_summary += chunk_summary
        actual_list= [x.strip() for x in combined_summary.split('\n')]
        joined_summary= "".join(actual_list)
        summary_list= textwrap.wrap(joined_summary,14000)
        output_summary=""
        for summary in summary_list:
            generated_summary= get_final_summary(summary,llm)
            output_summary+=generated_summary
        final_summary= remove_headers(output_summary)
        return final_summary
    except Exception as e:
        logging.error(e)
        raise e
    

def get_document_theme_summary(chunk_dictionary,llm):
    '''Get theme-based summary of document'''
    try:
        theme_based_summary={}
        for theme,chunk in chunk_dictionary.items():
            if chunk:
                print("Theme summary started")
                theme_based_summary[theme]= generate_theme_summary(theme,chunk,llm)
                print("Theme summary generated")
            else:
                continue
        return theme_based_summary
    except Exception as e:
        logging.error(e)
        raise e











def main():
    tcs_chunks= ["Unfortunately, Mr. Milind Lakkad – our Chief HR Officer could not join us today due to a bereavement in his family. Our management team will give a brief overview of the company’s performance, followed by a Q&A session. As you are aware, we do not provide specific revenue or earnings guidance. And anything said on this call which reflects our outlook for the future. or which could be construed as a forward-looking statement, must be reviewed in conjunction with the risks that the company faces. We have outlined these risks in the second slide of the quarterly fact sheet available on our website and mailed out to those who have subscribed to our mailing list. With that, I’d like to turn the call over to Rajesh. Rajesh Gopinathan: Thank you, Kedar, and good morning, good afternoon and good evening to all of you. We are starting out in FY 23 on a strong note, growing 16.2% in rupee terms, 15.5% in constant currency terms and 10.2% in dollar terms. : Page 1 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) We announced our salary increases with effect from April 1st. Reflecting that and other employee costs we incurred in Q1, our operating margin for the quarter was at 23.1%, a contraction of 1.9% sequentially and 2.4% year-on-year. Net margin was at 18%. I will now invite Samir and NGS to go over different aspects of our performance during the quarter. I'll step in again later to provide some more color on the demand trends that we're seeing. Over to you, Samir. Samir Seksaria: Thank you, Rajesh. Let me first walk you through the headline number. In the first quarter of FY 23 our revenue grew 15.5% YoY on a constant currency basis. Reported revenue in INR was `527.58 billion, a year-on-year growth of 16.2%. In dollar terms, revenue was $6.78 billion, a year-on-year growth of 10.2%. Let me now go over the financials",
                 "I'll step in again later to provide some more color on the demand trends that we're seeing. Over to you, Samir. Samir Seksaria: Thank you, Rajesh. Let me first walk you through the headline number. In the first quarter of FY 23 our revenue grew 15.5% YoY on a constant currency basis. Reported revenue in INR was `527.58 billion, a year-on-year growth of 16.2%. In dollar terms, revenue was $6.78 billion, a year-on-year growth of 10.2%. Let me now go over the financials. As Rajesh mentioned, we announced salary increases of 5% to 8% and much higher for top performers with effect from April 1. This had a 1.5% impact on operating margins. Continued supply side challenges entailed additional expenses, such as backfilling expenses and higher subcontractor usage. This and normalizing travel expenses negated various operational efficiencies, resulting in an operating margin of 23.1%, a sequential contraction of 1.9%. Net income margin was at 18%. Our effective tax rate for the quarter was 25.5% and our accounts receivable was at 63 day sales outstanding in dollar terms, down one day compared to Q4. Net cash from operations was ₹108.1 billion, which is a cash conversion of 114%. Free cash flows were ₹100.68 billion. Invested funds as on 30th June stood at ₹527.6 billion, and the board has recommended an interim dividend of ₹8 per share. Since Milind is not here today, I will take you through the HR numbers now. On the people front, our workforce strength crossed the 600,000 mark this quarter, ending this quarter with 606,331 employees. We continue to hire talent from across the world, with a net addition of 14,136. It is a very diverse workforce with 153 nationalities represented and with the women making up 35.5% of the base. We remain committed to investing in organic talent development towards building the next generation G&T workforce. In Q1, TCSers clocked 12 million learning hours, resulting in the acquisition of 1.7 million competencies",
                'We continue to hire talent from across the world, with a net addition of 14,136. It is a very diverse workforce with 153 nationalities represented and with the women making up 35.5% of the base. We remain committed to investing in organic talent development towards building the next generation G&T workforce. In Q1, TCSers clocked 12 million learning hours, resulting in the acquisition of 1.7 million competencies. LTM attrition in IT services was at 19.7%, and we think it will rise further in Q2, after which it should start tapering. Now over to you, NGS, for some color on our segments and products and platforms. N G Subramaniam: Thank you, Samir. Let me walk you through our segmental performance details for the quarter. All the growth numbers are on year-on-year constant currency basis. : Page 2 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) All our verticals showed good growth in Q1. Growth was led by Retail and CPG which grew 25.1% after a similar strong growth last quarter, Communications and Media grew 19.6%, while the Manufacturing as well as Technology & Services verticals both grew 16.4%. BFSI, our largest vertical, grew 13.9% while Life Sciences and healthcare grew by 11.9%. By geography, growth was led by North America which grew 19.1%. UK grew 12.6%, while continental Europe grew 12.1%. In emerging markets, India grew by 20.8%, Asia Pacific grew 6.2%, Latin America by 21.6% and Middle East and Africa grew by 3.2%. Our portfolio of products and platforms continue to do well. ignio™, our cognitive automation software suite signed up 28 new customers and five clients went live during the quarter. In addition, 15 existing clients acquired new licenses of the suite during the quarter. TCS filed three patents around ignio during the quarter and was granted one. The market demand for ignio trained professional continue to grow',
                "2%, Latin America by 21.6% and Middle East and Africa grew by 3.2%. Our portfolio of products and platforms continue to do well. ignio™, our cognitive automation software suite signed up 28 new customers and five clients went live during the quarter. In addition, 15 existing clients acquired new licenses of the suite during the quarter. TCS filed three patents around ignio during the quarter and was granted one. The market demand for ignio trained professional continue to grow. The number of ignio trained professionals stand at 14,134 while number of ignio-certified professionals is 4,294 to-date. A global luxury hotel brand has deployed ignio AIOps to manage digital assets, including hotel content and promotional offers published globally for thousands of its properties across many brands. The solution detects anomalies in the properties across all brands, eliminates noise and averts outages in reservation and offers, thereby enhancing customer experience and averting loss of business. TCS BaNCS™, our flagship product suite in the financial services domain had three new wins and four go lives during the quarter. I'm very pleased to share that in the recently published IBS Intelligence Sales League Table 2022, TCS was ranked number one in investment and fund management and in fraud management; and number two worldwide in the areas of insurtech, Islamic banking, wholesale banking, treasury and capital markets. Our Banking Service Bureau in Israel, which powers the country's first digital only bank has won a second client. One of UK's leading insurance organizations has expanded its partnership with TCS Diligenta to launch innovative new products to deliver improved customer experience and drive competitive differentiation using the TCS Insurance Platform, powered by TCS BaNCS™. : Page 3 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Our Quartz Blockchain platform had two go lives in Q1",
                "One of UK's leading insurance organizations has expanded its partnership with TCS Diligenta to launch innovative new products to deliver improved customer experience and drive competitive differentiation using the TCS Insurance Platform, powered by TCS BaNCS™. : Page 3 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Our Quartz Blockchain platform had two go lives in Q1. The Quartz Smart Solution for Bond Issuance is now live at one of the leading central securities depositories in India. It facilitates real-time exchange of information among multiple stakeholders, like issuers, trustees, credit rating agencies, depositories and stock exchanges on a private permissioned blockchain platform. The solution helps eliminate potential double counting of underlying assets and ensures greater transparency and governance in the bond issuance lifecycle. In Life Sciences, our award-winning advanced drug development suite had one new win, our first client for this suite in Japan. Our HOBS™ suite of solutions for communication service provider had one new win and four go lives in Q1. We also launched a new release of HOBS™ Business Assurance, which enables clients to deploy faster and more easily integrate with real-time data sources, visualize revenue leakages and take corrective actions in near real-time. TCS TwinX™, our AI-based digital twin solution had one win during this quarter and one client went live. TCS Optumera™, our AI-powered retail merchandising suite had one significant win and three go lives. TCS iON continues to expand its presence in the vocational education domain, entering into partnerships with three academic institutions – CRISP, Apollo Medskills and MIT World Peace for learning programs",
                'TCS TwinX™, our AI-based digital twin solution had one win during this quarter and one client went live. TCS Optumera™, our AI-powered retail merchandising suite had one significant win and three go lives. TCS iON continues to expand its presence in the vocational education domain, entering into partnerships with three academic institutions – CRISP, Apollo Medskills and MIT World Peace for learning programs. TCS National Qualifier Test, which is gaining traction as the preferred entry level hiring platform for corporate India has made significant progress over the last one year with over 900 corporate partners at the end of Q1. Let me now cover our client metrics. Movement of clients up the revenue bands is a clear demonstration of our customer- centric strategy at work. By providing a great experience and building transformational solutions that deliver high impact outcomes, we gain goodwill and trust which translates into a steady broadening and deepening of our relationships with our clients. In Q1, we had robust client additions in every revenue bucket compared to the year ago period. We added nine more clients in the $100 million+ band, bringing the total to 59. We added 19 more clients in the $50 million+ band bringing the total to 124. We added 31 more clients to the $20 million+ band, bringing the total to 272. We added 41 : Page 4 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) more clients in the $10 million+ band bringing the total to 446. We added 64 more clients in the $5 million+ band, bringing the total to 650. We added 78 more clients in the $1 million+ band, bringing the total to 1,196. Over to you, Rajesh, to give your insights on demand drivers during the quarter. Rajesh Gopinathan: Thank you, NGS. All the demand drivers we have been speaking about for the last two years continue to be very much in play',
                'We added 64 more clients in the $5 million+ band, bringing the total to 650. We added 78 more clients in the $1 million+ band, bringing the total to 1,196. Over to you, Rajesh, to give your insights on demand drivers during the quarter. Rajesh Gopinathan: Thank you, NGS. All the demand drivers we have been speaking about for the last two years continue to be very much in play. In Q1, we saw plenty of deals in each of these categories, be it cloud adoption, operating model transformation, vendor consolidations, or G&T engagements. Cloud adoption continues to be a powerful growth driver. We have plenty of deals this quarter once again and more in the pipeline. Our strong partnerships with the hyper scalers, deep expertise on their platforms, and our industry-specific solutions and domain knowledge have helped us gain share in this space. As you might have seen already, we won four Global Partner of the Year Awards and two Regional Partner Awards from Microsoft this quarter. Prior to that, we won two Partner of the Year Awards from Google Cloud. The full list is published in our earnings press release available on our website. Let me now spend a little bit of time on the operations transformation opportunity, which is a big contributor to our revenue growth, order book and pipeline. We see three distinct trends here. • There are more and more clients looking to leverage next-generation technologies to create leaner, agile, resilient and efficient operation with an intent to plow back the savings into the business transformation initiatives. We have been big beneficiaries of this trend. A game changer has been TCS Cognix™, our AI-driven human machine collaboration suite, which we launched in 2020. It features a large number of pre-built configurable and reusable digital solutions, covering a wide range of industries and business function. Today, the suite consists of 530 value builders, covering an extensive set of business and technology use cases',
                "We have been big beneficiaries of this trend. A game changer has been TCS Cognix™, our AI-driven human machine collaboration suite, which we launched in 2020. It features a large number of pre-built configurable and reusable digital solutions, covering a wide range of industries and business function. Today, the suite consists of 530 value builders, covering an extensive set of business and technology use cases. Over 267 customers have leveraged Cognix™ across business and operations to drive business outcomes. In Q1, we had six new wins featuring Cognix™. • The second trend is the growing incidence of multi services integrated deals. By bringing multiple elements of the operation stack, such as business processes, applications, databases, operating systems and underlying infrastructure, all within the scope of a single service provider, clients are not : Page 5 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) only able to drive greater accountability, but also take up transformation programs that are more holistic in nature. This trend plays to our strengths in terms of our structure, as well as our ability to bring together different capabilities from across TCS to create a seamless service delivery team. We have six new multi service integrated deals in Q1 versus two to three deals per quarter in FY'22. • An adjunct to these two trends is that clients are looking to reduce complexity by bringing down the number of service providers they work with to a few select partners who possess the right innovation capabilities and can scale. We won several large deals in Q1 which were vendor consolidation exercises. Let me switch gears and talk about the airline industry which is no stranger to consolidation. There is an industry level transformation taking place there, designed to make airlines much more competitive and customer-centric, and TCS is playing a big role in helping industry players transform",
                'We won several large deals in Q1 which were vendor consolidation exercises. Let me switch gears and talk about the airline industry which is no stranger to consolidation. There is an industry level transformation taking place there, designed to make airlines much more competitive and customer-centric, and TCS is playing a big role in helping industry players transform. The airline industry is heavily dependent on ticketing platforms whose legacy proprietary models restrict airlines’ ability to offer new products and services in a direct- to-consumer or B2C model. This is preventing them from establishing competitive differentiation and is resulting in an opportunity loss. To address this problem, the International Air Transportation Association or IATA, has come up with a new open standard called New Distribution Capabilities, or NDC, which airlines can use to dynamically create personalized offers for customers. This enables a customer-specific bundling of preferences, such as number of check-in bags, seat choices, in-flight amenities, refreshments, or even in-flight shopping, priced uniquely for sale through their own channels or through third-party aggregators and travel agents. TCS is currently engaged with three airlines in implementing NDC, and in discussion with a couple more. For a leading UK-based carrier, which is an early adopter of this initiative, TCS did the end-to-end business process redesign and solution development for NDC adoption. Since deployment, the airlines NDC channel daily bookings have grown 500%, contributing to more than 20% of the overall indirect bookings and helping the airline become a digital retailer and personalize the customers experience and drive new revenues',
                'For a leading UK-based carrier, which is an early adopter of this initiative, TCS did the end-to-end business process redesign and solution development for NDC adoption. Since deployment, the airlines NDC channel daily bookings have grown 500%, contributing to more than 20% of the overall indirect bookings and helping the airline become a digital retailer and personalize the customers experience and drive new revenues. Moving on to the next theme of how TCS is helping clients adopt innovative technology- enabled business models that drive new revenue streams, let me share two examples: : Page 6 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) • TCS has partnered with a senior care of a leading US-based provider of long- term care insurance in launching a new line of business and assisted living marketplace. The platform will allow seekers of long-term care to find the closest assisted living care providers and also offer subscription-based consultation. Using design thinking approach, TCS helped define the product roadmap and carved a playbook for product positioning and targeting customer base in competitive market. The new marketplace will be formally launched this month and is targeting million customers over the next three years. In addition to driving new subscription-based revenues, the platform will allow the parent company to embed its insurance product in every transaction in the marketplace. • Similarly, a large Fortune 500 electric gas utility has launched a new business model to generate a new revenue stream based on home energy services, that is providing warranty repair, refurbishment and replacement services of home appliances, such as air conditioners, washing machines and refrigerators. It partnered with TCS to build the platform needed to enable the service delivery that is central to this new business',
                '• Similarly, a large Fortune 500 electric gas utility has launched a new business model to generate a new revenue stream based on home energy services, that is providing warranty repair, refurbishment and replacement services of home appliances, such as air conditioners, washing machines and refrigerators. It partnered with TCS to build the platform needed to enable the service delivery that is central to this new business. We built a new cloud-based field service automation solution based on a third- party platform to provide advanced capabilities in intelligent scheduling, dispatching and mobile workforce management to transform field operations. The new solution has helped significantly improve the predictability of services, enhancing customer experience. It is also highly scalable. The utility now plans to expand to more states in the next six months, and achieve half a billion dollars in revenue over the next 24-months. In earlier calls, I have given many examples of enterprises partnering with us to achieve their sustainability goals. There are a couple of stories where sustainability is directly linked to revenue growth. • TCS is the strategic partner for a global alliance, that is focused on decarbonizing the food value chain by helping to transform farming practices globally, promote regenerative farming and generate reliable farm carbon credits and certified climate smart crops. We have helped them conceptualize the business model and build the enabling digital platform for farmer enablement and adoption. It provides a simple and quick way to onboard farmers and other stakeholders, validate carbon smart : Page 7 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) practices through analytics, enables a carbon credit declaration by farmers and provides personalized recommendations of the practices',
                'We have helped them conceptualize the business model and build the enabling digital platform for farmer enablement and adoption. It provides a simple and quick way to onboard farmers and other stakeholders, validate carbon smart : Page 7 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) practices through analytics, enables a carbon credit declaration by farmers and provides personalized recommendations of the practices. The platform drives revenue for the alliance through carbon credit trading, improves farmer incomes and reduces scope pre-emissions and helps the private sector decarbonize. • Similarly, in Massachusetts, we partnered with a leading utility to build a platform that would help them roll out the Solar Massachusetts Renewable Target or SMART incentive system to accelerate solar power adoption in the state. Consumers who install solar panels and storage on the property and to the utilities grid, qualify to receive a monthly incentive payment directly from the state government. The TCS-built solution includes onboarding of new generators, a customer application that helps keep track of the units generated, a pricing engine and a billing system. Using this, the utility was able to achieve regulatory targets on renewable capacity onboarding on the grid. It was able to enroll more than 10,000 residential customers in the last 12 months and is now expanding the program to other states. M&A remains the preferred growth strategy for many enterprises and over the last few years, we have built up a significant business catering to our clients need to integrate new entities that they have acquired or diverse businesses that are no longer strategic. This quarter too, we have few wins in that space. • A global leader in health, nutrition and biosciences is building its nutrition portfolio through acquisitions',
                'M&A remains the preferred growth strategy for many enterprises and over the last few years, we have built up a significant business catering to our clients need to integrate new entities that they have acquired or diverse businesses that are no longer strategic. This quarter too, we have few wins in that space. • A global leader in health, nutrition and biosciences is building its nutrition portfolio through acquisitions. In 2020, after executing its largest and most complex acquisition in two decades, it engaged TCS to help integrate the new entity. Our team helped envision changes to the enterprise model to allow for a seamless integration. The first phase of integration was completed in Q1 and helped bring onboard the new entity, provide common ways of working and deliver synergies. We commonly think of G&T as front-end customer-facing work. But we have plenty of examples of operations transformation that reimagine processes at the backend using digital technologies like AI, resulting in much faster turnaround times, much higher throughput, and therefore, more revenue. Let me give you a couple of examples of these: : Page 8 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) • For a leading global HR services firm, TCS partnered to transform their core recruitment process leveraging next-generation technologies. Their existing processes entailed recruiters spending 60% to 70% of their effort on profile sourcing and screening, of which 50% was devoted to candidate outreach. Over half the outreach efforts was wasted due to lack of response or declines. TCS redesigned the end-to-end recruitment process, taking a Machine First™ approach. We built bots to automate the background check initiations across clients and deployed a third-party AI-powered candidate outreach platform to significantly bring down recruiter effort and completely streamline the candidate screening process',
                'Over half the outreach efforts was wasted due to lack of response or declines. TCS redesigned the end-to-end recruitment process, taking a Machine First™ approach. We built bots to automate the background check initiations across clients and deployed a third-party AI-powered candidate outreach platform to significantly bring down recruiter effort and completely streamline the candidate screening process. This resulted in a 300% increase in number of shortlist candidates, a 15-20% reduction in turnaround time, and most importantly, a 42% increase in hiring throughput, which is directly linked to revenue growth for the company. • Similarly, a large US insurer has engaged TCS to transform its personal lines of business. The vision is to simplify business processes, modernize the technology stack, enable faster launch of new products and product enhancements for greater competitiveness, enhanced customer experience and profitable growth. TCS helped reimagine and implement significant improvements in pricing, risk assessment, acquisition and servicing of customers using a third-party platform. On completion, the new solution will enable ensure issuance of a new policy in eight minutes versus the current 24-hours. It will significantly improve the quality, granularity and timeliness of data and analytics to support better targeting and personalization. The solutions or features are expected to help improve business agility, enhance customer experience and drive growth. Coming to the Q1 order book, as you know, we had an all-time high order book TCV last quarter. On the back of that, we again had a strong set of deal wins in Q1 amounting to a TCV of $8.2 billion. The deal mix is very heterogeneous, with the largest two deals being just over $400 million in size. By vertical, BFSI had a TCV of $2.6 billion, while retail order book stood at $1.2 billion. The TCV of deals signed in North America stood at $4.5 billion',
                "Coming to the Q1 order book, as you know, we had an all-time high order book TCV last quarter. On the back of that, we again had a strong set of deal wins in Q1 amounting to a TCV of $8.2 billion. The deal mix is very heterogeneous, with the largest two deals being just over $400 million in size. By vertical, BFSI had a TCV of $2.6 billion, while retail order book stood at $1.2 billion. The TCV of deals signed in North America stood at $4.5 billion. Looking at the strong order book and our pipeline, this is good visibility for the next few months. We have not seen any budget cuts or deferments so far. In conversations with clients, we see continuing investments in technology. Some clients, particularly in Europe have expressed concerns about the macroeconomic fallout of the ongoing : Page 9 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) conflict there. But the predominant sense is that technology spending will be resilient. That said, given the macro level uncertainties, we remain very watchful. With that, we'll open the line for questions. Over to you, Kedar. Moderator: We will now begin the question-and-answer session. The first question is from the line of Ankur Rudra from JP Morgan. Please go ahead. Ankur Rudra: A few questions from me today. First, Rajesh, could you maybe elaborate on the tone of conversations you had with clients on perhaps new growth and transformation contracts. How has that evolved this quarter? And in addition to that, how do you think the pipeline formation has been? And finally, how should we interpret the fact that deal signing to the growth on a year-over-year basis seems to be sort of flattening out and the book-to-bill ratio seem to be lower than the last couple of years? Rajesh Gopinathan: Hi Ankur. As you can imagine, we have been staying very close to our customers given the overall news flow that we see all around us",
                "How has that evolved this quarter? And in addition to that, how do you think the pipeline formation has been? And finally, how should we interpret the fact that deal signing to the growth on a year-over-year basis seems to be sort of flattening out and the book-to-bill ratio seem to be lower than the last couple of years? Rajesh Gopinathan: Hi Ankur. As you can imagine, we have been staying very close to our customers given the overall news flow that we see all around us. We have been at all levels personally reaching out, meeting with as many customers as we can. The general sense that we're getting is that at the operating level, the demand continues to be very strong and unabated. There is high visibility of project funding; there is appetite for continuing investments and in fact, for acceleration. So, the demand environment on an immediate basis continues to be very strong. Some conversations at senior executive levels – CEO, COO, level, etc., are more about what they see overall about the whole macro environment that you spoke about. But that elevated conversation does not seem to be reflecting in the actual budgets and the spend. We have not seen any project cancellations, pull backs, nothing of that sort. And this is across both transformation projects as well as on the optimization projects. Overall, we are, as you can imagine, staying very vigilant, maximizing contact with customers and taking it on a case-to-case basis, and reacting to what we have on hand and maintaining that stance. Ankur Rudra: Second part, Rajesh, should investors read anything into the fact that the book-to-bill ratios have dropped a bit this year versus the last years at this time? Rajesh Gopinathan: TCV is a forward-looking number. It is what it is. I gave you the commentary on what we're seeing on the field. The actual closures, that number is there. When we look at our pipeline and overall trend, I don't think there's anything that is alarming for us. A 1",
                "Ankur Rudra: Second part, Rajesh, should investors read anything into the fact that the book-to-bill ratios have dropped a bit this year versus the last years at this time? Rajesh Gopinathan: TCV is a forward-looking number. It is what it is. I gave you the commentary on what we're seeing on the field. The actual closures, that number is there. When we look at our pipeline and overall trend, I don't think there's anything that is alarming for us. A 1.2 times book-to-bill is still quite strong. So, nothing more than that from our perspective, but we're also being very vigilant. : Page 10 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Ankur Rudra: Of course, nobody knows the future, but I know you've always said it like you see it, if we do hit the R word recession, what are the key points you would look for and how do you feel about TCS position in the market now versus the previous down cycles over the last two decades? Rajesh Gopinathan: Ankur, we have been continuously investing in improving our strength on both sides of the market opportunity. That is what we call the twin-engine strategy of cost and optimization, and growth and transformation. The transformation part of the agenda is equally applicable on either side of that economic scenario. So, from a positioning perspective, we are quite happy with where we are both in terms of capabilities as well as our positioning in the market and our client engagement. That's an area that we have been continuously investing in. From an operating perspective, we have built up the capacity, we are well positioned to be able to move rapidly in either direction, if things change in any way. So, I'm quite confident that our agility is going to be able to be our biggest strength, and also our reputation and our client relationships will come through significantly if there is some uncertainty or any volatility. Ankur Rudra: Last question on margins if I can",
                "That's an area that we have been continuously investing in. From an operating perspective, we have built up the capacity, we are well positioned to be able to move rapidly in either direction, if things change in any way. So, I'm quite confident that our agility is going to be able to be our biggest strength, and also our reputation and our client relationships will come through significantly if there is some uncertainty or any volatility. Ankur Rudra: Last question on margins if I can. Were there more headwinds in 1Q than previously thought, Samir, and do you think the range of outcomes now for the year looks maybe softer than what you thought a quarter ago? Rajesh Gopinathan: Could you repeat that, Ankur? You're saying as far as margin goes… Ankur Rudra: As far as margin goes, do you think you saw more headwinds in June quarter versus what you thought in the beginning of the quarter in terms of the percentage and how do you think about the rest of the year now? Rajesh Gopinathan: Nothing different from what we saw in the beginning of the quarter. There are two things that are playing out there. One is the wage increase, which is a fairly well understood one. The other is the continuing demand environment and the attrition environment that is leading to the increased operating costs on the employee side. The attrition is not totally unanticipated, but it is continuing and we think that probably it will take another few more months before it will start to come down. So, till then the margin pressures will continue but we hope to sequentially improve from where we are, given that we have taken a hit on that completely. Moderator: The next question is from the line of Sandip Agarwal from Edelweiss. Please go ahead",
                "The attrition is not totally unanticipated, but it is continuing and we think that probably it will take another few more months before it will start to come down. So, till then the margin pressures will continue but we hope to sequentially improve from where we are, given that we have taken a hit on that completely. Moderator: The next question is from the line of Sandip Agarwal from Edelweiss. Please go ahead. : Page 11 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Sandip Agarwal: Rajesh, I have only one question that, when you're talking to the senior executives in your client side, do you see that there is increased recognition in last few quarters that technology is much more core to their business for their growth and profitability than it being something which could be contracted or expanded based on the overall environment or based on the company's performance? The thing I'm trying to understand here is if you see some of the US retailers, they have sounded caution in terms of revenue growth and PAT growth. But at the same time, I think that they have to spend in technology to be relevant in the omnichannel mode. And also secondly, I think, outsourcing is always the most efficient or cost-efficient way of managing things in tough times. So, what is your sense on that, is it really like two decades back or one decade back when you could contract or expand based on your profit and loss account or it is more core to business and to achieve the top line number you need to spend, so, it is not at all discretionary in nature, what is your sense when you speak to your top clients? Rajesh Gopinathan: Sandip, industry-after-industry, the primacy of the technology spend continues to increase and that's reflected in our conversations with clients at senior levels also. So, there is much greater understanding, much greater appreciation of what their technology agenda is",
                "So, there is much greater understanding, much greater appreciation of what their technology agenda is. More importantly, even the most senior person is fully clued in on what the technology strategy or the technology-enabled strategy for the company is respectively. So, that awareness and that centrality of that strategy to their overall plans is very high. Having said that, definitely, if there is an economy wide slowdown, it is likely to have some ripple effect across all lines of spend, but resiliency of technology in the overall mix is unlikely to get diluted from where it has been in the last two years. Sandip Agarwal: On this subcon expense, we are still seeing a significant jump on a quarter-on-quarter basis and what I understand is that it may be partly due to attrition numbers going up. So, is it purely because of attrition going up or it is also related to some restrictions, which were there still in travel side and not been able to spend resources on time or what it is or it is sudden demand, sudden startup project, what is causing that continuous shift in subcon? Rajesh Gopinathan: Sandip, it's a combination of all of it. Definitely, attrition has a role and supply constraints in some of the local markets have a role. Also the fact that it is a variable cost and therefore it has strategic value to us in an uncertain environment. So, all aspects of it play out. We are keeping a close watch on it and we will be reacting to it depending on how the demand/supply situation plays out. : Page 12 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Moderator: The next question is from the line of Kumar Rakesh from BNP Paribas. Please go ahead. Kumar Rakesh: My first question was again on the margin side. So, on the subcon side, hypothetically speaking, suppose we enter into a weak demand scenario, and that is one of the levers which we potentially have to improve our cost structure",
                ": Page 12 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Moderator: The next question is from the line of Kumar Rakesh from BNP Paribas. Please go ahead. Kumar Rakesh: My first question was again on the margin side. So, on the subcon side, hypothetically speaking, suppose we enter into a weak demand scenario, and that is one of the levers which we potentially have to improve our cost structure. In such a scenario, how much of this subcon benefit we can potentially drive? What I'm trying to understand is how much of our cost impact is coming from subcon currently given the supply constraint which we are facing? Samir Seksaria: So, our subcontractor expenses currently are at 9.7% of our revenues, and have moved up from about 7% levels to where we are currently. And as Rajesh pointed out, we have proactively and strategically invested in creating a bench. Our current priorities are to stay focused on capturing the demand. And we have known how to balance on the subcontractor side. And as the need arises, we'll be able to realign it or balance it. Kumar Rakesh: So, what I understand is that we can stabilize it at 7% if the need arises, right? Samir Seksaria: Yes. Kumar Rakesh: My second question was around the fresher hiring target we had set about 40,000. How we are progressing on that and has that changed that target? Rajesh Gopinathan: No, we are on track for that and we're progressing well on it. This quarter reflecting what we already had in the system, we've been a bit lighter on the one which typically has been our long term trend, Q1 is a lighter quarter for trainee absorption whereas Q2 and Q3 are the primary quarters. Last year, we had gone very aggressive and hired through the year to build up more than 100,000 trainee bench. This year is more than normal. We're progressing well on that 40,000 mark. Moderator: The next question is from the line of Gaurav Rateria from Morgan Stanley",
                "Last year, we had gone very aggressive and hired through the year to build up more than 100,000 trainee bench. This year is more than normal. We're progressing well on that 40,000 mark. Moderator: The next question is from the line of Gaurav Rateria from Morgan Stanley. Please go ahead. Gaurav Rateria: So, first question is with respect to the UK market. If you look at the YoY growth, it has been actually slowing down and it's a clear divergence compared to the North America market, which is continuing to remain very, very strong. So, what's really going on there, are the trends or the discussions actually fructifying in the form of a little bit of a slowdown in the velocity of the deal closures, or the ramp ups, any trend that you can highlight will be helpful. N G Subramaniam: NGS here. Both in the UK and North American market, we don't see any anomalies or abnormalities in terms of customer behavior or the deal closure trends. All of this : Page 13 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) remains normal and in line with typically what we experience. Specifically in the UK, there have been some concerns about the increased cost of living that people experience, etc., But then, it's all in the macro level discussions that we have. But overall, as Rajesh pointed out, customers across our verticals, Manufacturing or Retail or Financial services, have not expressed anything, which is something that that will cause a concern for us at this moment. Gaurav Rateria: Second question is on the margin. So, getting back to 25% looks like an immediate priority, but should one thing that from a full year perspective, the margins could remain a tad lower than 25% as it will take some bit of progression to come to 25% over the coming quarters? Samir Seksaria: That has always been the case. Our Q1 margins are subdued when we gave out the full cycle of increment",
                "Gaurav Rateria: Second question is on the margin. So, getting back to 25% looks like an immediate priority, but should one thing that from a full year perspective, the margins could remain a tad lower than 25% as it will take some bit of progression to come to 25% over the coming quarters? Samir Seksaria: That has always been the case. Our Q1 margins are subdued when we gave out the full cycle of increment. And we do look forward to clawing it back during the period. And our target would be to reach and cross 25% as far as possible. Gaurav Rateria: Lastly, any change in your expectations with respect to price improvements that you were expecting, realization improvement that you were expecting as part of your client conversations a couple of months back that you had highlighted? Rajesh Gopinathan: No, it continues to be positive. Rather than large scale contract level changes, we're seeing more pointed ones on existing contracts and the newer contracts getting some uplift. So, we're seeing smaller increases in existing ones and newer contracts coming in at better terms. Also with existing renewals, certain terms like COLA are much better being able to push through. But the aggregate impact of it is still not positive. So, our overall realization numbers reflect the excess capacity that we have, and are still negative on a sequential basis. Moderator: The next question is from the line of Ravi Menon from Macquarie. Please go ahead. Ravi Menon: Rajesh, are there any soft spots that you see in any vertical you're seeing in the pipeline shift towards more efficiency or programs? Rajesh Gopinathan: No, Ravi, we're not seeing much from a vertical perspective. It seems to be fairly balanced across. Nothing at this stage that you can call out at a vertical level. Ravi Menon: We had heard your comments in the press conference, you were talking about how the US will likely be the main driver of growth during the near term",
                "Ravi Menon: Rajesh, are there any soft spots that you see in any vertical you're seeing in the pipeline shift towards more efficiency or programs? Rajesh Gopinathan: No, Ravi, we're not seeing much from a vertical perspective. It seems to be fairly balanced across. Nothing at this stage that you can call out at a vertical level. Ravi Menon: We had heard your comments in the press conference, you were talking about how the US will likely be the main driver of growth during the near term. But I thought in Europe, our market share is still fairly small. So, we should have a lot of these gen-two outsourcing contracts come out. So, shouldn't that also be a driver at least medium term once uncertainty settles a bit? : Page 14 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Rajesh Gopinathan: The context of the comment is more in terms of what we're hearing from customers. You're right that traditionally, when Europe has done badly, we've still done much better, reflecting that lower market share and our continuous consolidation in that market. What I was mentioning was that when we look at the kind of commentary that we're hearing from customers, in US, the expectation is that they don't know if there is going to be a recession, when is it going to be, and even if it comes it will be a shallow one. Whereas the expectation in Europe is that it is more of a question of time, and it might be much deeper than what the US customers feel. So, that difference in perception is what we were commenting about, that US is the one where the confidence levels are high, and the demand environment is likely to stay very robust. How it plays out for us, that will depend on how well we are able to capitalize on both sides of that equation. Ravi Menon: One follow up, I think we've talked about the bench quite a bit. We've seen strong hiring well ahead of revenue growth for quite some time now",
                "So, that difference in perception is what we were commenting about, that US is the one where the confidence levels are high, and the demand environment is likely to stay very robust. How it plays out for us, that will depend on how well we are able to capitalize on both sides of that equation. Ravi Menon: One follow up, I think we've talked about the bench quite a bit. We've seen strong hiring well ahead of revenue growth for quite some time now. But we've also seen sub contracting keep going up. While you've been building up capacity and bench, so why is this not really showing up in lower subcontracting, when should we start seeing revenue growth and headcount growth start converging? Rajesh Gopinathan: See, subcontractors are also based on supply disruptions at a local market level. So, that definitely has a role to play. As travel opens up, as more normal talent movement opens up, our opportunity to optimize that will also improve. The aggregate bench versus the one will also play up depending on what our long term view or medium term view on demand is. So, there are both drivers at play. One is immediate capture of demand and the other is the supply disruption in local markets. Ravi Menon: When you said that you should see this normalize, should this also be linked to attrition, when we see attrition start coming off, should we think about you know, the utilization moving up? Rajesh Gopinathan: You're saying that as attrition comes down, will utilization move up? Yes, that's a reasonable assumption. That's our typical operating model also. When in a high demand scenario, we first address demand and attrition is also linked into that. So, we'll see. Moderator: The next question is from the line of Sandeep Shah from Equirus Securities. Please go ahead. : Page 15 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Sandeep Shah: Most of the questions have been answered",
                "That's our typical operating model also. When in a high demand scenario, we first address demand and attrition is also linked into that. So, we'll see. Moderator: The next question is from the line of Sandeep Shah from Equirus Securities. Please go ahead. : Page 15 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) Sandeep Shah: Most of the questions have been answered. Just on Europe, you have discussed about UK. Can you also discuss about continental Europe because in the last four to five quarters, the growth in continental Europe on a sequential basis is a bit lower and softish versus company average. So, one can believe with the macro concerns or this is more specific to TCS as a whole? Rajesh Gopinathan: That we'll have to look across, but we are not seeing any significant change to the demand environment in Europe across the board. 12% is still a decent growth number. As I said, the commentary that we're hearing from clients is a bit more mixed in Europe compared to North America, but at an immediate demand basis, we're not seeing anything that is significant. Sandeep Shah: And just a second question that with increasing macro concerns, some of your global peers are saying clients have also started discussion on outsourcing led deals, which you also highlighted, Rajesh, in your comments where the multi-tower outsourcing deals are increasing in first quarter versus the last four quarters. So, is it fair to say that outsourcing led deals can increase and large caps including you could be a fair beneficiary of this going forward? Rajesh Gopinathan: Our twin engine strategy is essentially predicated on that we have relevance in both scenarios, both consolidation scenario as well as growth scenario",
                "So, is it fair to say that outsourcing led deals can increase and large caps including you could be a fair beneficiary of this going forward? Rajesh Gopinathan: Our twin engine strategy is essentially predicated on that we have relevance in both scenarios, both consolidation scenario as well as growth scenario. So, we do believe that our continuous investment in a broad-based set of services and holistic structure positions us very well to be a single source strategic provider to the client, who can actually use that opportunity to drive the transformation and set themselves up for the growth phase also. And that value is what is driving some of those deals that we have described in this quarter. Sandeep Shah: Last year because of attrition and supply side issue, our margin pull off from the first quarter seasonal decline was lower in the second to fourth quarter. One can say we are largely behind and from 2Q to 4Q of FY'23 our normal seasonality may start where margin pull off on the upward side could be higher versus what we have seen in FY'22. Samir Seksaria: Yes, that is reasonable to expect. We should see an upward trajectory, that's our target. Last year, we saw the full impact of supply side playing out and that's what played out on the margins as well. Moderator: The next question is from the line of Debashish Mazumdar from B&K Securities. Please go ahead. D Mazumdar: Just one small question. I need to understand as far as the margin trajectory is concerned, in a normal year, we see the entry margins are lower and exit margins are normally higher, which was not the case last year; last year, we started with 25.5%, we : Page 16 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) ended with 25%. So, this year, we are starting with 23.1%. So, how confident we are that this 23",
                "D Mazumdar: Just one small question. I need to understand as far as the margin trajectory is concerned, in a normal year, we see the entry margins are lower and exit margins are normally higher, which was not the case last year; last year, we started with 25.5%, we : Page 16 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) ended with 25%. So, this year, we are starting with 23.1%. So, how confident we are that this 23.1% trajectory when you move out from hereon and it is kind of bottomed- out? Samir Seksaria: I just responded the same thing to the previous question as well. Our target is we expect that it would be on an increasing trajectory. And that has been the usual trend except for last year. Our target is that it should improve and we should get closer to what we exited in Q4. D Mazumdar: If we see the competition statements around attrition, the quarterly annualized attrition for other competition has started coming down from Q4 onwards, whereas for us the commentary that is stabling currently. So, just wanted to get some sense and from the salary hikes it seems to be the salary hikes are almost similar to the salary hike that we have given last year, whereas some of our competitions were extremely aggressive in giving salary hikes. So, just wanted to get some sense that in terms of controlling attrition… I'm not asking about the overall number, because it is much lower than peers, but in terms of trajectory of controlling attrition, like we are a little behind the curve as compared to others. Rajesh Gopinathan: Our employee rewards program is much more holistic than period-to-period one. I believe that our employee retention numbers also reflect this holistic employee engagement and total rewards program. So, we are quite confident about where we are. You will recall that even at the height of the pandemic, we were the ones who first said that we will not be laying off anybody",
                "Rajesh Gopinathan: Our employee rewards program is much more holistic than period-to-period one. I believe that our employee retention numbers also reflect this holistic employee engagement and total rewards program. So, we are quite confident about where we are. You will recall that even at the height of the pandemic, we were the ones who first said that we will not be laying off anybody. We honored all offers, we were the first to announce the salary hikes in October of 2020. So, our program is not entirely based on what the short-term impacts are and we're quite confident and comfortable with what we're executing. We believe that these numbers will reflect that. So, as I said, our expectation is that attrition will start tapering off in the next few months being three months, four months, whatever is the time. Moderator: The next question is from the line of Dipesh Mehta from Emkay Global. Please go ahead. Dipesh Mehta: A couple of questions. Starting with regional market, if I look at our regional market performance remain muted for some time and even though BFSI also remains softer, so if you can provide some sense about how we should look at regional market? Second question is about the fresher hiring. We are indicating about 40,000 for FY 23. But considering that we plan one year advance, how many offers are we planning for : Page 17 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) FY 24? And last question is about deal pipeline. Any softness or any uptick we're seeing in deal pipeline build up, if you can comment on that? N G Subramaniam: Regional markets, by definition, is volatile. We group some of our emerging businesses and emerging market businesses into that bucket",
                "Any softness or any uptick we're seeing in deal pipeline build up, if you can comment on that? N G Subramaniam: Regional markets, by definition, is volatile. We group some of our emerging businesses and emerging market businesses into that bucket. But overall, if you look at it, how the Asia Pacific market or functions of the Middle East, Africa market functions, these markets do not provide big opportunities for an annuity-based revenue, there are typically project-based revenues. So, there is always a volatility that is there. The second question was with respect to the fresher hiring of 40,000. As Rajesh mentioned, we are – Dipesh Mehta: No, question was for '24. N G Subramaniam: I think, on average, we always started with 40,000 given our size and shape of our business. But then you really look at the last year as well when we said the 40,000, but then when we went on to hire up to 100,000, our hiring engine was able to do that. So, we have that ability and agility to hire talent and fortunately, we are able to attract the right talent given our brand, given the opportunities that we offer to our employees. Dipesh Mehta: Last question was about deal pipeline built up. Any changes we are seeing there? N G Subramaniam: We are not seeing any major deviation or abnormalities in that. With a book-to-bill ratio of 1.2, it is looking good and the combination of the pipeline that we are pursuing today across the markets is quite similar to what we have experienced now in the previous years and the type of opportunities are also in line with the demand that we see for the type of technologies and a good combination of both cost optimization as well as growth and transformation. It also has several large, medium as well as smaller deals. Good broad-based pipeline that we have, and we are happy with what we have been able to put in the pipeline and then pursuing those opportunities. Dipesh Mehta: Last question is about onsite salary",
                "It also has several large, medium as well as smaller deals. Good broad-based pipeline that we have, and we are happy with what we have been able to put in the pipeline and then pursuing those opportunities. Dipesh Mehta: Last question is about onsite salary. Can you quantify what kind of hike we give for onsite employees this quarter? N G Subramaniam: Average is between 4% and 5% for the onsite salary hikes we have given. But give allowance to the fact that for top performers, the potential earnout has been higher based on their performance. Moderator: The next question is from the line of Manik Taneja from JM Financial. Please go ahead. Manik Taneja: I had a couple of questions. Number one is that you mentioned that some of your senior client executive conversations are suggesting some concerns on the macro economic : Page 18 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) environment. Is that also translating in terms of any potential impact on pricing discussions? Rajesh Gopinathan: As I said earlier, the discussions on macro are not getting reflected in any of the actual project contract or deal pipeline currently. And I just gave you that color in terms of the nature of the competition and how that is different market-to-market. We are not seeing any reflection of that at the actual operating and deal levels. Manik Taneja: I wanted to understand if you would be looking to revisit the onboarding timelines for the 40,000-odd freshers offers that we made for FY'23 if the situation deteriorates. Rajesh Gopinathan: We do not do that. We honor all offers that we make. We do not defer joining or do any form of such management. That's not part of our philosophy. Any offer that we make, we honor it based on whatever is the timeline committed on it. Moderator: The next question is from the line of Apurva Prasad from HDFC Securities. Please go ahead. Apurva Prasad: Rajesh, just a couple of quick ones",
                "Rajesh Gopinathan: We do not do that. We honor all offers that we make. We do not defer joining or do any form of such management. That's not part of our philosophy. Any offer that we make, we honor it based on whatever is the timeline committed on it. Moderator: The next question is from the line of Apurva Prasad from HDFC Securities. Please go ahead. Apurva Prasad: Rajesh, just a couple of quick ones. Most of the others have been covered. So, first one is on pricing. Are you seeing any incremental challenges in getting price increase even though that being selective say versus the previous quarter? And the second question is on Retail and CPG. Are there some early signs of some softness there, and I'm referencing to the book-to-bill has been significantly higher than the past three years, this looks excellent down a lot more. Rajesh Gopinathan: No, on pricing, actually, the conversations are picking up momentum rather than losing momentum. So, absolutely no reflection there. On retail, those TCV numbers are normal nature of TCV rather than anything else. We are seeing very strong transformation agendas across many retailers, especially grocery and essential retailers, where significant transformation programs are getting executed and we see continuing demand. So, overall the pipeline is quite strong and the demand environment also in retail is very strong. Moderator: Ladies and gentlemen, that was the last question for today. I now hand the conference over to the management for the closing comments. Rajesh Gopinathan: Thank you. It has been a good start to the year with 15.5% growth in constant currency, and with all our industry verticals showing good growth. Our order book and pipeline is also very strong, giving us good visibility for the next few months. Our margin dipped this quarter due to salary increase and supply side related costs. But we stay confident in our ability to bring it back to a preferred range over time",
                'Rajesh Gopinathan: Thank you. It has been a good start to the year with 15.5% growth in constant currency, and with all our industry verticals showing good growth. Our order book and pipeline is also very strong, giving us good visibility for the next few months. Our margin dipped this quarter due to salary increase and supply side related costs. But we stay confident in our ability to bring it back to a preferred range over time. : Page 19 of 20 Tata Consultancy Services Q1 & FY23 Earnings Conference Call July 08, 2022, 19:00 pm IST (09:30 hrs US ET) On the people front, we continue to hire across all our markets and added over 14,000 employees on a net basis in Q1. Our attrition continues to be elevated at 19.7% in IT services on an LTM basis. This will probably peak next quarter and will start tapering out. With that we wrap up our call. Thank you all for joining us on this call today. Enjoy the rest of your evening or day and stay safe. Moderator: Thank you members of the management. On behalf of TCS that concludes this conference. Thank you all for joining us and you may now disconnect your lines. Note: This transcript has been edited for readability and does not purport to be a verbatim record of the proceedings. : Page 20 of 20'
            ]
    llm_model= load_llama_model()
    print("llm_model_loaded")
    transcript_themes= get_final_transcript_themes(llm_model,tcs_chunks)
    print("all themes generated")
    print(transcript_themes)
    # overall_doc_summary= get_overall_document_summary(llm_model,tcs_chunks)
    # print("Overall summary generated")
    # print(overall_doc_summary)
    e5_embedding_model = SentenceTransformer('intfloat/e5-large')
    chunk_embedding_pair={}
    for chunk_text in tcs_chunks:
        chunk_embedding= generate_embeddings(e5_embedding_model,chunk_text)
        chunk_embedding_pair[chunk_text]= chunk_embedding
    relevant_chunks_dict= filter_relevant_chunks(e5_embedding_model,transcript_themes,chunk_embedding_pair)
    theme_based_summary= get_document_theme_summary(relevant_chunks_dict,llm_model)
    print("Final theme based summary generated")
    print(theme_based_summary)


main()
