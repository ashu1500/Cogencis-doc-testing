import logging
import os
import deepdoctection as dd
import psycopg2
import json

def get_page_metadata(input_pdf_path):
    ''' Get metadata of each page of pdf'''
    try:
        analyzer= dd.get_dd_analyzer()
        df= analyzer.analyze(path=input_pdf_path)
        df.reset_state()
        chunk_data=[]
        for page in df:
            chunk_data.append(page.chunks)
        return chunk_data
    except Exception as e:
        logging.error(e)

def get_details_from_page_chunk(page_metadata):
    ''' Extract title and text data from page chunks'''
    try:
        chunk_list=[]
        for data in page_metadata:
            page_data={}
            page_metadata_list= list(map(lambda x: str(x),data))
            if 'LayoutType.title' in page_metadata_list:
                page_data["title"]= page_metadata_list[(page_metadata_list.index('LayoutType.title'))+1]
            elif 'LayoutType.text' in page_metadata_list:
                page_data["text"]= page_metadata_list[(page_metadata_list.index('LayoutType.text'))+1]
            elif 'LayoutType.list' in page_metadata_list:
                page_data["text"]= page_metadata_list[(page_metadata_list.index('LayoutType.list'))+1]
            elif 'LayoutType.title' in page_metadata_list:
                page_data["text"]= page_metadata_list[(page_metadata_list.index('LayoutType.line'))+1]
            chunk_list.append(page_data)
        return chunk_list
    
    except Exception as e:
        logging.error(e)

def get_required_details(input_pdf):
    ''' Get the required data format from the extracted metadata'''
    try:
        doc_chunks= get_page_metadata(input_pdf)
        doc_chunk_list=[]
        for page_chunk in doc_chunks:
            doc_chunk_list.append(get_details_from_page_chunk(page_chunk))
        return doc_chunk_list

    except Exception as e:
        logging.error(e)

def combine_chunk_data(doc_chunks):
    '''Combine entire pdf chunk data together'''
    try:
        # document_chunks= get_required_details(input_pdf_path)
        combined_chunks=[]
        for chunk in doc_chunks:
            combined_chunks+= chunk
        return combined_chunks
    
    except Exception as e:
        logging.error(e)

def get_title_list(combined_element_list):
    ''' Get index of all title contents'''
    try:
        title_list=[]
        for element in range(len(combined_element_list)):
            if 'title' in list(combined_element_list[element].keys()):
                title_list.append(element)
        if title_list[-1]< len(combined_element_list)-1:
            title_list.append(len(combined_element_list)-1)
        return title_list
    
    except Exception as e:
        logging.error(e)

def generate_combined_chunks(combined_chunk,title_list):
    ''' Generate combined chunks based on title content'''
    try:
        new_chunk_list=[]
        for x in range(len(title_list)-1):
            initial_chunk= combined_chunk[title_list[x]:title_list[x+1]]
            values_list=[list(x.values())[0] for x in initial_chunk]
            chunk_element= ".".join(values_list)
            new_chunk_list.append(chunk_element)
        return new_chunk_list
    except Exception as e:
        logging.error(e)

def generate_exact_chunks(document_chunks):
    ''' Check for token limit of 2000 and generate chunks'''
    try:
        combined_chunk= combine_chunk_data(document_chunks)
        title_list= get_title_list(combined_chunk)
        new_chunk_list= generate_combined_chunks(combined_chunk,title_list)
        actual_chunk_list=[]
        for chunk in new_chunk_list:
            if len(chunk)>2000:
                actual_chunk_list.append(chunk[:2000])
                actual_chunk_list.insert(new_chunk_list.index(chunk)+1,chunk[2000:])
            else:
                actual_chunk_list.append(chunk)
        return actual_chunk_list
    except Exception as e:
        logging.error(e)

def create_connection():
    ''' Create connection for the database'''
    try:
        connection = psycopg2.connect(database = 'cogencis_db',
                                      user = 'postgres',
                                      host = '3.238.139.215',
                                      password = 'Cadmin123$',
                                      port = 5432) 
        return connection
    except Exception as e:
        logging.error(e)

def update_file_content(file_id,file_name,file_content):
    '''Upate file content in the database'''
    try:
        connection= create_connection()
        current_connection= connection.cursor()
        data_format="text"
        current_connection.execute("INSERT INTO file_parsed_data (file_id, file_name, data_format,file_content) VALUES(%s,%s,%s,%s)", (file_id,file_name,data_format,file_content));
        connection.commit()
        connection.close()
    except Exception as e:
        logging.error(e)

def main():
    input_pdf_path= os.path.join("annual_reports","HDFC Bank Ltd--AR--2019-2020.pdf")
    document_chunks= get_required_details(input_pdf_path)
    combined_chunks= combine_chunk_data(document_chunks)
    json_data= json.dumps(combined_chunks)
    # doc_chunk_list= generate_exact_chunks(document_chunks)
    logging.info("document processed")
    file_id=3
    file_name="HDFC-BANK-AR-2019-20"
    update_file_content(file_id,file_name,str(json_data))
    logging.info("database updated")

main()