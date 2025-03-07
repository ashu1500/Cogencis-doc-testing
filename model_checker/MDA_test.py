import psycopg2
import os
import re
import ast

def create_connection():
    ''' Create connection for the database'''
    try:
        connection = psycopg2.connect(database = "cogencis_db",
                                      user = "cogencis_user",
                                      host= '172.17.10.31',
                                      password = "Cusr123",
                                      port = 5432)
        return connection
    except Exception as e:
       print(e)


def get_required_file_data(file_id):
    ''' Get file_data for the uploaded file'''
    try:
        connection= create_connection()
        current_connection= connection.cursor()
        current_connection.execute("SELECT file_content from file_parsed_data where file_id=%s",(file_id,))
        file_data= current_connection.fetchall()
        current_connection.close()
        return ast.literal_eval(file_data[0][0])
    except Exception as e:
        print(e)
        raise e
    
if __name__=='__main__':
    file_data= get_required_file_data(81)
    print(file_data)