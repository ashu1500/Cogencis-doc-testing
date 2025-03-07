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
    
def check_title_headers(json_data):
   ''' Check title headers for MDA'''
   try:
      mda_index=[]
      next_index_range=[]
      for i in range(len(json_data)):
         if 'title' in list(json_data[i].keys()):
            title=json_data[i]['title'].lower()
            if list(filter(re.compile(".*management discussion").match,[title])) and not list(filter(re.compile(".*corporate governance").match,[title])):
               mda_index.append(i)
            elif list(filter(re.compile(".*discussion and analysis").match,[title])):
               mda_index.append(i)
            if list(filter(re.compile(".*corporate governance").match,[title])):
               next_index_range.append(i)
            if list(filter(re.compile(".*governance report").match,[title])):
               next_index_range.append(i)
            if list(filter(re.compile(".*board's report").match,[title])):
               next_index_range.append(i)
            if list(filter(re.compile(".*corporate social responsibility").match,[title])):
               next_index_range.append(i)
            if list(filter(re.compile(".*business responsibility report").match,[title])):
               next_index_range.append(i)
         if list(filter(re.compile(".*report on corporate social responsibility").match,[list(json_data[i].values())[1].lower()])):
            next_index_range.append(i)
      return mda_index,next_index_range
   except Exception as ex:
      print(f"Error in check title headers. {ex.args}")
      raise ex

def get_exact_mda_index(mda_index,json_data):
   ''' Get final MDA index'''
   try:
      mda_index_dict={}
      for x in mda_index:
         mda_index_dict[x]= list(json_data[x].values())[1]

      seen_values= set()
      output_dict = {k: v for k, v in mda_index_dict.items() if not (v in seen_values or seen_values.add(v))}
      mda_result= list(output_dict.keys())
      number_pattern = re.compile(r'\d')
      if len(mda_result)>1:
         for i in mda_result:
            if number_pattern.search(list(json_data[i].values())[1].lower()):
               continue
            if list(filter(re.compile(".*(contd)").match,[list(json_data[i].values())[1].lower()])):
               continue
            if list(filter(re.compile(".*management discussion").match,[list(json_data[i].values())[1].lower()])) and not (list(filter(re.compile(".*management discussion and analysis report").match,[list(json_data[i].values())[1].lower()])) or list(filter(re.compile(".*management discussion & analysis report").match,[list(json_data[i].values())[1].lower()]))):
               final_mda_index=i
               break
            elif list(filter(re.compile(".*mana discussion").match,[list(json_data[i].values())[1].lower()])) and not (list(filter(re.compile(".*management discussion and analysis report").match,[list(json_data[i].values())[1].lower()])) or list(filter(re.compile(".*management discussion & analysis report").match,[list(json_data[i].values())[1].lower()]))):
               final_mda_index=i
               break
            elif list(filter(re.compile(".*management’s discussion").match,[list(json_data[i].values())[1].lower()])) and not list(filter(re.compile(".*management’s discussion and analysis report").match,[list(json_data[i].values())[1].lower()])):
               final_mda_index=i
            elif (list(filter(re.compile(".*management discussion and analysis report").match,[list(json_data[i].values())[1].lower()])) or list(filter(re.compile(".*management discussion & analysis report").match,[list(json_data[i].values())[1].lower()]))):
               final_mda_index=i

      else:
         final_mda_index= mda_result[0]
      return final_mda_index

   except Exception as ex:
      print(f"Error in get exact mda index. {ex.args}")
      raise ex
    

def get_final_mda_data(json_data):
   ''' Get final MDA index range'''
   try:
      mda_index,next_index= check_title_headers(json_data)
        
      final_mda= get_exact_mda_index(mda_index,json_data)
      index_list = list(filter(lambda x: x>final_mda, next_index))
      if len(index_list)>1:
         if list(filter(re.compile(".*corporate governance").match,[list(json_data[index_list[0]].values())[1].lower()])) and not list(filter(re.compile(".*report").match,[list(json_data[index_list[0]].values())[1].lower()])):
            last_index= index_list[1]
         else:
            last_index= index_list[0]
      else:
         last_index= index_list[0]

      return json_data[final_mda:last_index]

   except Exception as ex:
      print(f"Error in get final mda data. {ex.args}")
      raise ex


if __name__=='__main__':
    file_data= get_required_file_data(81)
    print(file_data)
    mda_data= get_final_mda_data(file_data)
    print("MDA DATA")
    print(mda_data)