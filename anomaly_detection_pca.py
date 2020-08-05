from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pandas as pd
import json
import os
import re
import time
import pickle
import joblib


def run_pca(train_dir = "data/train_json_data/", inf_dir = "data/inference_json_data/",
webpage_dir = "analysis_tool/webpage/", pca_encoded_dir = "alg_data/pca_encoded_result/"):
    pca = PCA(train_dir,inf_dir,webpage_dir,pca_encoded_dir)
    # pca.encoding_logs(train_dir)
    # pca.encoding_logs(inf_dir)
    model, doc= pca.tf_idf()
    pca_model, inf_df = pca.pca_train(model,doc)
    pca.anomaly_detect(pca_model,inf_df)

class PCA:

    def __init__(self,train_dir,inf_dir,webpage_dir,pca_encoded_dir):
        self.train_json_dir = train_dir
        self.inf_json_dir = inf_dir
        self.webpage_dir = webpage_dir
        self.pca_encoded_dir = pca_encoded_dir

    def encoding_logs(self,directory):
        for file in os.scandir(directory):
            print(file.name)
            if file.name.endswith("-json.log"):   
                t_start = time.time()
                build_number = re.findall(r'\d+',file.name)            
                df_log = self.log_to_matrix(file.name,directory)
                encoded_log = self.encoding_matrix(df_log)
                encoded_log.to_csv(f'{self.pca_encoded_dir}{build_number[0]}encoded_matrix_pca.csv',index=False)
                print(f"Time spent for encoding {build_number[0]} is {time.time()-t_start}")
        print("All log files have been successfully loaded")
        # return df_log
        
    def log_to_matrix(self, filename, directory):
        logdf = pd.DataFrame(columns=["lineId","monotonic_time","priority","transport","content"])
        log_messages = []
        log_timestamp = []
        log_priority = []
        log_transport = []
        linecount = 0 
        lineIds = []
        with open(directory + filename, 'r') as fin:
            for line in fin:
                message = ''
                log_time = ''
                priority = 8 #empty value as a new category / to deal with missing value
                transport = ''
                try:
                    if 'MESSAGE' in line:
                        message = re.sub(r'[^\x00-\x7F]+','<NASCII>',str(json.loads(line)["MESSAGE"]))
                    if '__MONOTONIC_TIMESTAMP' in line:
                        log_time = str(json.loads(line)["__MONOTONIC_TIMESTAMP"])
                    if 'PRIORITY' in line:
                        priority = str(json.loads(line)["PRIORITY"])
                    if '_TRANSPORT' in line:
                        transport = re.sub(r'[^\x00-\x7F]+','<NASCII>',str(json.loads(line)["_TRANSPORT"]))
                except:
                    print(f"Error at reading json file line{linecount}")

                log_messages.append(message)
                log_timestamp.append(log_time)          
                log_priority.append(priority)
                log_transport.append(transport)
                lineIds.append(linecount)
                linecount += 1
        print(f"File {filename} loaded")
        logdf['lineId']=lineIds
        logdf['monotonic_time']=log_timestamp
        logdf['priority']=log_priority
        logdf['transport']=log_transport
        logdf['content']=log_messages

        print(f'log {filename} to dataframe ready, length: {len(logdf)}')
        return logdf
    
    def encoding_matrix(self,dataframe):
        encoded_transport = self.one_hot_encoding(dataframe,'transport')
        encoded_priority = self.encoding_priority(encoded_transport)
        cleaned_data = self.clean_data(encoded_priority)
        encoded_dataframe = encoded_priority.drop('content',axis=1)
        encoded_dataframe = pd.concat([encoded_dataframe,cleaned_data],axis = 1)
        encoded_dataframe.to_csv('encoded_data_test.csv')
        return encoded_dataframe

    def encoding_priority(self,dataframe):
        pd.to_numeric(dataframe['priority'], errors='coerce')
        dataframe.loc[(dataframe['priority'] == 'None'),'priority']= 8
        priority_nan_encoding = dataframe['priority'].fillna(8)
        dataframe = pd.concat([dataframe.drop('priority',axis=1),priority_nan_encoding],axis=1)
        # print(dataframe.isnull().sum())
        # print(dataframe['priority'].value_counts())
        return dataframe

    def one_hot_encoding(self, dataframe, column):
        dataframe = pd.concat([dataframe.drop(column,axis=1),pd.get_dummies(dataframe[column])],axis=1)
        return dataframe

    def clean_data(self,log):
        messages = log['content']
        cleaned_content = []
        cleaned_content_tokens = []
        for messagesLine in messages:
            cleaned_meesage = self.clean_meesages_re(messagesLine)
            cleaned_content.append(cleaned_meesage)
            cleaned_content_tokens.append(cleaned_meesage.split())
        cleaned_content = pd.DataFrame(cleaned_content,columns = ['content'])
        # print("cleaned-content",cleaned_content)
        return cleaned_content

    def clean_meesages_re(self,messageline):
        regex = [
            r'[^\w\s]',
            r'\b[0-9]+\s',
            r'\b([a-z0-9])*[0-9]+.\b',
            r'[\s]+'
            ]
        for currentRex in regex:
            messageline = messageline.lower()
            messageline = re.sub(currentRex,' ',messageline)
        return messageline

    def tf_idf(self):
        document =[]
        data_to_be_transformed = {}
        for file in os.scandir(self.pca_encoded_dir): 
            if file.name.endswith("csv"):   
                number = re.findall(r'\d+',file.name)[0]  
                data_temp = []
                data = pd.read_csv(file)
                content = data['content']
                doc = ''
                for line in content:
                    doc = doc + line
                    data_temp.append(line)
                document.append(doc)
                data_to_be_transformed[number] = data_temp

        vectorizer = TfidfVectorizer(max_features=3000)
        model = vectorizer.fit(document)
        joblib.dump(model, 'vectroizer.pkl')
    
        return model,data_to_be_transformed


    def combine_tfidf_matrix(self,model,data_to_be_transformed):
        i = 0
        for key in data_to_be_transformed.keys():
            i += 1
            data = data_to_be_transformed[key]
            matrix = model.transform(data) 
            matrix_dense = pd.DataFrame(matrix.todense())
            matrix_encoded = pd.read_csv(f'{self.pca_encoded_dir}{key}encoded_matrix_pca.csv')  
            encoded_dataframe = pd.concat([matrix_dense,matrix_encoded.drop(['lineId','content','monotonic_time'],axis=1)],axis=1)
            yield key, encoded_dataframe
            del key
            del encoded_dataframe

    def pca_train(self,model,data):
        #171 is chosen here for its keeping 95% variance of PCA
        ipca = IncrementalPCA(n_components=171, batch_size=3000)
        start_time = time.time()
        chunk_size = 20000
        for file in os.scandir(self.inf_json_dir):
            inf_num = re.findall(r'\d+',file.name)[0]
        for key,df in self.combine_tfidf_matrix(model,data):
            if key != inf_num:
                for i in range(0,df.shape[0],chunk_size):
                    df_chunk = df[i:i+chunk_size]
                    print(df_chunk)
                    ipca.partial_fit(df_chunk)
            elif key == inf_num:
                inf_df = df
        ratio = 0
        #check expalained variance and modify n_components for different datasets
        for num in ipca.explained_variance_ratio_:
            ratio += num
            print(ratio)
        print(f"It takes {time.time()-start_time} to finish the pca training")
        return ipca,inf_df
    
    def anomaly_detect(self,model,inf_df,threshold = 5):
        X_test_ipca = model.fit_transform(inf_df)
        print(X_test_ipca.shape)
        result = []
        pos = []
        is_anomalies = []
        for num in range(X_test_ipca.shape[0]):
            y_a = X_test_ipca[num]
            SPE = np.power(np.linalg.norm(y_a),2)
            if SPE > threshold:
                is_anomalies.append(True)
                pos.append(80000*(num-1) + num)
            else:
                is_anomalies.append(False)
            result.append(1/SPE)
        #print(result)
        fjson = open(f"{self.webpage_dir}PCA_info_{threshold}.json", "w")
        fjson.write(json.dumps((is_anomalies,pos,result)))
        fjson.close()
