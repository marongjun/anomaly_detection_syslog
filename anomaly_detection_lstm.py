import os
import time
import glob
import socket
import numpy as np
import pandas as pd 
import tensorflow as tf
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def batch_generator(x, y, batch_size = 512):
    '''
    Generate batch for traninng, trasfer sparse matrix to dense input
    params:
    batch_size: depends on RAM, 512 as default
    '''
    number_of_batches = x.shape[0]//batch_size
    counter = 0
    shuffle_index = np.arange(x.shape[0])
    np.random.shuffle(shuffle_index)
    x = x[shuffle_index, :]
    y = y[shuffle_index, :]
    # print(x[1:10])
    while 1:
        index_batch = shuffle_index[batch_size*counter: batch_size*(counter+1)]
        x_batch = x[index_batch, :].todense()
        y_batch = y[index_batch, :].todense()
        counter += 1
        x_batch = np.array(x_batch).reshape((x_batch.shape[0], 1 ,x_batch.shape[1]))
        # print(x_batch[1:10])
        yield(x_batch, np.array(y_batch))
        del x_batch
        del y_batch
        if counter >= number_of_batches:
            np.random.shuffle(shuffle_index)
            counter = 0

class DynamicModeling:
    def __init__(self,train_dir ,inf_dir, webpage_dir):
        '''
        Model: to save initialized or trained model 
        m_max: Max Event Id till now
        '''
        self.history = {
            'loss': [], 'categorical_accuracy': [],'val_loss': [], 'val_categorical_accuracy': [], 
            'top_5_categorical_accuracy': [], 'top_10_categorical_accuracy':[],'top_15_categorical_accuracy':[],
            'val_top_5_categorical_accuracy': [], 'val_top_10_categorical_accuracy':[],'val_top_15_categorical_accuracy':[]
        }
        self.model = Sequential()
        self.m_max = 0
        self.train_dir =  f"{train_dir}/build*-json.log_structured.csv"
        self.inf_dir = f"{inf_dir}/build*-json.log_structured.csv"
        self.webpage_dir = webpage_dir
        train_files = glob.glob(self.train_dir)
        inference_files = glob.glob(self.inf_dir)
        files = train_files + inference_files
        # Find max event id in current train + inf data set 
        dense_num = self.find_max_eventId(files)
        self.initialize_model(dense_num)
    
    def initialize_model(self,dense_num):
        # Initializing model whenever max id changes 
        self.model.add(LSTM(256, input_shape = (1,dense_num*10), 
        recurrent_initializer='orthogonal', return_sequences=True))
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(Dropout(0.2))
#         self.model.add(LSTM(256,return_sequences=True))
#         self.model.add(Dropout(0.2))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(dense_num, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_categorical_accuracy"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="top_10_categorical_accuracy"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=15, name="top_15_categorical_accuracy"),
                 tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
                ])

    def find_max_eventId(self,files):
        #search for max eventId till when the model is called 
        start = time.time()
        for m_file in files:
            with open(m_file) as f:
                df = pd.read_csv(f)
                event1 = df['EventId']
                if event1.max() > self.m_max:
                    self.m_max = event1.max()
                del event1
                del df
        print(f"Find max id in {time.time()-start}s")
        print("MAX Event Id is", self.m_max)
        dense_num = self.m_max
        return dense_num

    def modeling(self, train_data, train_label):  
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_label,
        test_size = 0.2, random_state = 3)
        t_start = time.time()
        batch_size = 1024
        times = int(train_data.shape[0] / batch_size)+ 1 
        # load the model
        filepath = "checkpoint.h5"  
        if os.path.isfile(f'./{filepath}'):
            model = load_model(filepath)
        else:
            model = self.model
        # define the checkpoint
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        #model.summary()
        history = model.fit_generator(batch_generator(X_train, y_train, 
        batch_size),
                    steps_per_epoch=times,
                    verbose=1,
                    epochs=10,
                    validation_data=batch_generator(X_test, y_test,
                    batch_size),
                    validation_steps=1,
                    callbacks=callbacks_list
                    )
        
        model.save('checkpoint.h5')
        print(f"It takes {time.time() - t_start}s for modeling")
        return history

    def plot_history(self, history):
        m_dic = history.history
        print(m_dic)
        for ele in m_dic:
            self.history[ele].extend(m_dic[ele])
        print(self.history)
        # to save accuracy traning history 
        # with open('history.txt', 'w') as file: 
        #     file.write(json.dumps(str(self.history),ensure_ascii=False).encode('utf8').decode())    

    def predict(self, test_data, inf_label,batch_size = 2000):
        length = test_data.shape[0]
        iter_times = length // batch_size + 1
        count = 1
        model = load_model('checkpoint.h5')
        anomalies = []
        prediction_array = []
        array_01 = [0] * 10
        while count <= iter_times: 
            x_batch = test_data[batch_size*(count -1) : batch_size*count].todense()
            inf_label_batch = inf_label[batch_size*(count -1) : batch_size*count]
            print(x_batch.shape)
            x_batch = np.array(x_batch).reshape((x_batch.shape[0], 1, x_batch.shape[1]))
            preds = model.predict(x_batch,verbose=2)
            #preds = self.model.predict(x_batch,verbose=2)
            arr, anomoly_position, pre = self.detect_anomoly(preds, inf_label_batch,count, batch_size)
            prediction_array.extend(pre.numpy().astype(int)) 
            anomalies.extend(anomoly_position)
            array_01.extend(arr)
            count += 1
        arr_first_10_rows = [np.array([-1]*10)]*10
        df_predictions = pd.DataFrame(prediction_array)
        data_predictions = df_predictions.to_numpy().tolist()
        new_predictions = arr_first_10_rows + data_predictions

        with open(self.webpage_dir + "LSTM_event_ids_top_10.bin", "wb") as f1:
            np.array(new_predictions).tofile(f1)
        return anomalies,array_01
    
    def detect_anomoly(self, preds, inf_label,count, batch_size):
        #find anomaly based on top k predictions, k = 15 as default value
        anomaly = tf.nn.top_k(preds,15)
        # pred_array = tf.map_fn(lambda x: x,anomaly[1])
        pred_array = anomaly[1]
        arr = []
        for idx, ele in np.ndenumerate(inf_label):
            if ele-1 in pred_array[idx[0]]:             
                arr.append(0)
            else:
                arr.append(1)
        arr = np.array(arr)
        print(f"Find {(arr == 1).sum()} anomalies out of {inf_label.shape[0]}")
        anomoly_position = list(map(lambda x: x + (count-1)*batch_size + 10, np.where(arr == 1)[0]))
        #print(anomoly_position)
        return arr, anomoly_position, pred_array
    
    def init_onehot_encoder(self):
        x_enc = OneHotEncoder(categories = [[x for x in range(1, 
                        self.m_max + 1)] for i in range(10)], handle_unknown = 'ignore')
        y_enc = OneHotEncoder(categories = [[x for x in range(1, 
                        self.m_max + 1)]], handle_unknown = 'ignore')
        return x_enc, y_enc

    def raw_data_to_one_hot(self,path,act):
        x_enc, y_enc = self.init_onehot_encoder()
        with open(path) as csvfile:
                df = pd.read_csv(csvfile)
                event = df['EventId']
        arr = event.array
        num = len(arr)
        data = []
        label = []
        for i in range(10,num):
            data.append(arr[i-10:i])
            label.append(arr[i])
        label = np.array(label).reshape(len(label),1)          
        data_one_hot = x_enc.fit_transform(data)
        if act == 'train':
            label_one_hot = y_enc.fit_transform(label)
            return data_one_hot, label_one_hot
        elif act == 'predict':
            return data_one_hot, label
    
    def interact_with_client(self,act,path):
        start = time.time()
        X_train, y_train = self.raw_data_to_one_hot(path,act)  
        msg = f"train data processed in {time.time() - start}\n-----{act}ing start------"
        print(msg)
        start = time.time()
        if act == 'train':
            history = self.modeling(X_train, y_train)
            self.plot_history(history)
        elif act == 'predict':
            self.predict(X_train, y_train)
        msg = f"-----training end------\n{act} finishes in {time.time() - start}"
        print(msg)

        # summarize history for accuracy  
        # ploting the accuracy and loss curve
        plt.plot(self.history['categorical_accuracy'])  
        plt.plot(self.history['val_categorical_accuracy'])  
        plt.title('model accuracy')  
        plt.ylabel('accuracy')  
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')  
        plt.savefig('./accuracyVSepoch.png')
        plt.clf()
        # summarize history for loss  
        
        plt.plot(self.history['loss'])  
        plt.plot(self.history['val_loss'])  
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'test'], loc='upper left')  
        plt.savefig('./lossVSepoch.png')
        plt.clf()      



def run_lstm(train_dir = "data/spell_train_result/",
inf_dir = "data/spell_inference_result/", webpage_dir = "analysis_tool/webpage/"):
    dym = DynamicModeling(train_dir ,inf_dir, webpage_dir)
    for file in os.scandir(train_dir):
        if file.name.endswith('-json.log_structured.csv'):   
            dym.interact_with_client('train',f'{train_dir}{file.name}')
            print(file.name)
    for file in os.scandir(inf_dir):
        dym.interact_with_client('predict',f'{inf_dir}{file.name}')
    