#!/usr/bin/env python
import scipy.io as sio
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
import pickle
from multiprocessing import Pool

np.random.seed(0)

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,        0,          0,          data[0],    0,          data[16],   0,          0,          0       )
    data_2D[1] = (0,        0,          0,          data[1],    0,          data[17],   0,          0,          0       )
    data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
    data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
    data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
    data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
    data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
    data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
    # return shape:9*9
    return data_2D

def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized

def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],9,9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize( data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D

def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size

def segment_signal_without_transition(data,label_aro,label_val,label_index,window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if((len(data[start:end]) == window_size)):
            if(start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels_aro = np.array(label_aro[label_index])
                labels_val = np.array(label_val[label_index])
                labels_aro = np.append(labels_aro, np.array(label_aro[label_index]))
                labels_val = np.append(labels_val, np.array(label_val[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels_aro = np.append(labels_aro, np.array(label_aro[label_index])) # labels = np.append(labels, stats.mode(label[start:end])[0][0])
                labels_val = np.append(labels_val, np.array(label_val[label_index])) # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels_aro, labels_val

def apply_mixup(dataset_file,window_size): # initial empty label arrays
    print("Processing",dataset_file,"..........")
    # data_file_in = sio.loadmat(dataset_file)
    data_file_in = pickle.load(open(dataset_file,"rb"),encoding="latin1")
    data_in = data_file_in["data"].transpose(0,2,1)
    #0 valence, 1 arousal, 2 dominance, 3 liking
    label_aro=1
    label_val=0
    label_in_aro= data_file_in["labels"][:,label_aro]>5
    label_in_val= data_file_in["labels"][:,label_val]>5
    label_inter_aro = np.empty([0]) # initial empty data arrays
    label_inter_val = np.empty([0]) # initial empty data arrays
    data_inter_cnn  = np.empty([0,window_size, 9, 9])
    data_inter_rnn  = np.empty([0, window_size, 32])
    trials = data_in.shape[0]

    # Data pre-processing
    for trial in range(0,trials):
        print('\r',trial*100//trials,'%',end='')
        base_signal = (data_in[trial,0:128,0:32]+data_in[trial,128:256,0:32]+data_in[trial,256:384,0:32])/3
        data = data_in[trial,384:8064,0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0,60):
            data[i*128:(i+1)*128,0:32]=data[i*128:(i+1)*128,0:32]-base_signal
        label_index = trial
        #read data and label
        data = norm_dataset(data)
        data, label_aro, label_val = segment_signal_without_transition(data, label_in_aro,label_in_val,label_index,window_size)
        # cnn data process
        data_cnn    = dataset_1Dto2D(data)
        data_cnn    = data_cnn.reshape ( int(data_cnn.shape[0]/window_size), window_size, 9, 9)
        # rnn data process
        data_rnn    = data. reshape(int(data.shape[0]/window_size), window_size, 32)
        # append new data and label
        data_inter_cnn  = np.vstack([data_inter_cnn, data_cnn])
        data_inter_rnn  = np.vstack([data_inter_rnn, data_rnn])
        label_inter_aro = np.append(label_inter_aro, label_aro)
        label_inter_val = np.append(label_inter_val, label_val)
    '''
    print("total cnn size:", data_inter_cnn.shape)
    print("total rnn size:", data_inter_rnn.shape)
    print("total label size:", label_inter.shape)
    '''
    # shuffle data
    print('\r100 %')
    index = np.array(range(0, len(label_inter_aro)))
    np.random.shuffle(index)
    shuffled_data_cnn   = data_inter_cnn[index]
    shuffled_data_rnn   = data_inter_rnn[index]
    shuffled_label_aro  = label_inter_aro[index]
    shuffled_label_val  = label_inter_val[index]
    return shuffled_data_cnn ,shuffled_data_rnn,shuffled_label_aro,shuffled_label_val

PROCESSES=4
def multi_pre_process_all():
    MUL=32//PROCESSES
    Pool().starmap(pre_process_range,[(i*MUL+1,(i+1)*MUL+1) for i in range(PROCESSES)])

def pre_process_range(start,end):
    begin = time.time()
    dataset_dir     =   "../dataset_emotion_nn/"
    window_size     =   128
    output_dir      =   "./pre_processed_data/"
    if os.path.isdir(output_dir)==False:
        os.makedirs(output_dir)
    # print(record_list)
    for i in range(start,end):
        if i<10:
            file_name='s'+str(0)+str(i)
        else:
            file_name='s'+str(i)
        print("[+] Reading subject:",file_name)
        shuffled_cnn_data,shuffled_rnn_data,shuffled_label_aro,shuffled_label_val = apply_mixup(dataset_dir+file_name+".dat", window_size)

        pred_aro=np.zeros((shuffled_label_aro.shape[0],2))        
        pred_val=np.zeros((shuffled_label_val.shape[0],2))        
        for lbl_ind in range(shuffled_label_aro.shape[0]):
            pred_aro[lbl_ind][int(shuffled_label_aro[lbl_ind])]=1
            pred_val[lbl_ind][int(shuffled_label_val[lbl_ind])]=1

        output_data_cnn = output_dir+file_name+"_cnn.pkl"
        output_data_rnn = output_dir+file_name+"_rnn.pkl"
        output_label_aro= output_dir+file_name+"_aro_labels.pkl"
        output_label_val= output_dir+file_name+"_val_labels.pkl"

        print("[+] Writing processed:",file_name)
        with open(output_data_cnn, "wb") as fp:
            pickle.dump( shuffled_cnn_data,fp, protocol=4)

        with open( output_data_rnn, "wb") as fp:
            pickle.dump(shuffled_rnn_data, fp, protocol=4)

        with open(output_label_aro, "wb") as fp:
            pickle.dump(pred_aro, fp)

        with open(output_label_val, "wb") as fp:
            pickle.dump(pred_val, fp)
        end = time.time()
        print("[*] Time consumed:",(end-begin))

PROCESSED_DIR = "./pre_processed_data/"
def get_subject_data(subject_no, aro_or_val):
    if subject_no<10:
        file_name='s'+str(0)+str(subject_no)
    else:
        file_name='s'+str(subject_no)
    file_dir = PROCESSED_DIR+file_name
    if not os.path.exists(file_dir+"_val_labels.pkl"):
        pre_process_range(subject_no,subject_no+1)
    eeg_1D = pickle.load(open(file_dir+"_rnn.pkl", 'rb'))   #(40*60, 128, 32)
    eeg_2D = pickle.load(open(file_dir+"_cnn.pkl", 'rb'))   #(40*60, 128, 9, 9)
    #Fetching valance and arousal
    labels=pickle.load(open(file_dir+"_"+aro_or_val+"_labels.pkl", 'rb'))  #(2400,2)

    return eeg_1D, eeg_2D, labels

if __name__ == '__main__':
    multi_pre_process_all()