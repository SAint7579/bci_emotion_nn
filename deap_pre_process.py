import _pickle as pickle
import numpy as np
from os import path
from multiprocessing import Pool
from time import time

THREADS=4
DATASET_DIR ="../dataset_emotion_nn/"  #Default dataset directory
PROCESSED_DIR = "pre_processed_data/"

# smaple rate 128 Hz
sample_rate=128
    
def data_1Dto2D(data, Y=9, X=9):
    '''
        Converts one dataset of channel readings form 1d to 2d (9*9)
    '''
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          data[0],    0,          data[16], 	0,  	    0, 	        0       )
    data_2D[1] = (0,  	   	0,          0,          data[1],    0,          data[17],   0,          0,          0       )
    data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
    data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
    data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
    data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
    data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
    data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
    # return shape:9*9
    return data_2D

def dataset_1Dto2D(dataset):
    '''
        Converts the entire dataset for 1D to 2D
    '''
    dataset_2D = np.zeros([dataset.shape[0],9,9])
    for i in range(dataset.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset[i])
    # return shape: m*9*9
    return dataset_2D

def feature_normalize(data):
    '''
        Normalize the EEG data
    '''
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized

def norm_1D_dataset(dataset_1D):
    '''
        Normalizes 1D dataset
    '''
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D

def norm_2D_dataset(dataset_1D):
    '''
        Normalizes 2D dataset
    '''
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D

def pre_process_range(directory=DATASET_DIR,start_range=1,end_range=33):
    for i in range(start_range,end_range):
        if i<10:
            file_name='s'+str(0)+str(i)
        else:
            file_name='s'+str(i)
        print("[+] Reading:",file_name)
        temp=pickle.load(open(directory+file_name+'.dat','rb'),encoding='latin1')

        temp['labels'] = (temp['labels'][:,:2]>5)*1
        print("[*] Processing:",file_name)
        
        N = (sample_rate*3)
        data=temp['data']
        base_data=data[:,:32,:N]
        base_mean=base_data.sum(axis=2)/N
        data=data[:,:32,N:]
        data=data-base_mean[:,:,None]      #(40, 32, 7680)
        eeg_1D=[]
        eeg_2D=[]
        i=0
        for media in data:
            print('\r',i*100//data.shape[0],'%',end='')
            i+=1
            eeg = media.transpose()
            eeg_1D.append(norm_1D_dataset(eeg))
            eeg_2D.append(norm_2D_dataset(eeg))

        print('\r100 %')
        eeg_1D=np.array(eeg_1D)
        eeg_2D=np.array(eeg_2D)

        print("[+] Writing processed:",file_name)
        with open(PROCESSED_DIR+file_name+"_cnn.pkl","wb") as f:
            pickle.dump(eeg_2D,f,protocol=4)
        with open(PROCESSED_DIR+file_name+"_rnn.pkl","wb") as f:
            pickle.dump(eeg_1D,f,protocol=4)
        with open(PROCESSED_DIR+file_name+"_labels.pkl","wb") as f:
            pickle.dump(temp['labels'],f)
        print("[*] Done.\n")

def multi_pre_process_all():
    MUL=32//THREADS
    Pool().starmap(pre_process_range,[(DATASET_DIR,i*MUL+1,(i+1)*MUL+1) for i in range(THREADS)])

def get_subject_data(subject_no,media_no):
    '''
        Gets the data for a preticular subject
        ARGS:
        subject_no: ID of subject
        media_no: The meida for which the data was shown
        RETURN VALS:
        eeg data 1D (normalized)
        eeg data 2D (normalized)
        valance value
        arousal value
    '''
    # For the subject selection (1-32)
    if subject_no<10:
        file_name='s'+str(0)+str(subject_no)
    else:
        file_name='s'+str(subject_no)
    print("[+] Reading subject:",file_name,"Media:",media_no)
    #Importing the file
    file_dir = PROCESSED_DIR+file_name
    #Fetching the required data
    if not path.exists(file_dir+"_labels.pkl"):
        pre_process_range(DATASET_DIR,subject_no,subject_no+1)
    else:
        pass

    eeg_1D_normalized = pickle.load(open(file_dir+"_rnn.pkl", 'rb'))[media_no]
    eeg_2D_normalized = pickle.load(open(file_dir+"_cnn.pkl", 'rb'))[media_no]
    #Fetching valance and arousal
    labels=pickle.load(open(file_dir+"_labels.pkl", 'rb'))
    valance = labels[media_no][0]
    arousal = labels[media_no][1]
    #Returning the dataset
    return eeg_1D_normalized,eeg_2D_normalized,valance,arousal

if __name__ == '__main__':
    try:
        t=time()
        multi_pre_process_all()
        print("[*] Time Taken:",(time()-t))
    except KeyboardInterrupt:
        print("\n[!] Exitting ...")