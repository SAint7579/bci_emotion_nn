import _pickle as pickle
import numpy as np

DATASET_DIR ="./data_preprocessed_python/"  #Default dataset directory

class Dataset():
    def __init__(self,directory=DATASET_DIR):
        '''
        Default location for .dat files: ./data_preprocessed_python/
        '''
        # For the subject selection (1-32)
        self.subject = 0
        # For selecting the media (1-40)
        self.media = 0
        # For file directory
        self.dir = directory
        
    def data_1Dto2D(self,data, Y=9, X=9):
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
    
    def dataset_1Dto2D(self, dataset):
        '''
            Converts the entire dataset for 1D to 2D
        '''
        dataset_2D = np.zeros([dataset.shape[0],9,9])
        for i in range(dataset.shape[0]):
            dataset_2D[i] = self.data_1Dto2D(dataset[i])
        # return shape: m*9*9
        return dataset_2D
    
    def feature_normalize(self, data):
        '''
            Normalize the EEG data
        '''
        mean = data[data.nonzero()].mean()
        sigma = data[data.nonzero()].std()
        data_normalized = data
        data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
        # return shape: 9*9
        return data_normalized
    
    def norm_1D_dataset(self, dataset_1D):
        '''
            Normalizes 1D dataset
        '''
        norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
        for i in range(dataset_1D.shape[0]):
            norm_dataset_1D[i] = self.feature_normalize(dataset_1D[i])
        # return shape: m*32
        return norm_dataset_1D
    
    def norm_2D_dataset(self, dataset_1D):
        '''
            Normalizes 2D dataset
        '''
        norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
        for i in range(dataset_1D.shape[0]):
            norm_dataset_2D[i] = self.feature_normalize( self.data_1Dto2D(dataset_1D[i]))
        # return shape: m*9*9
        return norm_dataset_2D
            
    def get_subject_data(self,subject_no,media_no):
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
        self.subject = subject_no
        if self.subject<10:
            self.subject = '0'+str(self.subject)
        else:
            self.subject = str(self.subject)
        # For selecting the media (1-40)
        self.media = media_no
        #Importing the file
        file_dir = self.dir + "s"+self.subject+'.dat'
        data = pickle.load(open(file_dir, 'rb'),encoding='latin1')
        #Fetching the required data
        eeg_time_data = data['data'][self.media]
        #Removing unnecessary channels
        eeg = eeg_time_data[:32].transpose()
        eeg_2D_normalized = self.norm_2D_dataset(eeg)
        eeg_1D_normalized = self.norm_1D_dataset(eeg)
        #Fetching valance and arousal
        valance = data['labels'][self.media][0]
        arousal = data['labels'][self.media][1]
        #Returning the dataset
        return eeg_1D_normalized,eeg_2D_normalized,valance,arousal