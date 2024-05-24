from pathlib import Path
import torch
import os
class DataConfig:
    def __init__(self) -> None:
        # SFEW STARTS    

        source = 'sfew'
        self.cuda = torch.cuda.is_available()
        
        self.IN_COLAB = False
        if os.path.exists('/content'): # check for google colab
            self.IN_COLAB = True

        self.GDRIVE_FOLDER = '/content/gdrive/MyDrive/CV_FER'


        if self.IN_COLAB:
            # self.SFEW_BASE_PATH = Path(self.GDRIVE_FOLDER,'dataset') # for gdrive
            self.GDRIVE_SFEW_FILE_PATH = Path(self.GDRIVE_FOLDER,'dataset','SFEW_2.zip')

        else: 
            self.GDRIVE_SFEW_FILE_PATH = Path('dataset_2','SFEW_2.zip')
    
        self.SFEW_BASE_PATH = 'dataset' # for local
        self.SFEW_EXTRACT_PATH = Path(self.SFEW_BASE_PATH,"sfew")
        self.SFEW_ZIP_FILE_PATH = Path(self.SFEW_BASE_PATH,'SFEW_2.zip')
        self.SFEW_mean_ds = [0.2197, 0.1858, 0.1569]
        self.SFEW_std_dev_ds = [0.1810, 0.1635, 0.1511]
        if self.IN_COLAB:
            self.SFEW_BATCH_SIZE = 16
        else:
            self.SFEW_BATCH_SIZE = 4
        self.SFEW_TRANSFORMS = None #'A' # None or 'A'
        
        # BALANCED SFEW STARTS

        if self.IN_COLAB:
            self.GDRIVE_SFEW_BALANCED_FILE_PATH = Path(self.GDRIVE_FOLDER,'dataset','Balanced_SFEW.zip')
        else: 
            self.GDRIVE_SFEW_BALANCED_FILE_PATH = Path('dataset_2','Balanced_SFEW.zip')

        self.SFEW_BALANCED_BASE_PATH = 'dataset' # for local
        
        self.SFEW_BALANCED_EXTRACT_PATH = Path(self.SFEW_BALANCED_BASE_PATH,"balanced_SFEW")
        self.SFEW_BALANCED_DATA_PATH = Path(self.SFEW_BALANCED_BASE_PATH,"balanced_SFEW", "Balanced_SFEW")

        if self.IN_COLAB:
            self.SFEW_BALANCED_BATCH_SIZE = 16
        else:
            self.SFEW_BALANCED_BATCH_SIZE = 4
        # self.SFEW_BALANCED_TRANSFORMS = None #'A' # None or 'A'


        # EXPW STARTS    

        self.EXPW_LINK = "https://www.kaggle.com/datasets/mohammedaaltaha/expwds"
        
        if self.IN_COLAB:
            # self.EXPW_BASE_PATH = Path(self.GDRIVE_FOLDER,'dataset') # for gdrive
            # self.PICKLE_IMAGE_LIST_PATH = Path(self.GDRIVE_FOLDER,'dataset','expw_raceds_image_list.pkl') # for gdrive
            self.PICKLE_LIST_DICT_PATH = Path(self.GDRIVE_FOLDER,'dataset','expw_raceds_image_dict.pkl') # for gdrive
            self.GDRIVE_EXPW_FILE_PATH = Path(self.GDRIVE_FOLDER,'dataset','expwds','expwds.zip')
            self.DUPLICATE_LIST_PATH = Path(self.GDRIVE_FOLDER,'dataset','duplicate_image_list.pkl') # for gdrive
            self.RACEDS_CROP_LIST_PATH = Path(self.GDRIVE_FOLDER,'dataset','expw_raceds_crop_image_dict.pkl') # for gdrive, racial cropped dataset list
            self.RACEDS_CROP_PATH = Path(self.GDRIVE_FOLDER,'dataset','RaceTest_EXPWCrop') # for local, racial cropped dataset

        else:   
            self.GDRIVE_EXPW_FILE_PATH = Path('dataset_2','expwds','expwds.zip') # for local
            # self.PICKLE_IMAGE_LIST_PATH = Path('dataset','expw_raceds_image_list.pkl') # for local
            self.PICKLE_LIST_DICT_PATH = Path('dataset','expw_raceds_image_dict.pkl') # for local
            self.DUPLICATE_LIST_PATH = Path('dataset','duplicate_image_list.pkl') # for local
            self.RACEDS_CROP_LIST_PATH = Path('dataset','expw_raceds_crop_image_dict.pkl') # for local, racial cropped dataset list
            self.RACEDS_CROP_PATH = Path('dataset','RaceTest_EXPWCrop') # for local, racial cropped dataset
        
        self.EXPW_BASE_PATH = 'dataset' # for local
        self.EXPW_EXTRACT_PATH = Path(self.EXPW_BASE_PATH,"expwds")
        self.EXPW_ZIP_FILE_PATH = Path(self.EXPW_EXTRACT_PATH,'expwds.zip')
        self.EXPW_DATA_PATH = Path(self.EXPW_EXTRACT_PATH,'origin')
        self.EXPW_LABEL_PATH = Path(self.EXPW_EXTRACT_PATH,'label')
        self.EXPW_LABEL_FILE_PATH = Path(self.EXPW_LABEL_PATH,'label.lst')
        self.EXPW_CROP_PATH = Path(self.EXPW_EXTRACT_PATH,'cropped')
        #011960f626b19ef4ab6e3f9ffe8ba027
        self.EXPW_PARTIAL = 0.005 # Fraction of dataset to be used
        self.EXPW_TRAIN_TEST_SPLIT = 0.95
        self.EXPW_VAL_DECISION = 'race' # 'race' or 'partial'
        self.EXPW_TRAINING_SAMPLES = 2000 #1000
        self.EXPW_mean_ds = [0.3917, 0.3120, 0.2759]
        self.EXPW_std_dev_ds = [0.2205, 0.2134, 0.2277]
        if self.IN_COLAB:
            self.EXPW_BATCH_SIZE = 16
        else:
            self.EXPW_BATCH_SIZE = 4


        if self.IN_COLAB:
            self.MODEL_DIR = Path(self.GDRIVE_FOLDER,"models") # for drive
        else:
            self.MODEL_DIR = "models" # for local
            
        self.NON_DANN_SFEW_DIR = Path(self.MODEL_DIR,"non_dann_sfew")
        self.DANN_SFEW_EXPW_DIR = Path(self.MODEL_DIR,"dann_sfew_expw")
        self.EWC_DANN_SFEW_EXPW_DIR = Path(self.MODEL_DIR,"ewc_dann_sfew_expw")



if __name__ =='__main__':
    dataconfig = DataConfig()
    
