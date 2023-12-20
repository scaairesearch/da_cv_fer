from pathlib import Path
import torch
class DataConfig:
    def __init__(self) -> None:
        source = 'sfew'
        self.cuda = torch.cuda.is_available()

        self.GDRIVE_FOLDER = '/content/gdrive/MyDrive/CV_FER'

        self.SFEW_BASE_PATH = Path(self.GDRIVE_FOLDER,'dataset')
        # self.SFEW_BASE_PATH = 'dataset'

        self.SFEW_EXTRACT_PATH = Path(self.SFEW_BASE_PATH,"sfew")
        self.SFEW_ZIP_FILE_PATH = Path(self.SFEW_BASE_PATH,'SFEW_2.zip')
        self.SFEW_mean_ds = [0.2197, 0.1858, 0.1569]
        self.SFEW_std_dev_ds = [0.1810, 0.1635, 0.1511]
        self.SFEW_BATCH_SIZE = 4
        

        self.EXPW_LINK = "https://www.kaggle.com/datasets/mohammedaaltaha/expwds"
        # self.EXPW_BASE_PATH = 'dataset'
      
        self.EXPW_BASE_PATH = Path(self.GDRIVE_FOLDER,'dataset')
        self.EXPW_EXTRACT_PATH = Path(self.EXPW_BASE_PATH,"expwds")
        self.EXPW_DATA_PATH = Path(self.EXPW_EXTRACT_PATH,'origin')
        self.EXPW_LABEL_PATH = Path(self.EXPW_EXTRACT_PATH,'label')
        self.EXPW_LABEL_FILE_PATH = Path(self.EXPW_LABEL_PATH,'label.lst')
        #011960f626b19ef4ab6e3f9ffe8ba027
        self.EXPW_mean_ds = [0.3917, 0.3120, 0.2759]
        self.EXPW_std_dev_ds = [0.2205, 0.2134, 0.2277]
        self.EXPW_BATCH_SIZE = 4

        # self.MODEL_DIR = "models"
        self.MODEL_DIR = Path(self.GDRIVE_FOLDER,"models")
        self.NON_DANN_SFEW_DIR = Path(self.MODEL_DIR,"non_dann_sfew")
        self.DANN_SFEW_EXPW_DIR = Path(self.MODEL_DIR,"dann_sfew_expw")

        


if __name__ =='__main__':
    dataconfig = DataConfig()
    