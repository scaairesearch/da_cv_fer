from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder # for datasets
from data_config import DataConfig
from pathlib import Path
import zipfile
import torchvision.transforms as transforms # transformation with respect to mean, std, 3 channel
from torchvision.datasets import ImageFolder # for datasets (reference: Sai's usage)
from torch.utils.data import Dataset
import torch
from utils import copy_file,extract_zip_files

# class OneHotSFEWDataset(Dataset):
#     def __init__(self, root, transform = None) -> None:
#         super().__init__()
#         self.image_folder = ImageFolder(root, transform)
#         self.class_labels = self.image_folder.classes
#         self.one_hot_labels = self.get_one_hot_labels()
    
#     def __len__(self):
#         return len(self.image_folder)

#     def __getitem__(self, idx):
#         image, label = self.image_folder[idx]
#         one_hot_label = self.one_hot_labels[idx]
#         image_name = self.image_folder.imgs[idx][0]
#         return image, one_hot_label, image_name

#     def get_one_hot_labels(self):
#         one_hot_labels = []
#         for _, label in self.image_folder:
#             one_hot = torch.zeros(len(self.class_labels))
#             one_hot[label] = 1
#             one_hot_labels.append(one_hot)
#         return torch.stack(one_hot_labels)
    
class OneHotSFEWDataset(Dataset):
    def __init__(self, root, transform = None) -> None:
        super().__init__()
        self.image_folder = ImageFolder(root, transform)
        self.class_labels = self.image_folder.classes
        # self.one_hot_labels = self.get_one_hot_labels()
    
    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        one_hot_label = torch.zeros(len(self.class_labels))
        one_hot_label[label]=1

        # one_hot_label = self.one_hot_labels[idx]

        image_name = self.image_folder.imgs[idx][0]
        return image, one_hot_label, image_name

    # def get_one_hot_labels(self):
    #     one_hot_labels = []
    #     for _, label in self.image_folder:
    #         one_hot = torch.zeros(len(self.class_labels))
    #         one_hot[label] = 1
    #         one_hot_labels.append(one_hot)
    #     return torch.stack(one_hot_labels)

class DatasetSFEW():
    def __init__(self) -> None:
        # 1. Download data
        
        dataconfig = DataConfig()
        self.BASE_PATH = dataconfig.SFEW_BASE_PATH
        self.origin_file_path = dataconfig.GDRIVE_SFEW_FILE_PATH
        self.EXTRACT_PATH = dataconfig.SFEW_EXTRACT_PATH
        self.ZIP_FILE_PATH = dataconfig.SFEW_ZIP_FILE_PATH
        self.labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
        self.label_matrix = torch.eye(len(self.labels))
        self.dict_dataset = {'TRAIN_DIR' : Path(self.EXTRACT_PATH,'Train'),
                             'TEST_DIR' : Path(self.EXTRACT_PATH,'Test'),
                             'VAL_DIR' : Path(self.EXTRACT_PATH,'Val')}

        

        print(f'self.BASE_PATH -{self.BASE_PATH },\n self.EXTRACT_DIR-{self.EXTRACT_PATH},\n self.ZIP_FILE_PATH - {self.ZIP_FILE_PATH} ')
        
        # 2. Extract data
        self.extract_dataset()

        # 3. Creating Dataset Object
        self.mean_ds = dataconfig.SFEW_mean_ds 
        self.std_dev_ds = dataconfig.SFEW_std_dev_ds 
        self.train_ds, self.val_ds = None, None # initialization
        self.train_ds, self.val_ds = self.get_dataset(self.mean_ds,self.std_dev_ds)
        
        # 4. Creating Dataloader Object
        self.BATCH_SIZE = dataconfig.SFEW_BATCH_SIZE
        self.cuda = dataconfig.cuda
        self.train_dl, self.val_dl = self.get_dataloader()
        return

    def extract_dataset(self):

        if not self.EXTRACT_PATH.exists():
            # Create the directory
            self.EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
            print(f'Directory {self.EXTRACT_PATH} created successfully.')
        else:
            print(f'Directory {self.EXTRACT_PATH} already exists.')


        ## extracting data into EXTRACT_DIR

        # Open the zip if files are not unzipped before
        if len(list(self.EXTRACT_PATH.glob("*"))) > 0:
            print(f"Files exist in {self.EXTRACT_PATH}, extraction not done")
        else:
            # copy the zip file, as nothing exists
            print(f"No files (including zip file) found in {self.EXTRACT_PATH}.Copying file")
            copy_file(self.origin_file_path,self.EXTRACT_PATH)
        
        # extract the files if not already present
        extract_zip_files(self.EXTRACT_PATH, self.EXTRACT_PATH)


            # with zipfile.ZipFile(self.ZIP_FILE_PATH,'r') as zip_ref:
            #     # printing all the contents of the zip file
            #     zip_ref.printdir()
            #     # Extract all files to the specified directory
            #     zip_ref.extractall(self.EXTRACT_PATH)
            # print("File extraction complete.")

        # Extracting the files within the folder if not extracted before
        if len(list(self.EXTRACT_PATH.glob("*"))) > 0: # checking if the zip files exists 
            for dir_name, dir in self.dict_dataset.items():
                non_zip_files = [file for file in Path.iterdir(dir) if not file.name.endswith(".zip")]
                if len(non_zip_files)==0:
                    for zips in Path.iterdir(dir):
                        temp_file_name = (zips.name).split(".")[0]
                        if temp_file_name.lower() in self.labels:
                            print(dir, zips.name, temp_file_name)
                            with zipfile.ZipFile(Path(dir,zips.name), 'r') as zip_ref:
                                zip_ref.extractall(dir)
                                print(f'...completed for {dir}/{zips.name}')
                else:
                    print(f'Unzipped Files already exist in {dir}, not extracted')

        return 

    def create_dataset(self, mean_ds = None, std_dev_ds=None):
        if mean_ds is None or std_dev_ds is None:
            # imagenet data values as default
            mean_ds = [0.485, 0.456, 0.406] 
            std_dev_ds = [0.229, 0.224, 0.225]

        # Train Phase transformations
        #TODO: Use albumentations in later versions, first iteration does not include any transformations
        sfew_train_transforms = transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean_ds, std_dev_ds)
                                       ])

        # Val Phase transformations
        sfew_val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean_ds, std_dev_ds)
                                            ])
        sfew_train_ds = OneHotSFEWDataset(root=self.dict_dataset['TRAIN_DIR'],
                                    transform=sfew_train_transforms)
        
        sfew_val_ds = OneHotSFEWDataset(root=self.dict_dataset['VAL_DIR'],
                                   transform=sfew_val_transforms)
        
        # sfew_train_ds = ImageFolder(root=self.dict_dataset['TRAIN_DIR'],
        #                             transform=sfew_train_transforms)
        
        # sfew_val_ds = ImageFolder(root=self.dict_dataset['VAL_DIR'],
        #                            transform=sfew_val_transforms)
        
        return sfew_train_ds, sfew_val_ds
    
    def get_dataset(self, mean_ds = None, std_dev_ds=None):
        if self.train_ds is None and self.val_ds is None:
            if self.mean_ds is None and self.std_dev_ds is None:
               return self.create_dataset(mean_ds=mean_ds,std_dev_ds=std_dev_ds)
            else:
                return self.create_dataset(mean_ds=self.mean_ds,std_dev_ds=self.std_dev_ds)

        return self.train_ds, self.val_ds

    def get_dataloader(self,BATCH_SIZE=None):
        if BATCH_SIZE is not None:
            self.BATCH_SIZE = BATCH_SIZE
        
        dataloader_args = dict(shuffle=True, batch_size=self.BATCH_SIZE, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=self.BATCH_SIZE)

        # train dataloader
        sfew_train_loader = DataLoader(self.train_ds, **dataloader_args)

        # val dataloader
        sfew_val_loader = DataLoader(self.val_ds, **dataloader_args)

        return sfew_train_loader, sfew_val_loader

if __name__ =='__main__':
    #%%
    import utils
    import os
    from ds_sfew import DatasetSFEW
    sfew = DatasetSFEW()
    sfew_train_loader, sfew_val_loader = sfew.get_dataloader()
    utils.show_batch(sfew_train_loader,sfew.labels,4)
    # images, labels = next(iter(sfew_train_loader))
    # print(images.shape, labels.shape)
    # print("sfew labels", labels)
    images, labels, image_names = next(iter(sfew_train_loader))
    print(images.shape, labels.shape)
    print("sfew labels", labels)
    print("sfew image names full path", image_names)
    print("sfew image names:\n", image_names[0], type(image_names[0]), os.path.sep, image_names[0].split(os.path.sep)[-1])

    # print("sfew image names:\n", image_names[1].split(os.path.sep))
    

# %%
