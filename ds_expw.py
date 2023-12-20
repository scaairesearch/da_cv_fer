from typing import Any
from data_config import DataConfig
from torch.utils.data import Dataset,DataLoader

import opendatasets as od
from pathlib import Path
import random # for random image index
from PIL import Image
import torchvision.transforms as transforms # transformation with respect to mean, std, 3 channel
import torch

class DatasetEXPW(Dataset):
    def __init__(self, train = True, transform=None) -> None:
        
        self.Train = train
        self.transform = transform

        dataconfig = DataConfig()

        # 1. download dataset and extracting data
        expw_link = dataconfig.EXPW_LINK
        expw_data_dir = Path(dataconfig.EXPW_BASE_PATH)
        expw_extract_dir = Path(dataconfig.EXPW_EXTRACT_PATH)


        if not expw_extract_dir.exists():
            # Create the directory
            expw_extract_dir.mkdir(parents=True, exist_ok=True)
            print(f'Directory {expw_extract_dir} created successfully.')
        else:
            print(f'Directory {expw_extract_dir} already exists.')

        if len(list(expw_extract_dir.glob("*"))) == 0: # checking if the zip files exists
            od.download(dataset_id_or_url=expw_link, data_dir=str(expw_data_dir), force=True) 
        # if len(list(expw_data_dir.glob("*"))) == 0: # checking if the zip files exists
        #     od.download(dataset_id_or_url=expw_link, data_dir=str(expw_data_dir), force=True)

        # 2. Preprocessing  extracted data to understand image and labels

        expw_label_dir = dataconfig.EXPW_LABEL_PATH
        expw_label_file = dataconfig.EXPW_LABEL_FILE_PATH
        self.expw_image_path = dataconfig.EXPW_DATA_PATH


        file = open(str(expw_label_file),"r")
        data = file.readlines()
        image_label_dict = {} 
        for item in data:
            values=item.split(" ")
            image=values[0]
            label=values[-1].replace("\n", "")
            image_label_dict[image]=int(label)

        file.close()

        labels_map={"0":"angry",
                    "1":"disgust",
                    "2":"fear",
                    "3":"happy",
                    "4":"sad",
                    "5":"surprise",
                    "6":"neutral"}
        

        self.labels=list(labels_map.values())
        self.label_matrix = torch.eye(len(self.labels)) # one hot matrix

        # 2. splitting into train and val - 80/20
        total_im=len(image_label_dict)
        num_train=int(len(image_label_dict)*0.8)
        num_val=total_im-num_train

        full_list_dict=list(image_label_dict.items())
        random.shuffle(full_list_dict)

        # self.train_list_dict = full_list_dict[:num_train]
        # self.val_list_dict = full_list_dict[num_train:num_train+num_val]

        if self.Train:
            self.list_img_label = full_list_dict[:num_train]
        else:
            self.list_img_label = full_list_dict[num_train:num_train+num_val]
        
        #  # 3. Creating Dataset Object
        # self.mean_ds = dataconfig.EXPW_mean_ds
        # self.std_dev_ds = dataconfig.EXPW_mean_ds
        # self.train_ds, self.val_ds = None, None # initialization
        # self.train_ds, self.val_ds = self.get_dataset(self.mean_ds,self.std_dev_ds)


    def __getitem__(self, idx):

        # img_name = list(self.list_img_label.keys())[idx]
        img_name = self.list_img_label[idx][0]

        # label = self.list_img_label[img_name]
        label = self.list_img_label[idx][1]
        label_onehot = self.label_matrix[int(label),:]

        img = Image.open(Path(self.expw_image_path,img_name))

        if self.transform:
            img = self.transform(img)
        
        # return img, label
        return img, label_onehot
    
    def __len__(self):   
        return len(self.list_img_label)
    

class EXPW():
    def __init__(self,mean_ds = None, std_dev_ds=None, BATCH_SIZE = None):

        self.dataconfig = DataConfig()
        if BATCH_SIZE is None:
            self.BATCH_SIZE = self.dataconfig.EXPW_BATCH_SIZE
        else:
            self.BATCH_SIZE = BATCH_SIZE

        self.mean_ds, self.std_dev_ds = None, None

        if mean_ds is None and std_dev_ds is None:
            self.mean_ds = self.dataconfig.EXPW_mean_ds
            self.std_dev_ds = self.dataconfig.EXPW_std_dev_ds
        else:
            self.mean_ds = mean_ds
            self.std_dev_ds = std_dev_ds
    
        self.train_ds, self.val_ds = None, None # initialization
        self.train_transforms = None
        self.val_transforms = None
        self.train_loader, self.val_loader = None, None

    def get_dataset(self):
        # Train Phase transformations
        #TODO: Use albumentations in later versions, first iteration does not include any transformations
        self.train_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean_ds, self.std_dev_ds)
                                        ])

        # Val Phase transformations
        self.val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean_ds, self.std_dev_ds)
                                            ])
    
        self.train_ds = DatasetEXPW(train= True, transform=self.train_transforms)
        self.val_ds = DatasetEXPW(train = False, transform=self.val_transforms)

        return self.train_ds,self.val_ds

    def get_dataloader(self, BATCH_SIZE = None):
        if self.train_ds is None and self.val_ds is None:
            self.train_ds,self.val_ds = self.get_dataset()
        
        if BATCH_SIZE is not None:
            self.BATCH_SIZE = BATCH_SIZE
        
        dataloader_args = dict(shuffle=True, batch_size=self.BATCH_SIZE, num_workers=4, pin_memory=True) if self.dataconfig.cuda else dict(shuffle=True, batch_size=self.BATCH_SIZE)

        # train dataloader
        self.train_loader = DataLoader(self.train_ds, **dataloader_args)

        # test dataloader
        self.val_loader = DataLoader(self.val_ds, **dataloader_args)

        return self.train_loader, self.val_loader
        


def get_expw_dataloaders(BATCH_SIZE = None):
    dataconfig = DataConfig()
    expw_mean_ds = dataconfig.EXPW_mean_ds
    expw_std_dev_ds = dataconfig.EXPW_std_dev_ds
    BATCH_SIZE = dataconfig.EXPW_BATCH_SIZE

    # Train Phase transformations
    #TODO: Use albumentations in later versions, first iteration does not include any transformations
    expw_train_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        #  transforms.RandomCrop(224, padding=10, padding_mode='reflect'),
                                        #  transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomRotation(5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expw_mean_ds, expw_std_dev_ds)
                                        ])

    # Val Phase transformations
    expw_val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expw_mean_ds, expw_std_dev_ds)
                                        ])
    
    expw_train_ds = DatasetEXPW(train= True, transform=expw_train_transforms)
    expw_valid_ds = DatasetEXPW(train = False, transform=expw_val_transforms)

    dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) if dataconfig.cuda else dict(shuffle=True, batch_size=BATCH_SIZE)

    # train dataloader
    expw_train_loader = DataLoader(expw_train_ds, **dataloader_args)

    # test dataloader
    expw_val_loader = DataLoader(expw_valid_ds, **dataloader_args)

    return expw_train_loader, expw_val_loader




    
if __name__ =='__main__':
    # expw = DatasetEXPW()

    # expw_mean_ds = [0.3917, 0.3120, 0.2759]
    # expw_std_dev_ds =[0.2205, 0.2134, 0.2277]

    # # Train Phase transformations
    # #TODO: Use albumentations in later versions, first iteration does not include any transformations
    # expw_train_transforms = transforms.Compose([
    #                                     transforms.Resize((224, 224)),
    #                                     #  transforms.RandomCrop(224, padding=10, padding_mode='reflect'),
    #                                     #  transforms.RandomHorizontalFlip(),
    #                                     #  transforms.RandomRotation(5),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(expw_mean_ds, expw_std_dev_ds)
    #                                     ])

    # # Val Phase transformations
    # expw_val_transforms = transforms.Compose([transforms.Resize((224, 224)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(expw_mean_ds, expw_std_dev_ds)
    #                                     ])
    
    # expw_train_ds = DatasetEXPW(train= True, transform=expw_train_transforms)
    # expw_valid_ds = DatasetEXPW(train = False, transform=expw_val_transforms)

    # # dataloader arguments
    # dataconfig = DataConfig()
    # BATCH_SIZE = 16
    # dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) if dataconfig.cuda else dict(shuffle=True, batch_size=BATCH_SIZE)

    # # train dataloader
    # expw_train_loader = DataLoader(expw_train_ds, **dataloader_args)

    # # test dataloader
    # expw_val_loader = DataLoader(expw_valid_ds, **dataloader_args)
    # #%%
    
    # from ds_expw import *
    # import utils
    # dataset = DatasetEXPW(train= True, transform=None) # just to get labels!!
    
    # expw_train_loader,expw_val_loader = ds_expw.get_expw_dataloaders()

    # utils.show_batch(expw_train_loader,dataset.labels,2)

    #%% 
    import utils
    from torchvision.transforms import transforms
    from ds_expw import DatasetEXPW
    from data_config import DataConfig
    from torch.utils.data import Dataset, DataLoader
    expw_mean_ds = [0.3917, 0.3120, 0.2759]
    expw_std_dev_ds =[0.2205, 0.2134, 0.2277]
    expw_train_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        #  transforms.RandomCrop(224, padding=10, padding_mode='reflect'),
                                        #  transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomRotation(5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expw_mean_ds, expw_std_dev_ds)
                                        ])
    expw_train_ds = DatasetEXPW(train= True, transform=expw_train_transforms)
    dataconfig = DataConfig()
    BATCH_SIZE = 4
    dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) if dataconfig.cuda else dict(shuffle=True, batch_size=BATCH_SIZE)

    # train dataloader
    expw_train_loader = DataLoader(expw_train_ds, **dataloader_args)

    utils.show_batch(expw_train_loader,expw_train_ds.labels,2)

    images, labels = next(iter(expw_train_loader))
    print(images.shape, labels.shape)
    print("data labels",labels)

    # %%
    import utils
    from ds_expw import EXPW
    expw_object = EXPW(BATCH_SIZE=6)
    expw_train_ds,expw_val_ds = expw_object.get_dataset()
    expw_train_loader, expw_val_loader = expw_object.get_dataloader()
    utils.show_batch(expw_train_loader,expw_train_ds.labels,3)

    images, labels = next(iter(expw_train_loader))
    print(images.shape, labels.shape)
    print("data labels",labels)





# %%
