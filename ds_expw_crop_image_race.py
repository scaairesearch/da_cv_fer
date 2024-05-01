from typing import Any
from data_config import DataConfig
from torch.utils.data import Dataset,DataLoader
import pickle
from torchvision.datasets import ImageFolder # for datasets (reference: Sai's usage)

import opendatasets as od
from pathlib import Path
import random # for random image index
from PIL import Image
import torchvision.transforms as transforms # transformation with respect to mean, std, 3 channel
import torch
import shutil # for copying files
import os
from utils import *

from facenet_pytorch import MTCNN

class DatasetEXPWIMAGECROPRACE(Dataset):
    def __init__(self, 
                 train = True, 
                 transform=None,
                 crop_at_runtime=False):
            
        self.basic_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor()])

        self.Train = train
        self.transform = transform
        self.crop_at_runtime = crop_at_runtime

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.crop_at_runtime:
            self.mtcnn = MTCNN(image_size=224, device= self.device)#MTCNN(image_size=224).to(device=self.device)
        else:
            print("---NOT CROPPING AT RUNTIME --")
            self.mtcnn = MTCNN(image_size=224).to(device='cpu') # always wanted on CPU

        dataconfig = DataConfig()

        # 1. download dataset and extracting data
        expw_link = dataconfig.EXPW_LINK
        expw_data_dir = Path(dataconfig.EXPW_BASE_PATH)
        expw_extract_dir = Path(dataconfig.EXPW_EXTRACT_PATH)
        self.crop_dir = Path(dataconfig.EXPW_CROP_PATH)
        self.val_crop_folder_path = dataconfig.RACEDS_CROP_PATH
        self.expw_trianing_samples = int(dataconfig.EXPW_TRAINING_SAMPLES)
       


        # 2. Preprocessing  extracted data to understand image and labels

        expw_label_dir = dataconfig.EXPW_LABEL_PATH
        expw_label_file = dataconfig.EXPW_LABEL_FILE_PATH
        self.expw_image_path = dataconfig.EXPW_DATA_PATH
        
        image_label_dict = {} 

        if self.Train:
            file = open(str(expw_label_file),"r")
            data = file.readlines()
            # image_label_dict = {} 
            for item in data:
                values=item.split(" ")
                image=values[0]
                label=values[-1].replace("\n", "")
                image_label_dict[image]=int(label)

            file.close()
        
            print(" before splitting : image_label_dict", len(image_label_dict))


        self.labels_map={"0":"angry",
                    "1":"disgust",
                    "2":"fear",
                    "3":"happy",
                    "4":"sad",
                    "5":"surprise",
                    "6":"neutral"}
        

        self.labels=list(self.labels_map.values())
        self.label_matrix = torch.eye(len(self.labels)) # one hot matrix
        
        # 2. splitting into train and val - 
        self.list_img_label =[]

        decision_val = dataconfig.EXPW_VAL_DECISION
        if decision_val == 'race': # decision is based on race
            try:
                print("*** Starting creation of dataset based on RACE data ***")
                pickle_raceds_crop_list_path = dataconfig.RACEDS_CROP_LIST_PATH
                print("pickle_raceds_crop_list_path: ", pickle_raceds_crop_list_path)
                with open(pickle_raceds_crop_list_path, 'rb') as f:
                    val_image_label_list_dict = pickle.load(f)

                pickle_duplicate_list_path = dataconfig.DUPLICATE_LIST_PATH
                try:
                    with open(pickle_duplicate_list_path, 'rb') as fd:
                        duplicate_image_list = pickle.load(fd)
                except Exception as e:
                    print(e)
                    duplicate_image_list =[]

                if self.Train:
                    full_list_dict = list(image_label_dict.items())
                    set_full = set(full_list_dict)
                    set_val = set(val_image_label_list_dict)
                    self.list_img_label = random.sample([(image,label) for (image,label) in list(set_full.difference(set_val)) if image not in duplicate_image_list],
                                                        self.expw_trianing_samples) # 1000 images sampled

                    print("train list", self.list_img_label[5:7], type(self.list_img_label),len(self.list_img_label))

                else:
                    self.list_img_label = [(image,label) for image,label in val_image_label_list_dict.items() if image not in duplicate_image_list] #val_image_label_list_dict
                    print("self.list_img_label: ",self.list_img_label[:5])
                    
                    # get [image,race,emotion] info
                    labels_maps_2 = {v:int(k) for k,v in self.labels_map.items()}
                    raceds_list = os.listdir(self.val_crop_folder_path)
                    expw_raceds_dict = {}
                    expw_raceds_dict_list = []
                    for emotion in raceds_list:
                        race_list = os.listdir(os.path.join(self.val_crop_folder_path,emotion))
                        expw_raceds_dict[labels_maps_2[emotion.lower()]] = {race: None for race in race_list}
                        for race in race_list:
                            expw_raceds_dict[labels_maps_2[emotion.lower()]][race] = os.listdir(os.path.join(self.val_crop_folder_path,emotion,race))
                            expw_raceds_dict_list.extend([(image,race,labels_maps_2[emotion.lower()]) for image in os.listdir(os.path.join(self.val_crop_folder_path,emotion,race))])


                    print("expw_raceds_dict_list:", expw_raceds_dict_list[:5])
                    self.expw_raceds_dict_list_selected =[]

                    list_img_label_set = set([tup[0] for tup in self.list_img_label]) # getting image names
                    for tup in expw_raceds_dict_list:
                        if tup[0] in list_img_label_set:
                            self.expw_raceds_dict_list_selected.append(tup)

                    print(f'expw_raceds_dict_list_selected: {len(self.expw_raceds_dict_list_selected)}, {self.expw_raceds_dict_list_selected[:5]}')
                      
                
                print("size of dataset (list_img_label):", len(self.list_img_label))

                print("*** Completed creation of dataset based on RACE data ***")
            except Exception as e:
                print("*** not able to create dataset based on RACE data ***")
                print("Exception message:", str(e))
                # decision_val = 'partial' # if race fails then partial works

         
        if not self.crop_at_runtime:
            flag_create_crop_contents = False # initialize

            print("---NOT CROPPING AT RUNTIME--, flag_create_crop_contents value:", flag_create_crop_contents)

            if not os.path.exists(self.crop_dir):# check if the directories are already present under expw
                create_directory(self.crop_dir) # creates if not there

            if is_directory_empty(self.crop_dir): # check for contents inside them, if contents then exists else print that nothing in crop directory
                print(f'**** {self.crop_dir} is empty***')
                if  self.Train:
                    flag_create_crop_contents = True

            ##------------<added for val problem>--------------#
            else: # if directory is not empty but the relevant files are not present then we have to crop
                if not self.Train:
                    flag_create_crop_contents = False # CHANGE THIS TO GET INFO FROM FOLDER
                else:
                    flag_create_crop_contents = True
                    for img_label in self.list_img_label:
                        if os.path.isfile(os.path.join(self.crop_dir,img_label[0])): # if the files are present, we don't need to crop
                            flag_create_crop_contents = False
                            print("***The cropped files are present, no need to crop****", img_label)
                            break # only checking for one instance
            ##------------</added for val problem>--------------#

            
            if flag_create_crop_contents:
                # populate the directories
                print("\n WARNING: It may take a long  time to crop the images, please be patient\n ")
                list_failure_tuples = []
                for image_label_tuple in self.list_img_label:
                    img_name = image_label_tuple[0]
                    img = Image.open(Path(self.expw_image_path,img_name)).convert("RGB")
                    img_save_path = os.path.join(self.crop_dir,img_name)
                    img_cropped = self.mtcnn(img,save_path = img_save_path)

                    if img_cropped is None:
                        list_failure_tuples.append(image_label_tuple)
                
                self.list_img_label= [tup for tup in self.list_img_label if tup not in list_failure_tuples]
                print(f'{len(list_failure_tuples)} were not able to crop')
                print(f'{len(os.listdir(self.crop_dir))} cropped images created in {os.path.basename(self.crop_dir)}')

    def __getitem__(self, idx):
        
        if self.Train:
            img_name = self.list_img_label[idx][0]
            label = self.list_img_label[idx][1]
            # print("list_img_label[idx]  || label.......", self.list_img_label[idx], label)
            label_onehot = self.label_matrix[int(label),:]
        
        else:
            img_name = self.expw_raceds_dict_list_selected[idx][0]
            race = self.expw_raceds_dict_list_selected[idx][1]
            label = self.expw_raceds_dict_list_selected[idx][2]
            label_onehot = self.label_matrix[int(label),:]

        if self.crop_at_runtime:
            img = Image.open(Path(self.expw_image_path,img_name))
            img_cropped = self.mtcnn(img)#.to(device=self.device)

            if img_cropped is None:
                if self.transform:
                    image_transformed = self.transform(img) # original image
                else:
                    image_transformed = self.basic_transform(img) # original image
            
                return image_transformed, label_onehot, img_name
            else:
                # Rescale the tensor from the range [-1, 1] to [0, 1]
                image_tensor_rescaled = (img_cropped + 1) / 2 
                return image_tensor_rescaled, label_onehot, img_name 

        else:
            if self.Train:
                try: # it may be possible that not all images are cropped
                    img = Image.open(Path(self.crop_dir,img_name))
                except Exception as e:
                    img = Image.open(Path(self.expw_image_path,img_name))
                    print(f'{img_name} : cropped image not found, replacing with original image')
                    print(f' The execption is {e}')

                if self.transform:
                    img_cropped = self.transform(img)
                else:
                    img_cropped = self.basic_transform(img)
                # print(f'  pixel range value = {torch.max(img_cropped.view(-1))} | {torch.min(img_cropped.view(-1))}')

                return img_cropped, label_onehot, img_name
            
            else:
                emotion_folder = self.labels_map[str(label)].capitalize()
                img = Image.open(Path(self.val_crop_folder_path,emotion_folder,race,img_name))# default as validation
                if self.transform:
                    img_cropped = self.transform(img)
                else:
                    img_cropped = self.basic_transform(img)

                return img_cropped, label_onehot, img_name, race
    
    def __len__(self):   
        return len(self.list_img_label)
    

class EXPWIMAGECROPRACE():
    def __init__(self,
                 mean_ds = None, 
                 std_dev_ds=None,
                 type = None, # values : None, train, test 
                 BATCH_SIZE = None,
                 crop_at_runtime=False):

        self.dataconfig = DataConfig()
        self.crop_at_runtime = crop_at_runtime
        self.type = type

        # 1 download data
        self.origin_file_path = self.dataconfig.GDRIVE_EXPW_FILE_PATH
        self.extract_path = self.dataconfig.EXPW_EXTRACT_PATH
        self.destination_file_path = self.dataconfig.EXPW_ZIP_FILE_PATH
        self.base_path = self.dataconfig.EXPW_BASE_PATH
        self.data_path = self.dataconfig.EXPW_DATA_PATH
        self.label_path = self.dataconfig.EXPW_LABEL_PATH
        self.duplicate_path = self.dataconfig.DUPLICATE_LIST_PATH
        self.race_crop_list_path = self.dataconfig.RACEDS_CROP_LIST_PATH

        #1-----Get / Download Data

        # Download data in case of type is None or Train

        if type is None or type == "train":
        
            print(f'desitination file path = {self.destination_file_path}')

            if not self.dataconfig.IN_COLAB:
                if os.path.exists(self.origin_file_path):
                    print(f'origin file path = {self.origin_file_path} exists')
                else:
                    print(f'origin file path = {self.origin_file_path} does NOT exist')
        
            # if the zip file is not there in destination, we have to download/copy it and later unzip it
            if not os.path.exists(self.destination_file_path):
                # if extract path is not there, we need to create 
                if not self.extract_path.exists():
                    # Create the directory
                    # print( "in EXPW()...")
                    self.extract_path.mkdir(parents=True, exist_ok=True)
                    print(f'Directory {self.extract_path} created successfully.')
                else:
                    print(f'Directory {self.extract_path} already exists.')
            
                # copy the contents from origin to destination, if file is not there in destination
                
                if not self.dataconfig.IN_COLAB:
                    try:
                        print(f"Starting File copying from {self.origin_file_path} to {self.destination_file_path}")
                        shutil.copy(self.origin_file_path,self.destination_file_path)
                        print(f"File copied successfully from {self.origin_file_path} to {self.destination_file_path}")
                    except FileNotFoundError as e:
                        print(f'Error : {e}')
                    except:
                        print(f'Error: Not able to copy from {self.origin_file_path} to {self.destination_file_path}')
                
                else:
                    if not os.path.exists(self.data_path) and os.path.exists(not self.label_path):
                        try:
                            od.download(self.dataconfig.EXPW_LINK,
                                        data_dir=self.base_path, # self.extract_path,
                                        keep_archive=True,
                                        force=True,)
                            print(f"File downloaded successfully from {self.dataconfig.EXPW_LINK} to {self.destination_file_path}")
                        except:
                            print(f'Error: Not able to download from {self.dataconfig.EXPW_LINK} to {self.destination_file_path}')

        
        #2------ extract data
                        
        # check if the file is already extracted, extracting it
        # Extract data in case of type is None or Train

        if type is None or type == "train":

            extract_zip_files(self.extract_path,self.extract_path)
        
        # check if the folders are already present
        
        if BATCH_SIZE is None:
            self.BATCH_SIZE = self.dataconfig.EXPW_BATCH_SIZE
        else:
            self.BATCH_SIZE = BATCH_SIZE

        self.mean_ds, self.std_dev_ds = None, None # initialization

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
        self.train_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(self.mean_ds, self.std_dev_ds)
                                        ])

        # Val Phase transformations
        self.val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(self.mean_ds, self.std_dev_ds)
                                            ])
        
        if self.type is None:
            self.train_ds = DatasetEXPWIMAGECROPRACE(train= True, 
                                            transform=self.train_transforms,
                                            crop_at_runtime = self.crop_at_runtime)
            
            self.val_ds = DatasetEXPWIMAGECROPRACE(train = False, 
                                            transform=self.val_transforms,
                                            crop_at_runtime = self.crop_at_runtime)
            
            return self.train_ds,self.val_ds
        
        elif self.type == 'train':
            self.train_ds = DatasetEXPWIMAGECROPRACE(train= True, 
                                            transform=self.train_transforms,
                                            crop_at_runtime = self.crop_at_runtime)
            return self.train_ds
        elif self.type == 'val' or self.type == 'test':
            self.val_ds = DatasetEXPWIMAGECROPRACE(train = False, 
                                            transform=self.val_transforms,
                                            crop_at_runtime = self.crop_at_runtime)

            return self.val_ds

        return 0

    def get_dataloader(self, BATCH_SIZE = None):

        if self.type is None and self.train_ds is None and self.val_ds is None:
            self.train_ds,self.val_ds = self.get_dataset()
        
        if self.type == "train" and self.train_ds is None:
            self.train_ds = self.get_dataset()
        
        if self.type =="test" and self.val_ds is None:
            self.val_ds = self.get_dataset()            
        
        if self.type == "val" and self.val_ds is None:
            self.val_ds = self.get_dataset()          
        
        
        if BATCH_SIZE is not None:
            self.BATCH_SIZE = BATCH_SIZE
        
        dataloader_args = dict(shuffle=True, # random info would be provided
                               batch_size=self.BATCH_SIZE,
                               num_workers=4, 
                               pin_memory=True) if self.dataconfig.cuda else dict(shuffle=True, 
                                                                                  batch_size=self.BATCH_SIZE)

        if self.type is None:
            # train dataloader
            self.train_loader = DataLoader(self.train_ds, **dataloader_args)

            # test dataloader
            self.val_loader = DataLoader(self.val_ds, **dataloader_args)

            return self.train_loader, self.val_loader
        

        elif self.type == 'train':
            return DataLoader(self.train_ds, **dataloader_args)
        elif self.type == 'val' or self.type == 'test':
            return DataLoader(self.val_ds, **dataloader_args)

        return 0
        
    
if __name__ =='__main__':

    # expw_mean_ds = [0.3917, 0.3120, 0.2759]
    # expw_std_dev_ds =[0.2205, 0.2134, 0.2277]

    # # %%
    # import utils
    # from ds_expw_crop_image_race import EXPWIMAGECROPRACE
    # expw_crop_race_object = EXPWIMAGECROPRACE(BATCH_SIZE=6, crop_at_runtime=False)
    # expw_train_ds,expw_val_ds = expw_crop_race_object.get_dataset()
    # # expw_train_loader, expw_val_loader = expw_object.get_dataloader()

    # print("*"*80)

    # expw_train_loader, expw_val_loader = expw_crop_race_object.get_dataloader()
    # utils.show_batch(expw_val_loader,expw_train_ds.labels,6,normalized=False)
    # # images, labels,image_names = next(iter(expw_train_loader))

    # images, labels,image_names = next(iter(expw_val_loader))

    # print(images.shape, labels.shape, type(images), type(labels), type(image_names))
    # # print("data labels",labels)
    # # print("image_name\n", image_names)

    # %%
    import utils
    from ds_expw_crop_image_race import EXPWIMAGECROPRACE
    expw_crop_race_object = EXPWIMAGECROPRACE(type='val', BATCH_SIZE=6, crop_at_runtime=False)
    expw_val_ds = expw_crop_race_object.get_dataset()
    # expw_train_loader, expw_val_loader = expw_object.get_dataloader()

    print("*"*80)

    expw_val_loader = expw_crop_race_object.get_dataloader()
    utils.show_batch(expw_val_loader,expw_val_ds.labels,6,normalized=False)
    # images, labels,image_names = next(iter(expw_train_loader))

    images, labels,image_names,race = next(iter(expw_val_loader))

    print(images.shape, labels.shape, race, type(images), type(labels), type(image_names), type(race))
    # print("data labels",labels)
    # print("image_name\n", image_names)
# %%
# utils.show_batch(expw_train_loader,expw_train_ds.labels,6,normalized=False)

# %%
