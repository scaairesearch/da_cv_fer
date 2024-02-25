from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder # for datasets
from data_config import DataConfig
from pathlib import Path
import zipfile
import torchvision.transforms as transforms # transformation with respect to mean, std, 3 channel
from torchvision.datasets import ImageFolder # for datasets (reference: Sai's usage)
from torch.utils.data import Dataset
import torch
from utils import *
from PIL import Image
from torchvision.transforms import ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
import os


import numpy as np

from facenet_pytorch import MTCNN

class OneHotSFEWCROPDataset(Dataset):
    def __init__(self, root, transform = None, crop_at_runtime = False) -> None:
        super().__init__()

        self.crop_at_runtime = crop_at_runtime 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.crop_at_runtime:
            self.mtcnn = MTCNN(image_size=224,device=self.device)#MTCNN(image_size=224).to(device=self.device)
        

        dataconfig = DataConfig()
        self.transform = transform
        self.basic_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor()])
        self.image_folder = ImageFolder(root, transform=self.basic_transform)
        self.class_labels = self.image_folder.classes
        self.to_pil = ToPILImage()
        self.mean_ds = dataconfig.SFEW_mean_ds
        self.std_dev_ds = dataconfig.SFEW_std_dev_ds
        self.tranforms_type = dataconfig.SFEW_TRANSFORMS
        # self.one_hot_labels = self.get_one_hot_labels()
    
    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]        
        one_hot_label = torch.zeros(len(self.class_labels))
        one_hot_label[label]=1
        image_name = self.image_folder.imgs[idx][0]
        # print(f'  pixel range value = {torch.max(image.view(-1))} | {torch.min(image.view(-1))}')
        # print("image before \n", image)
        # self.to_pil(image).show()

        if image is None or (torch.all(image == 0).item() == 1):
            return torch.zeros((3,224,224)),one_hot_label, image_name
        
        if self.crop_at_runtime:
            image_pil = self.to_pil(image)
            # image_pil.show()

            # Get cropped and prewhitened image tensor
            img_cropped = self.mtcnn(image_pil,device=self.device)#.to(device=self.device)

            if img_cropped is None: # case where face is not detected
                if self.transform:
                    image = self.transform(image_pil)
                else:
                    image = self.basic_transform(image_pil)

                return image,one_hot_label, image_name
            else:# Rescale the tensor from the range [-1, 1] to [0, 1]
                image_tensor_rescaled = (img_cropped + 1) / 2
                # print(f'  pixel range value = {torch.max(image_tensor_rescaled.view(-1))} | {torch.min(image_tensor_rescaled.view(-1))}')
                # self.to_pil(image_tensor_rescaled).show()
                return image_tensor_rescaled, one_hot_label, image_name
        
        else:
            # print(f'  pixel range value before = {torch.max(image.view(-1))} | {torch.min(image.view(-1))}')

            if self.transform:
                image = self.transform(self.to_pil(image))
            else:
                image = self.basic_transform(self.to_pil(image))
            # print(f'  pixel range value = {torch.max(image.view(-1))} | {torch.min(image.view(-1))}')

            return image, one_hot_label, image_name




        # try:
        #     image_tensor_rescaled = (img_cropped + 1.0) / 2.0
        # except:
        #     print("%"*80)
        #     if img_cropped is None:
        #         print(f'img_cropped could not be retrived')
        #     print(type(img_cropped))
        #     print(img_cropped.__class__)
        #     print(image_name)
        #     print(type(img_cropped))
        #     image_pil.show()
        #     image = self.transform(image_pil)
        #     return image,one_hot_label, image_name

        # image_cropped_pil = self.to_pil(image_tensor_rescaled)
        # image_cropped_pil.show()


        # one_hot_label = self.one_hot_labels[idx]

        # if self.transform:
        #     if self.tranforms_type == 'A': # using albumentations
        #         # image = lambda x : self.transform(image=x)["image"]
        #         print("0...")
        #         image_np = image.numpy()#np.array(image) # image # np.array(image)
        #         print("1...")
        #         augmented = self.transform(image=image_np)
        #         print("2...")
        #         image = augmented['image']
        #         print("3...")
        #         # image_pil = Image.fromarray(image_transformed) # Convert NumPy array back to PIL image
        #         # print("4...")
        #         # image = ToTensorV2()(image_pil) # Convert PIL image to PyTorch tensor

        #     else:
        #         image = unnormalize(image,
        #                             mean=self.mean_ds, 
        #                             std=self.std_dev_ds) # image is already normalized
        #         image = self.to_pil(image) # converting into PIL object
        #         image = self.transform(image) # applying transform on PIL object

        # return image, one_hot_label, image_name

    # def get_one_hot_labels(self):
    #     one_hot_labels = []
    #     for _, label in self.image_folder:
    #         one_hot = torch.zeros(len(self.class_labels))
    #         one_hot[label] = 1
    #         one_hot_labels.append(one_hot)
    #     return torch.stack(one_hot_labels)

class DatasetSFEWCROP():
    def __init__(self,crop_at_runtime=False) -> None:
        # 1. Download data
        self.crop_at_runtime = crop_at_runtime
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.crop_at_runtime:
            self.mtcnn = MTCNN(image_size=224,device=self.device)#MTCNN(image_size=224).to(device='cpu') # offline changes are on CPU and not GPU
            # self.mtcnn = MTCNN(image_size=224) # offline changes are on CPU and not GPU


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
        self.dict_crop_dataset = {'CROP_TRAIN_DIR' : Path(self.EXTRACT_PATH,'Train_Crop'),
                                  'CROP_VAL_DIR': Path(self.EXTRACT_PATH,'Val_Crop')}

        self.tranforms_type = dataconfig.SFEW_TRANSFORMS

        print(f' self.BASE_PATH -{self.BASE_PATH }, \n self.EXTRACT_DIR-{self.EXTRACT_PATH},\n self.ZIP_FILE_PATH - {self.ZIP_FILE_PATH} ')
        
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

        # if the train directory exists, look for folders which are not zip
        # corresponding create train_crop and val_crop

        if not self.crop_at_runtime:
            flag_create_crop_contents = False
            for dir_name, dir_path in self.dict_crop_dataset.items():
                if not os.path.exists(dir_path): # check if the directories are already present under sfew
                    create_directory(dir_path) # creates if not there
                    flag_create_crop_contents = True

                else:
                    if is_directory_empty(dir_path): # check for contents inside them, if contents then exists else print that nothing in crop directory
                        print(f'**** {dir_name}/{dir_path} is empty***')
                        flag_create_crop_contents = True

            if flag_create_crop_contents:
                self.create_crop_contents()


        return 

    def create_crop_contents(self):
        for dir_name, dir_path in self.dict_dataset.items():
            if 'TEST' in dir_name:
                pass # no treatment for test directory
            else:
                crop_dir_name = None      
                for key in self.dict_crop_dataset:
                    if str(dir_name) in key:
                        crop_dir_name = f'CROP_{dir_name}'
                    
                if crop_dir_name:
                    # for each file in dir_path, do the treatment and store in approrpiate directory
                    list_subdir = [ os.path.join(dir_path,subdir) for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,subdir))]
                    # print(list_subdir)
                    for subdir in list_subdir:
                        image_file_names = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
                        target_dir = os.path.join(self.dict_crop_dataset[crop_dir_name],
                                                         os.path.basename(subdir))
                        if os.path.exists(target_dir):
                            print(f'***No files cropped for { os.path.basename(subdir)}, it is assumed to have files already')
                        else:
                            for image_name in image_file_names:
                                img = Image.open(os.path.join(subdir, image_name)).convert("RGB")
                                img_save_path = os.path.join(target_dir,
                                                            image_name)
                                # print(f'{os.path.join(subdir, image_name)} || {image_name} || {img_save_path}')
                                img_cropped = self.mtcnn(img, save_path = img_save_path) #.to(device=self.device)
                        
                            print(f'{len(image_file_names)} cropped images created in {os.path.basename(subdir)}')




        

    def create_dataset(self, mean_ds = None, std_dev_ds=None):
        if mean_ds is None or std_dev_ds is None:
            # imagenet data values as default
            mean_ds = [0.485, 0.456, 0.406] 
            std_dev_ds = [0.229, 0.224, 0.225]
    

        # Train Phase transformations
        #TODO: Use albumentations in later versions, first iteration does not include any transformations
        print(f'----------mean_ds = {mean_ds}, std_dev_ds = {std_dev_ds}----------')
        if self.tranforms_type == 'A': # Albumentations based
            sfew_train_transforms = A.Compose([
                A.Resize(224,224),# Resize the image to a specific size while maintaining the aspect ratio
                A.HorizontalFlip(p=0.7),# Apply horizontal flip with a probability of 50%
                A.Rotate(limit =15, p=0.7), # Apply a random rotation between +/- 7 degrees with 50% probability
                # A.GaussNoise( p=0.2), # Apply noise
                # A.RandomBrightnessContrast(p=0.5),# Random brigtness and Contrast
                # A.Normalize(mean=mean_ds, std=std_dev_ds),  # Normalize with calculated mean and std
                ToTensorV2(p=1.0) # Convert the image to a PyTorch tensor       
            ])
        else:
            sfew_train_transforms = transforms.Compose([
                                        # transforms.CenterCrop(size = (224,224)),
                                        transforms.Resize((224, 224)),
                                        transforms.RandomApply([transforms.RandomResizedCrop(size=(224,224),scale=(0.8,1.0))],p=0.7),  
                                        transforms.RandomApply([transforms.RandomHorizontalFlip(p=0.7)]),  # Horizontal flip with 70% probability
                                        transforms.RandomApply([transforms.RandomRotation(degrees=(-15, 15),fill=(1,))], p=0.7),  # Random rotation with 70% probability
                                        transforms.RandomApply([transforms.Grayscale(num_output_channels = 3)], p=0.3) , # gray scale
                                        transforms.RandomApply([v2.ColorJitter(brightness=.5, hue=.3)], p=0.3) , # color jitter
                                        transforms.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.3) , # gaussian blur
                                        transforms.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=2)], p=0.3) , # sharpness
                                        transforms.RandomApply([v2.RandomAutocontrast()], p=0.3) , # autocontrast
                                        transforms.RandomApply([v2.RandomEqualize()], p=0.3) , # equalize
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(255,255,255)) , # cut out white
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(255,255,255)) , # cut out white
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(255,255,255)) , # cut out white
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(255,255,255)) , # cut out white
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(255,255,255)) , # cut out white
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(0,0,0)), # cut out black
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(0,0,0)), # cut out black
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(0,0,0)), # cut out black
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(0,0,0)), # cut out black
                                        cutout(mask_size=24,p=0.9,cutout_inside=False, mask_color=(0,0,0)), # cut out black

                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean_ds, std_dev_ds)
                                        ])

        # Val Phase transformations
        if self.tranforms_type == 'A': # Albumentations based
            sfew_val_transforms=A.Compose([
                A.Resize(224,224),# Resize the image to a specific size while maintaining the aspect ratio
                # A.Normalize(mean=mean_ds, std=std_dev_ds),  # Normalize with calculated mean and std
                ToTensorV2() # Convert the image to a PyTorch tensor
            ])
        else:
            sfew_val_transforms = transforms.Compose([
                                                # transforms.CenterCrop(size = (224,224)),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean_ds, std_dev_ds)
                                                ])
        if self.crop_at_runtime:
            sfew_train_ds = OneHotSFEWCROPDataset(root=self.dict_dataset['TRAIN_DIR'],
                                    transform=sfew_train_transforms,crop_at_runtime = True)
            sfew_val_ds = OneHotSFEWCROPDataset(root=self.dict_dataset['VAL_DIR'],
                                   transform=sfew_val_transforms, crop_at_runtime = True)
        else:
            sfew_train_ds = OneHotSFEWCROPDataset(root=self.dict_crop_dataset['CROP_TRAIN_DIR'],
                                    transform=sfew_train_transforms,
                                    crop_at_runtime = False)
        
            sfew_val_ds = OneHotSFEWCROPDataset(root=self.dict_crop_dataset['CROP_VAL_DIR'],
                                   transform=sfew_val_transforms,
                                   crop_at_runtime = False)
        

        
        # sfew_train_ds = OneHotSFEWDataset(root=self.dict_dataset['TRAIN_DIR'],
        #                             transform=None)
        
        # sfew_val_ds = OneHotSFEWDataset(root=self.dict_dataset['VAL_DIR'],
        #                            transform=None)
        
        # sfew_train_ds = ImageFolder(root=self.dict_dataset['TRAIN_DIR'],
        #                             transform=sfew_train_transforms)
        
        # sfew_val_ds = ImageFolder(root=self.dict_dataset['VAL_DIR'],
        #                            transform=sfew_val_transforms)
        
        return sfew_train_ds, sfew_val_ds
    
    def get_dataset(self, mean_ds = None, std_dev_ds=None):
        if self.train_ds is None and self.val_ds is None:
            if self.mean_ds is None and self.std_dev_ds is None:
               return self.create_dataset(mean_ds=self.mean_ds,std_dev_ds=self.std_dev_ds)
            else:
                return self.create_dataset(mean_ds=mean_ds,std_dev_ds=std_dev_ds)


        return self.train_ds, self.val_ds

    def get_dataloader(self,BATCH_SIZE=None):
        if BATCH_SIZE is not None:
            self.BATCH_SIZE = BATCH_SIZE
        
        dataloader_args = dict(shuffle=True, batch_size=self.BATCH_SIZE, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=self.BATCH_SIZE)

        if self.train_ds is None or self.val_ds is None:
            # self.train_ds, self.val_ds = self.get_dataset(mean_ds=self.mean_ds,std_dev_ds=self.std_dev_ds)
            self.train_ds, self.val_ds = self.create_dataset()

        # train dataloader
        sfew_train_loader = DataLoader(self.train_ds, **dataloader_args)

        # val dataloader
        sfew_val_loader = DataLoader(self.val_ds, **dataloader_args)

        return sfew_train_loader, sfew_val_loader

if __name__ =='__main__':
    #%%
    import utils
    import os
    from ds_sfew_crop import DatasetSFEWCROP
    sfew = DatasetSFEWCROP(crop_at_runtime=False)
    sfew_train_loader, sfew_val_loader = sfew.get_dataloader()
    utils.show_batch(sfew_train_loader,sfew.labels,4,normalized=False)
    # # images, labels = next(iter(sfew_train_loader))
    # # print(images.shape, labels.shape)
    # # print("sfew labels", labels)
    # images, labels, image_names = next(iter(sfew_train_loader))
    # print(images.shape, labels.shape)
    # print("sfew labels", labels)
    # print("sfew image names full path", image_names)
    # print("sfew image names:\n", image_names[0], type(image_names[0]), os.path.sep, image_names[0].split(os.path.sep)[-1])

    # print("sfew image names:\n", image_names[1].split(os.path.sep))
    # images, labels, _ = next(iter(sfew_train_loader))
    # # print(images)
    # print(images.shape, labels.shape)
    # print("sfew labels", labels)

    

# %%
# utils.show_batch(sfew_train_loader,sfew.labels,4,normalized=False)


# %%
