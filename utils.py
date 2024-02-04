
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import zipfile
from pathlib import Path
import shutil
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt # for visualizing images
from torchvision import transforms



def unnormalize(image_tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    unnormalize = Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    # Apply unnormalization
    image_tensor_unnormalized = unnormalize(image_tensor)

    # Ensure the values are within the expected range
    image_tensor_unnormalized[image_tensor_unnormalized > 1] = 1
    image_tensor_unnormalized[image_tensor_unnormalized < 0] = 0

    return image_tensor_unnormalized



   
# def show_batch(dataset_loader,label_names,num_images=5):
#   '''
#   shows a batch of images (default = 5)
#   '''
#   images, targets = next(iter(dataset_loader))
#   is_one_hot = (targets.sum(dim=-1) == 1).all()
#   plt.figure(figsize=(16, 8))
#   for i in range(num_images):
#     ax = plt.subplot(int(num_images//5)+1, 5, i + 1)
#     images[i] = images[i] / 2 + 0.5 # unnormalize, though not the best way
#     ax.imshow(images[i].permute(1, 2, 0))
#     if is_one_hot:
#       label_index = torch.argmax(targets[i], dim = -1)
#       plt.title(label_names[label_index])
#     else:
#       plt.title(label_names[targets[i]])
#     plt.axis("off")

def show_batch(dataset_loader,label_names,num_images=5,normalized=True):
  '''
  shows a batch of images (default = 5)
  '''
  batch = next(iter(dataset_loader))
  images, targets = batch[0], batch[1]
  if len(batch) == 3:
    image_names = batch[2]
  is_one_hot = (targets.sum(dim=-1) == 1).all()
  plt.figure(figsize=(16, 8))
  for i in range(num_images):

    ax = plt.subplot(int(num_images//5)+1, 5, i + 1)
    # print("image tensor: \n",images[i])
    # images[i] = images[i] / 2 + 0.5 # unnormalize, though not the best way
    if normalized:
      images[i] = unnormalize(images[i]) # created new function
    ax.imshow(images[i].permute(1, 2, 0))
    if is_one_hot:
      label_index = torch.argmax(targets[i], dim = -1)
      if len(batch) ==3:
        plt.title(label_names[label_index]+"\n"+image_names[i].split(os.path.sep)[-1] )
      else:
        plt.title(label_names[label_index])
    else:
      plt.title(label_names[targets[i]])
    plt.axis("off")

def show_batch_3(dataset_loader,label_names,num_images=5):
  '''
  shows a batch of images (default = 5)
  '''
  images, targets, image_names = next(iter(dataset_loader))
  is_one_hot = (targets.sum(dim=-1) == 1).all()
  plt.figure(figsize=(16, 8))
  for i in range(num_images):
    ax = plt.subplot(int(num_images//5)+1, 5, i + 1)
    images[i] = images[i] / 2 + 0.5 # unnormalize, though not the best way
    ax.imshow(images[i].permute(1, 2, 0))
    if is_one_hot:
      label_index = torch.argmax(targets[i], dim = -1)
      plt.title(label_names[label_index]+image_names[i].split(os.path.sep)[-1] )
    else:
      plt.title(label_names[targets[i]])
    plt.axis("off")

def get_correct_predictions(prediction, labels):
    """
    Function to return total number of correct predictions
    :param prediction: Model predictions on a given sample of data
    :param labels: Correct labels of a given sample of data
    :return: Number of correct predictions
    """
    return prediction.argmax(dim=1).eq(labels).sum().item()


def extract_zip_files_old(directory_path, extract_to):
    try:
        # List all files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        # Check if all files have a ".zip" extension
        if all(file.lower().endswith('.zip') for file in files):
            for file in files:
                file_path = os.path.join(directory_path, file)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all contents to the specified directory
                    print(f"Initiating extraction of {file} to {extract_to}")
                    zip_ref.extractall(extract_to)
                    print(f"Successfully extracted {file} to {extract_to}")
        else:
            print("Directory contains non-zip files. Skipping extraction.")

    except (zipfile.BadZipFile, OSError) as e:
        print(f"Error: {e}")

def extract_zip_files(directory_path, extract_to):
  ''' Extracts the zip files if not existing'''
  directory_path, extract_to = Path(directory_path), Path(extract_to) # converting from string to path
  if len(list(extract_to.glob("*"))) > 0: # checking if the zip files exists 
        non_zip_files = [file for file in Path.iterdir(directory_path) if not file.name.endswith(".zip")]
        if len(non_zip_files)==0:
            for zips in Path.iterdir(directory_path):
              with zipfile.ZipFile(Path(directory_path,zips.name), 'r') as zip_ref:
                  print(f"Initiating extraction of {Path(directory_path,zips.name)} to {extract_to}")
                  zip_ref.extractall(extract_to)
                  print(f'...completed for {directory_path}/{zips.name}')
        else:
            print(f'Unzipped Files already exist in {directory_path}, not extracted')

def copy_file(origin_file_path,destination_file_path):
  try:
    print(f"Starting File copying from {origin_file_path} to {destination_file_path}")
    shutil.copy(origin_file_path,destination_file_path)
    print(f"File copied successfully from {origin_file_path} to {destination_file_path}")
  except FileNotFoundError as e:
    print(f'Error : {e}')
  except:
    print(f'Error: Not able to copy from {origin_file_path} to {destination_file_path}')

def plot_loss_curves(dict_losses, mode = "train_losses", label ='Non DANN ' + 'SFEW' ):
    fig=plt.figure(figsize=(10,20))
    fig.add_subplot(5, 1, 2)
    # for embedding in dict_emb_file.keys():
    list1_to_plot= dict_losses[mode]
    plt.plot(range(1,len(list1_to_plot)+1),list1_to_plot, label = label)
    plt.xlabel('number of epochs', fontsize=10)
    plt.ylabel(str(mode), fontsize=10)
    plt.legend(loc = 'upper right')
    plt.title(" "+mode)
    plt.show()

def early_stopping_difference(list_loss: list, patience = 5, difference = 0.0003):
  if len(list_loss) > patience:
    reverse_list_loss = list_loss[::-1]
    reverse_list_loss = reverse_list_loss[0:patience+1]
    for index in range(0,len(reverse_list_loss)-1):
      if abs(reverse_list_loss[index] - reverse_list_loss[index+1]) > difference:
        return False
    return True
  else:
    return False

def create_directory(directory_path):
  ''' Creates directory if not present'''
  try:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    return True
  except OSError as e:
    print(f"Error: {e}")
    return False
  
def is_directory_empty(directory_path):
  ''' Check if there are no files /dir underneath a directory'''
  try:
      # List all files and directories in the given path
      contents = os.listdir(directory_path)

      # Filter out only files (not directories)
      # files = [item for item in contents if os.path.isfile(os.path.join(directory_path, item))]
      items = [item for item in contents]

      # Check if there are no files
      return len(items) == 0
  
  except OSError as e:
      print(f"Error: {e}")
      return True  # Treat errors as an empty directory
  
def save_image(image_data, save_path):
    '''Saves image on CPU'''
    # Convert PyTorch tensor to PIL Image if needed
    image_data_pil = transforms.ToPILImage()(image_data.cpu())
    # Save the PIL Image to the specified path
    image_data_pil.save(save_path)



   

if __name__ == '__main__':
  print(torch.tensor([1.,0]*10).view(-1,2))
  label_matrix = torch.eye(7)
  print(label_matrix)
  for i in range(7):
    print(label_matrix[i,:])
  
  extract_zip_files('dataset\expwds','dataset\expwds')

  
