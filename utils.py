
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import zipfile
from pathlib import Path
import shutil

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

def show_batch(dataset_loader,label_names,num_images=5):
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
    images[i] = images[i] / 2 + 0.5 # unnormalize, though not the best way
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




if __name__ == '__main__':
  print(torch.tensor([1.,0]*10).view(-1,2))
  label_matrix = torch.eye(7)
  print(label_matrix)
  for i in range(7):
    print(label_matrix[i,:])
  
  extract_zip_files('dataset\expwds','dataset\expwds')

  
