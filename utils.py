
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

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

def get_misclassified_images_list(model, device, test_loader, num_image = 10):
  '''
  returns list of misclassified images, it does not display the images
  '''
  model.eval() # setting the model in evaluation mode
  list_misclassified_images, labels_list,preds_list = [],[],[] # initialize
  emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
  with torch.no_grad():
    for batch in test_loader:
      images, labels,img_names = batch[0].to(device),batch[1].to(device),batch[2] #sending data to CPU or GPU as per device
      outputs = model(images) # forward pass, result captured in outputs (plural as there are many images in a batch)
      # the outputs are in batch size x one hot vector
      preds = outputs[0].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      # print("preds: \t\t\t", [emotions[emotion] for emotion in preds.squeeze().tolist()])
      # print("labels: \t\t", [emotions[emotion] for emotion in labels.argmax(dim=1).tolist()])
      output_match_list = preds.eq(labels.argmax(dim=1).view_as(preds)).squeeze().tolist()
      # print("output_match_list:\t", output_match_list)
      # print("img_names: \t",[img_name.split('\\')[-1] for img_name in img_names])

      labels_list = labels.squeeze().tolist()
      preds_list = preds.squeeze().tolist()

      for index, bool_value in enumerate(output_match_list):
        if not bool_value: # looking for misclassified
          if len(batch) == 3:
            # print(f'{batch[2][index]}, GT:{batch[1][index]}, Pred:{preds[index]}')
            list_misclassified_images.append((batch[0][index],batch[1][index],preds[index],batch[2][index]))
          else:
            list_misclassified_images.append((batch[0][index],batch[1][index],preds[index]))
          if len(list_misclassified_images) == num_image: break
      if len(list_misclassified_images) == num_image: break
  return list_misclassified_images


def plot_misclassified_images (list_misclassified_images,
                               labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] ):
  '''
  Plotting misclassified images function
  '''
  if len(list_misclassified_images) == 0: return
  

  bool_image_name_present = False
  if len(list_misclassified_images[0]) > 3:
    bool_image_name_present = True

  figure = plt.figure(figsize = (10,5))
  for index in range(1, len(list_misclassified_images) + 1):
      plt.subplot(2, int(len(list_misclassified_images)/2), index)
      plt.axis('off')
      image = np.transpose(list_misclassified_images[index-1][0], (1, 2, 0))
      # plt.imshow(list_misclassified_images[index-1][0].cpu().numpy().squeeze(), cmap='gray_r')
      plt.imshow(image, cmap='gray_r')
      GT_label = labels[torch.argmax(list_misclassified_images[index-1][1]).item()]
      Pred_Label = labels[list_misclassified_images[index-1][2].item()]
      # print(f'GT_value = {torch.argmax(list_misclassified_images[index-1][1]).item()} | Pred_value = {list_misclassified_images[index-1][2].item()}')
      if bool_image_name_present:
        file_name = list_misclassified_images[index-1][3].split(os.path.sep)[-1]
        plt.title(f'{file_name}\nGT: {GT_label} \nPred: {Pred_Label}',fontdict={'fontsize': 6})
      else:
        plt.title(f'GT: {GT_label} \nPred: {Pred_Label} ')
  plt.show()


if __name__ == '__main__':
  # print(torch.tensor([1.,0]*10).view(-1,2))
  label_matrix = torch.eye(7)
  print(label_matrix)
  # for i in range(7):
  #   print(label_matrix[i,:])
  
  # extract_zip_files('dataset\expwds','dataset\expwds')

  
