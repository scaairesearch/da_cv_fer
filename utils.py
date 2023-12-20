
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

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

if __name__ == '__main__':
  print(torch.tensor([1.,0]*10).view(-1,2))
  label_matrix = torch.eye(7)
  print(label_matrix)
  for i in range(7):
    print(label_matrix[i,:])
