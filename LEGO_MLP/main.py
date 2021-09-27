#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:34:19 2021
Main script to build a MLP with pytorch
@author: nikorose
"""

# Imports
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets
from torch.distributions import multinomial
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
import os

#Clean cache
#torch.cuda.empty_cache()

class Net(nn.Module):
  def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num):
    super(Net, self).__init__()
    self.input_feature_num = input_feature_num # save the input size for reshapinng later
    self.mlp = nn.Sequential() # Initialize layers of MLP

    in_num = input_feature_num # initialize the temporary input feature to each layer
    for i in range(len(hidden_unit_nums)): # Loop over layers and create each one
      out_num = hidden_unit_nums[i] # assign the current layer hidden unit from list
      layer = nn.Linear(in_num, out_num) # use nn.Linear to define the layer
      in_num = out_num # assign next layer input using current layer output
      self.mlp.add_module(f"Linear_{i}", layer) # append layer to the model with a name

      actv_layer = eval(f"nn.{actv}") # Assign activation function (eval allows us to instantiate object from string)
      self.mlp.add_module(f"Activation_{i}", actv_layer) # append activation to the model with a name

    out_layer = nn.Linear(in_num, output_feature_num) # Create final layer
    self.mlp.add_module('Output_Linear', out_layer) # append the final layer

  def forward(self, x):
    # reshape inputs to (batch_size, input_feature_num)
    # just in case the input vector is not 2D, like an image!
    x = x.view(-1, self.input_feature_num)

    logits = self.mlp(x) # forward pass of MLP
    return logits

def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# inform the user if the notebook uses GPU or CPU.

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n")
  else:
    print("GPU is enabled in this notebook.")

  return device


def train_test_classification(net, criterion, optimizer, train_loader,
                              test_loader, num_epochs=1, verbose=True,
                              training_plot=False, device='cpu'):

  net.to(device)
  net.train()
  training_losses = []
  for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device).float()
      labels = labels.to(device).long()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      # print("Label size: {}".format(labels.shape))
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      if verbose:
        training_losses += [loss.item()]

  net.eval()
  def test(data_loader):
    correct = 0
    total = 0
    for data in data_loader:
      inputs, labels = data
      inputs = inputs.to(device).float()
      labels = labels.to(device).long()

      outputs = net(inputs)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return total, acc

  train_total, train_acc = test(train_loader)
  test_total, test_acc = test(test_loader)

  if verbose:
    print(f'\nAccuracy on the {train_total} training samples: {train_acc:0.2f}')
    print(f'Accuracy on the {test_total} testing samples: {test_acc:0.2f}\n')

  if training_plot:
    plt.plot(training_losses)
    plt.xlabel('Batch')
    plt.ylabel('Training loss')
    plt.show()

  return train_acc, test_acc


def shuffle_and_split_data(X, y, seed):
  # set seed for reproducibility
  torch.manual_seed(seed)
  # Number of samples
  N = X.shape[0]
  # Shuffle data
  shuffled_indices = torch.randperm(N)   # get indices to shuffle data, could use torch.randperm
  X = X[shuffled_indices]
  y = y[shuffled_indices]

  # Split data into train/test
  test_size = int(0.2 * N)    # assign test datset size using 20% of samples
  X_test = X[:test_size]
  y_test = y[:test_size]
  X_train = X[test_size:]
  y_train = y[test_size:]

  return X_test, y_test, X_train, y_train

def show_grid(train, sample=True, save=True):
    count = 0
    for image, label in train:
        fig,ax = plt.subplots(figsize = (16,10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(image,nrow=10).permute(1,2,0))
        ax.set_xlabel(label)
        print('images.shape:', image.shape)
        count +=1
        if save:
            if not os.path.isdir('pictures'):
                os.mkdir('pictures')
            os.chdir('pictures')
            fig.savefig('sample_{}.png'.format(count))
            os.chdir('../')
        if sample:
            break

def run_depth_optimizer(max_par_count, feature_size, max_hidden_layer, device):

  def count_parameters(model):
    par_count = 0
    for p in model.parameters():
      if p.requires_grad:
        par_count += p.numel()
    return par_count

  # number of hidden layers to try
  hidden_layers = range(1, max_hidden_layer+1)

  # test test score list
  test_scores = []
  K=4 # ?

  for hidden_layer in hidden_layers:
    # Initialize the hidden units in each hidden layer to be 1
    hidden_units = np.ones(hidden_layer, dtype=np.int)

    # Define the the with hidden units equal to 1
    wide_net = Net('ReLU()', feature_size, hidden_units, num_out).to(device)
    par_count = count_parameters(wide_net)

    # increment hidden_units and repeat until the par_count reaches the desired count
    while par_count < max_par_count:
      hidden_units += 1
      wide_net = Net('ReLU()', feature_size, hidden_units, K).to(device)
      par_count = count_parameters(wide_net)

    # Train it
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(wide_net.parameters(), lr=1e-3)
    _, test_acc = train_test_classification(wide_net, criterion, optimizer,
                                            train_loader, test_loader,
                                            num_epochs=50, device=device)
    test_scores += [test_acc]

  return hidden_layers, test_scores


SEED = 2021
set_seed(seed=SEED)
DEVICE = set_device()

train_data_dir = "ParcialOptativa3/Imagenes/Imagenes/train"
test_data_dir = "ParcialOptativa3/Imagenes/Imagenes/val"

_size = 150 #Change this to increase or decrease the resolution.

transform = transforms.Compose([
                                transforms.Resize((_size, _size)),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomCrop(32,padding=4),
                                #transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

train_dataset = ImageFolder(train_data_dir,
                      transform = transform)

test_dataset = ImageFolder(test_data_dir,
                      transform = transform)

#Classes
classes_train = {v: k for k, v in train_dataset.class_to_idx.items()}
classes_test = {v: k for k, v in test_dataset.class_to_idx.items()}

print("Foll0wing classes are there : ",classes_train) 

img, label = train_dataset[0]
print(f"Image Shape : {img.shape}")
train_count = len(train_dataset)
test_count = len(test_dataset)
print(f"Images in training dataset : {train_count}")
print(f"Images in test dataset : {test_count}")

# DataLoader with random seed
batch_size = 128
g_seed = torch.Generator()
g_seed.manual_seed(SEED)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0,
                         worker_init_fn=seed_worker,
                         generator=g_seed,
                         )

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          drop_last=True,
                          shuffle=True,
                          worker_init_fn=seed_worker,
                          generator=g_seed,
                          )

#Printing one batch sample
show_grid(train_loader)
num_out = 4

set_seed(seed=SEED)
max_par_count = 100
max_hidden_layer = 5

## Uncomment below to test your function
feature_size = 3*_size**2
hidden_layers, test_scores = run_depth_optimizer(max_par_count, feature_size,
                                                 max_hidden_layer, DEVICE)

plt.xlabel('# of hidden layers')
plt.ylabel('Test accuracy')
plt.xticks(range(5))
plt.grid(True)
plt.suptitle('General test score in a range of Hidden Layers')
plt.plot(hidden_layers, test_scores)
plt.show()
plt.savefig('pictures/opt_hidden.png')