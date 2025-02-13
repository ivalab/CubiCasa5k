#! /usr/bin/python

import matplotlib
matplotlib.use('pdf')

import sys
import os

import argparse

import torch
import numpy as np
import pandas as pd
from datetime import datetime

from torchvision.transforms import transforms
from torch.utils import data
from tqdm import tqdm

from floortrans.loaders import FloorplanSVG

import matplotlib.pyplot as plt
matplotlib.use('TkAgg', force=True)     # set MPL's GUI back-end

def show_sample_doors(args):
  ind = args.i
  data_path = args.data_path
  text_file = args.f

  file_hdl = open(data_path + "/" + text_file, "r")
  floorplan_list = file_hdl.readlines()
  file_hdl.close()

  # Setup Dataloader
  train_set = FloorplanSVG(args.data_path, text_file, format='txt',
                            augmentations=None, is_transform=False)

  sel_samp = floorplan_list[ind]
  samp = train_set[ind]
  print(f'sample image size: ', {samp['image'].size()})
  print(f'sample label size: ', {samp['label'].size()})

  print(f'sample image type: ', {type(samp['image'])})
  print(f'sample label type: ', {type(samp['label'])})
  
  print(f'sample image max: ', {samp['image'].max()})
  print(f'sample image min: ', {samp['image'].min()})
  print(f'sample image mean: ', {samp['image'].mean()})

  print(f'sample label max: ', {samp['label'].max()})
  print(f'sample label min: ', {samp['label'].min()})
  print(f'sample label mean: ', {samp['label'].mean()})

  image = samp['image'].permute(1, 2, 0)
  label = samp['label'] # layer 0 -> regions; layer 1 -> icons
  
  # grayscale sample image
  gs_transform = transforms.Grayscale()
  gs_image = gs_transform(samp['image']).permute(1, 2, 0) # re-arrange to (row, col, RGB channel)

  # overlaid labeled image
  gs_img = gs_image[:, :, 0]    # 0-255
  label_img = label[1]    # 0-11 (??)

  label_select = 2    # house.py -> Doors icon id
  label_sel_img = (label_img == 2)
  label_sel_img = label_sel_img.to(torch.float64)*11.   # arbitrary color value for Doors pixels
  
  clr_img = torch.zeros(image.size(), dtype=torch.float64)

  clr_img[:, :, 0] = gs_img
  clr_img[:, :, 1] = gs_img
  clr_img[:, :, 2] = gs_img

  clr_img[:, :, 1] -= label_sel_img*255./15.
  clr_img[:, :, 2] -= label_sel_img*255./15.

  clr_img = torch.clamp(clr_img, min=0.)    # 0-255
  print(f'clr_img size: ', {clr_img.size()})
	
  # Display label-overlaid-on-original image + label image
  plt.figure()
  
  plt.subplot(121)
  plt.imshow(np.array(clr_img/255.))
#  plt.imshow(np.array(gs_image[:, :, 0]/255.))
  plt.title("[Index: " + str(ind) + "] Floorplan: " + sel_samp)
  plt.axis('equal')

  plt.subplot(122)
  plt.imshow(np.array(label_img))
  plt.axis('equal')

  plt.show()



if __name__ == '__main__':
  time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
  parser = argparse.ArgumentParser(description='Hyperparameters')
  parser.add_argument('--data-path', nargs='?', type=str, default='data/',
            help='Path to data directory')
  parser.add_argument('--f', nargs='?', type=str, default='train.txt',
            help='Floorplan list text file')
  parser.add_argument('--i', nargs='?', type=int, default=0,
            help='Sample index')
  args = parser.parse_args()

  show_sample_doors(args)
  
  
# Example usage:
# ./show_sample.py --data-path data/ --f train.txt --i 5
  
