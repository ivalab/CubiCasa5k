#! /usr/bin/python

import matplotlib
matplotlib.use('TkAgg', force=True)     # set MPL's GUI back-end

import matplotlib.pyplot as plt

import sys
import os
import math
import csv

import argparse

import torch
import cv2
import numpy as np
import pandas as pd

from datetime import datetime
from torchvision.transforms import transforms

from floortrans.loaders import FloorplanSVG

from skimage import measure
from sklearn.decomposition import PCA


def export_doors(args):
  # Arguments
  data_path = args.data_path
  text_file = args.f
  write_label_mask = args.write_label_mask
  write_door_props = args.write_door_props
  debug = args.debug
  label_id_select = args.label_id
  
  output_mask_file = '/F1_scaled_label_mask.png'
  output_label_props_file = '/label_props.csv'

  # Load image + ground truth segmentation
  train_set = FloorplanSVG(data_path, text_file, format='txt',
                            augmentations=None, is_transform=False)

  for jj in range(len(train_set)):
    samp = train_set[jj]
    
    print()
    print(f'==== Processing sample {jj}/{len(train_set)} ====')
    print(f'Sample folder: ', {samp['folder']})
    print(f'Image size: ', {samp['image'].size()})
    print()

    label = samp['label']
    
    # get label mask
    label_img = label[1]    # 0-11 (??)
    label_sel_bin = (label_img == label_id_select)

    ## Format for output processing
    if ( write_label_mask or write_door_props ):
      label_sel_bin = label_sel_bin.numpy().astype(np.uint8)

    ## Write door mask to file
    if ( write_label_mask ):
      print(f'  ### Writing label mask (label ID: {label_id_select}) ...')
      print()
      cv2.imwrite(data_path + samp['folder'] + output_mask_file, label_sel_bin*255)
    
    ## Find, write door properties
    #   Door props:
    #     (1) door centroid (px coordinate)
    #     (2) door length (px)
    if ( write_door_props ):
      # Open CSV file for write
      csv_filename = data_path + samp['folder'] + output_label_props_file
      csvfile = open(csv_filename, 'w', newline='')
      writer = csv.writer(csvfile)
      csvdata = [["CoordX", "CoordY", "Orient", "HalfLen", "HalfWid"]]
      writer.writerows(csvdata)    # record header
      
      # Extract connected regions -> parse region properties
      relabeled_img = measure.label(label_sel_bin)
      num_regions = relabeled_img.max()
      
      for ii in range(1, num_regions+1):
        reg_inds = np.where(relabeled_img == ii)
        reg_ii_centroid = np.array([np.mean(reg_inds[1]), np.mean(reg_inds[0])]) # [col, row] -> [x, y] (image space coord. frame)
        
        # PCA fit to region coordinates
        reg_coords = np.array([reg_inds[1], reg_inds[0]])  # [col, row] -> [x, y] (image space coord. frame)
      
        pca = PCA(n_components=2)  # Specify the number of components you want to keep
        princ_comps = pca.fit(reg_coords.transpose())
        pca_vals = pca.singular_values_
        pca_dirs = pca.components_

        # Orientation associated to highest variance
        if ( pca_vals[0] >= pca_vals[1] ):
          major_dir = pca_dirs[0, :]
          minor_dir = pca_dirs[1, :]
        else:
          major_dir = pca_dirs[1, :]
          minor_dir = pca_dirs[0, :]
        reg_ii_orient = math.atan2(major_dir[1], major_dir[0])
          
        # Door gap pixel coordinates (zero'd w.r.t. gap centroid) 
        rel_coords = np.zeros_like(reg_coords)
        rel_coords[0] = reg_coords[0] - reg_ii_centroid[0]
        rel_coords[1] = reg_coords[1] - reg_ii_centroid[1]

        # Door gap half-LENGTH via projection 
        proj_lens = np.matmul(rel_coords.transpose(),major_dir)
        reg_ii_halflen = np.max(np.abs(proj_lens))

        # Door gap half-WIDTH via projection 
        proj_wids = np.matmul(rel_coords.transpose(),minor_dir)
        reg_ii_halfwid = np.max(np.abs(proj_wids))
        
        
        # Write data CSV file: [["CoordX", "CoordY", "Orient", "HalfLen"]]
        print(f'  ### Writing label region props (label ID: {label_id_select})')
        print(f'    Centroid: [{reg_ii_centroid[0]}, {reg_ii_centroid[1]}], Orientation: {reg_ii_orient*180/math.pi} deg, Half-length: {reg_ii_halflen}, Half-width: {reg_ii_halfwid}')
        print()
        csvdata = [[reg_ii_centroid[0], reg_ii_centroid[1], reg_ii_orient, reg_ii_halflen, reg_ii_halfwid]]
        writer.writerows(csvdata)
                     
    ## [debug] Show sample image + selected label
    if ( debug ):
      image = samp['image'].permute(1, 2, 0)    # original (scaled) image

      # grayscale sample image
      gs_transform = transforms.Grayscale()
      gs_img = gs_transform(samp['image']).permute(1, 2, 0) # re-arrange to (row, col, RGB channel)
      gs_img = gs_img[:, :, 0]    # 0-255

      # overlay labeled image
      label_sel_img = (label_img == label_id_select).to(torch.float64)*11.   # arbitrary color value for Doors pixels
      
      clr_img = torch.zeros(image.size(), dtype=torch.float64)    # empty rgb image

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

      plt.subplot(122)
      plt.imshow(np.array(label_img))

      plt.show()


    

if __name__ == '__main__':
  time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
  parser = argparse.ArgumentParser(description='Hyperparameters')
  parser.add_argument('--data-path', nargs='?', type=str, default='data/',
            help='Path to data directory')
  parser.add_argument('--f', nargs='?', type=str, default='all.txt',
            help='Floorplan list text file')
  parser.add_argument('--write-label-mask', nargs='?', type=bool, default=True,
            help='Write label mask to data directory')
  parser.add_argument('--write-door-props', nargs='?', type=bool, default=True,
            help='Write door properties to file')
  parser.add_argument('--label-id', nargs='?', type=int, default=2,
            help='Label ID to filter')
  parser.add_argument('--debug', nargs='?', type=bool, default=False,
            help='Plot sample images')
  args = parser.parse_args()

  export_doors(args)
  
  
# Example usage:
# ./export_doors.py --data-path data/ --f all.txt
