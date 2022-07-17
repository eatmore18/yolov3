from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
  """
  Takes a configuration file
  
  Returns a list of blocks. Each blocks describes a block in the neural
  network to be built. Block is represented as a dictionary in the list
  
  """
  file = open(cfgfile,'r')
  lines = file.read().split('\n')                 #read the lines
  lines = [l for l in lines if len(l) > 0]        #get rid of the empty lines
  lines = [l for l in lines if l[0] != '#']       #get rid of comments
  lines = [l.lstrip().rstrip() for l in lines]    #get rid of whitespaces

  block = {}
  blocks = []

  for line in lines:
    if line[0] == '[':
      if len(block) != 0:
        blocks.append(block)
        block = {}
      block["type"] = line[1:-1].strip()
    else:
      key, value = line.split('=')
      block[key.strip()] = value.strip()

  if len(block) != 0:
    blocks.append(block)
  
  return blocks

def create_modules(blocks):
  net_info = blocks[0]     #Captures the information about the input and pre-processing    
  module_list = nn.ModuleList()
  prev_filters = 3
  output_filters = []
  for index, x in enumerate(blocks[1:]):
    module = nn.Sequential()
    if (x["type"] == "convolutional"):
      #Get the info about the layer
      activation = x["activation"]
      try:
        batch_normalize = int(x["batch_normalize"])
        bias = False
      except:
        batch_normalize = 0
        bias = True

      filters= int(x["filters"])
      padding = int(x["pad"])
      kernel_size = int(x["size"])
      stride = int(x["stride"])

      if padding:
        pad = (kernel_size - 1) // 2
      else:
        pad = 0

      #Add the convolutional layer
      conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
      module.add_module("conv_{0}".format(index), conv)

      #Add the Batch Norm Layer
      if batch_normalize:
        bn = nn.BatchNorm2d(filters)
        module.add_module("batch_norm_{0}".format(index), bn)

      #Check the activation. 
      #It is either Linear or a Leaky ReLU for YOLO
      if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace = True)
        module.add_module("leaky_{0}".format(index), activn)

    #If it's an upsampling layer
    #We use Bilinear2dUpsampling
    elif (x["type"] == "upsample"):
      stride = int(x["stride"])
      upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
      module.add_module("upsample_{}".format(index), upsample)



