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
  
        
