import numpy as np
import sys
import time
import os
import json
from easydict import EasyDict as edict
from natsort import natsorted

class TimeCounter:
    # Create an time counter.
    # To count the rest time.

    # Input the total times.
    def __init__(self, total):
      self.total = total
      self.cur = 0
      self.begin = time.time()

    def step(self):
      end = time.time() 
      self.cur += 1
      used = (end - self.begin)/self.cur
      rest = self.total - self.cur

      return np.max(rest * used, 0)
         

def readfolder(data, specific=None, reverse=False):

    
    folders = os.listdir(data.label)
    folders = natsorted(folders) 

    if specific is not None:
        
        if isinstance(specific, int):
            if specific < 0 or specific >= len(folders):
                raise ValueError(f"Specific index {specific} out of range (0 ~ {len(folders)-1}).")
            if reverse:
                folder = [f for i, f in enumerate(folders) if i != specific]
            else:
                folder = [folders[specific]]
        elif isinstance(specific, list) and all(isinstance(x, int) for x in specific):
            if reverse:
                exclude = set(specific)
                folder = [f for i, f in enumerate(folders) if i not in exclude]
            else:
                folder = [folders[i] for i in specific if 0 <= i < len(folders)]
        else:
           
            if isinstance(specific, list):
                specific = set(specific)
            else:
                specific = {specific}
            if reverse:
                folder = [f for f in folders if f not in specific]
            else:
                folder = [f for f in folders if f in specific]
    else:
        folder = folders

    
    data.label = [os.path.join(data.label, j) for j in folder]
    return data, folder

def DictDumps(content):
    return json.dumps(content,  ensure_ascii=False, indent=4)



def GetLR(optimizer):
    LR = optimizer.state_dict()['param_groups'][0]['lr']
    return LR

