# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


jpeg = TurboJPEG()
class IDLRWV2Dataset(Dataset):
    def __init__(self, phase, args, mode=None):
        
        self.list = []
        self.list_1 = []
        self.list_2 = []
        self.list_3 = []
        self.list_hard = []
        self.unlabel_list = []
        self.phase = phase        
        self.args = args
        self.mode = mode
        # self.num_word = 0
        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()
        

        if(not hasattr(self.args, 'is_aug')):
            setattr(self.args, 'is_aug', True)

        for (i, label) in enumerate(self.labels):


            files_1 = glob.glob(os.path.join('IDLRWV2_1', label, phase, '*.npz'))
            files_1 = sorted(files_1)
            files_2 = glob.glob(os.path.join('IDLRWV2_2', label, phase, '*.npz'))
            files_2 = sorted(files_2)
            files_3 = glob.glob(os.path.join('IDLRWV2_3', label, phase, '*.npz'))
            files_3 = sorted(files_3)
            files_hard = glob.glob(os.path.join('IDLRWV2_hard', label, phase, '*.npz'))
            files_hard = sorted(files_hard)

            

            self.list_1 += [file for file in files_1]
            self.list_2 += [file for file in files_2]
            self.list_3 += [file for file in files_3]
            self.list_hard += [file for file in files_hard]

        
    def __getitem__(self, idx):


        if(self.mode==None):
            tensor = np.load(self.list_hard[idx])                    
        elif(self.mode=='1'):
            tensor = np.load(self.list_1[idx])
        elif(self.mode=='2'):
            tensor = np.load(self.list_2[idx])
        elif(self.mode=='3'):
            tensor = np.load(self.list_3[idx])
        elif(self.mode=='hard'):
            tensor = np.load(self.list_hard[idx])
        
        inputs = np.stack(tensor["data"], 0) / 255.0 # jadi tensor 0-1
        
                
        if(self.phase == 'train'):
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
            batch_img = batch_selective_focus(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))
        
        result = {}            
        result['video'] = torch.FloatTensor(batch_img[:,np.newaxis,...])

        result['label'] = tensor.get('label')
        with open('label_sorted.txt') as myfile:
            list_labels = myfile.read().splitlines()
        for i in range(0,230):
            if list_labels[i]==result['label']:
                break
        result['label'] = torch.tensor(i)

        result['duration'] = 1.0 * tensor.get('boundary')
        
        return result

    def __len__(self):

        if self.mode==None:
            return len(self.list_hard)
        elif self.mode=='1':
            return len(self.list_1)
        elif self.mode=='2':
            return len(self.list_2)
        elif self.mode=='3':
            return len(self.list_3)
        elif self.mode=='hard':
            return len(self.list_hard)


    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = torch.zeros(25)
        mid = 25 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor     
        
