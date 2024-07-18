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
class IDLRWDataset(Dataset):
    def __init__(self, phase, args, mode=None):
        
        self.list = []
        self.list_easy = []
        self.list_medium = []
        self.list_hard = []
        self.unlabel_list = []
        self.phase = phase        
        self.args = args
        self.mode = mode
        # self.num_word = 0
        if self.mode==None:
            with open('label_sorted.txt') as myfile:
                self.labels = myfile.read().splitlines()
        elif self.mode=='easy':
            with open('label_sorted_easy.txt') as myfile:
                self.labels = myfile.read().splitlines()
        elif self.mode=='medium':
            with open('label_sorted_medium.txt') as myfile:
                self.labels = myfile.read().splitlines()
        elif self.mode=='hard':
            with open('label_sorted_hard.txt') as myfile:
                self.labels = myfile.read().splitlines()
        # self.num_word = len(self.labels)

        if(not hasattr(self.args, 'is_aug')):
            setattr(self.args, 'is_aug', True)

        for (i, label) in enumerate(self.labels):
            # default
            # files = glob.glob(os.path.join('lrw_roi_80_116_175_211_npy_gray_pkl_jpeg', label, phase, '*.pkl'))                    
            
            # idlrw
            # files = glob.glob(os.path.join('idlrw_roi_64_160_64_160_npy_gray_pkl_jpeg', label, phase, '*.pkl'))                    
            # files = glob.glob(os.path.join('idlrwv1_roi_64_160_64_160_npy_gray_pkl_jpeg_with_border_hard', label, phase, '*.npz'))                    

            # idlrwv1
            # files = glob.glob(os.path.join('idlrwv1_roi_64_160_64_160_npy_gray_pkl_jpeg', label, phase, '*.pkl'))                    
            # files = glob.glob(os.path.join('idlrwv1_roi_64_160_64_160_npy_gray_pkl_jpeg_with_border', label, phase, '*.pkl'))                    
            # files = sorted(files)

            # files_easy = glob.glob(os.path.join('idlrwv1_roi_64_160_64_160_npy_gray_pkl_jpeg_with_border_easy', label, phase, '*.npz'))
            # files_easy = sorted(files_easy)
            # files_medium = glob.glob(os.path.join('idlrwv1_roi_64_160_64_160_npy_gray_pkl_jpeg_with_border_medium', label, phase, '*.npz'))
            # files_medium = sorted(files_medium)
            files_hard = glob.glob(os.path.join('IDLRWv1_norm', label, phase, '*.npz'))
            files_hard = sorted(files_hard)

            # files_easy = glob.glob(os.path.join('300_IDLRW_easy', label, phase, '*.npz'))
            # files_easy = sorted(files_easy)
            # files_medium = glob.glob(os.path.join('300_IDLRW_medium', label, phase, '*.npz'))
            # files_medium = sorted(files_medium)
            # files_hard = glob.glob(os.path.join('300_IDLRW_hard', label, phase, '*.npz'))
            # files_hard = sorted(files_hard)

            # self.list += [file for file in files]
            # print(len(self.list))
            # self.list_easy += [file for file in files_easy]

            # self.list_medium += [file for file in files_easy]
            # self.list_medium += [file for file in files_medium]
            # 
            # self.list_hard += [file for file in files_easy]
            # self.list_hard += [file for file in files_medium]
            self.list_hard += [file for file in files_hard]
        # print(len(self.list))
        
    def __getitem__(self, idx):

        
        # if(self.mode==None):
        #     tensor = np.load(self.list[idx])                    
        # elif(self.mode=='easy'):
        #     tensor = np.load(self.list_easy[idx])
        # elif(self.mode=='medium'):
        #     tensor = np.load(self.list_medium[idx])
        # elif(self.mode=='hard'):
        #     tensor = np.load(self.list_hard[idx])

        if(self.mode==None):
            tensor = np.load(self.list_hard[idx])                    
        elif(self.mode=='easy'):
            tensor = np.load(self.list_easy[idx])
        elif(self.mode=='medium'):
            tensor = np.load(self.list_medium[idx])
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
        #print(result['video'].size())
        result['label'] = tensor.get('label')
        with open('label_sorted.txt') as myfile:
            list_labels = myfile.read().splitlines()
        for i in range(0,100):
            if list_labels[i]==result['label']:
                break
        result['label'] = torch.tensor(i)
        #sementara karena tidak pakai duration
        result['duration'] = 1.0 * tensor.get('boundary')
        
        return result

    def __len__(self):
        # print(self.mode)
        # if self.mode==None:
        #     return len(self.list)
        # elif self.mode=='easy':
        #     return len(self.list_easy)
        # elif self.mode=='medium':
        #     return len(self.list_medium)
        # elif self.mode=='hard':
        #     return len(self.list_hard)
        if self.mode==None:
            return len(self.list_hard)
        elif self.mode=='easy':
            return len(self.list_easy)
        elif self.mode=='medium':
            return len(self.list_medium)
        elif self.mode=='hard':
            return len(self.list_hard)
        # return len(self.list)

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
        