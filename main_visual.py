import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim 
import random
import pdb
import shutil
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter 
from focal_loss.focal_loss import FocalLoss
from lion_pytorch import Lion
import torch_optimizer as optim2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from pytorch_model_summary import summary
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from torchvision.ops import sigmoid_focal_loss 
from calflops import calculate_flops   
# import smote_variants as sv
from sophia import SophiaG
import timm.scheduler
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()


writer = SummaryWriter()
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
g = torch.Generator()
g.manual_seed(42)

def get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, power=1):
    weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

def get_weights_effective_num_of_samples(no_of_classes, beta, samples_per_cls):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights_for_samples = (1.0 - beta) / np.array(effective_num)
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

def get_weights_transformed_for_sample(sample_weighting_method, no_of_classes, samples_per_cls, b_labels, beta=None):
    if sample_weighting_method == 'ens':
        weights_for_samples = get_weights_effective_num_of_samples(no_of_classes, beta, samples_per_cls)
    elif sample_weighting_method == 'ins':
        weights_for_samples = get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls)
    elif sample_weighting_method == 'isns':
        weights_for_samples = get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, 0.5)
    else:
        return None
    return weights_for_samples


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--test', type=str2bool, required=True)

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True)

# dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--border', type=str2bool, required=True)
parser.add_argument('--mixup', type=str2bool, required=True)
parser.add_argument('--label_smooth', type=str2bool, required=True)
parser.add_argument('--se', type=str2bool, required=True)
parser.add_argument('--focal_loss', type=str2bool, required=True)
parser.add_argument('--focal_loss_weight', type=str2bool, required=True)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if(args.dataset == 'lrw'):
    from utils import LRWDataset as Dataset
elif(args.dataset == 'lrw1000'):    
    from utils import LRW1000_Dataset as Dataset
elif(args.dataset == 'idlrw'):    
    from utils import IDLRWDataset as Dataset
elif(args.dataset == 'idlrwv2'):    
    from utils import IDLRWV2Dataset as Dataset
else:
    raise Exception('lrw or lrw1000 or idlrw')    


video_model = VideoModel(args).cuda()
param_size = 0
for param in video_model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in video_model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

def parallel_model(model):
    model = nn.DataParallel(model)
    return model        


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
    
if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))    
    load_missing(video_model, weight.get('video_model'))
    
    
video_model = parallel_model(video_model)

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader =  DataLoader(dataset,
            batch_size = batch_size, 
            num_workers = num_workers,   
            shuffle = shuffle,         
            drop_last = False,
            pin_memory=True,
            generator=g)
    return loader

def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg    

def calculate_class_weights(labels):
    # Count the frequency of each class
    class_counts = np.bincount(labels)
    
    # Calculate the weight for each class
    # The weight is the inverse of the class frequency
    total_count = np.sum(class_counts)
    class_weights = total_count / class_counts
    
    return torch.FloatTensor(class_weights).cuda()

def val(mode_var=None):
    
    with torch.no_grad():
        dataset = Dataset('val', args, mode=mode_var)

        with open('label_sorted.txt') as myfile:
            temp_labels = myfile.read().splitlines()
        args.n_class = len(temp_labels)


        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=True)        
        

        v_acc = []
        entropy = []
        acc_mean = []
        total = 0
        cons_acc = 0.0
        cons_total = 0.0
        attns = []

        for (i_iter, input) in enumerate(loader):
            
            video_model.eval()
            
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)
            names = input.get('name')
            border = input.get('duration').cuda(non_blocking=True).float()
            
            with autocast():
                if(args.border):
                    y_v = video_model(video, border)                                           
                else:
                    y_v = video_model(video)                                           
                                

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if(i_iter % 10 == 0):  
                msg = ''              
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())                
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(loader)-i_iter)/3600.0)
                                
 

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)
        
        return acc, msg      


def test():
    
    with torch.no_grad():
        dataset = Dataset('test', args)
        print('Start Testing, Data Length:',len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=True)        
        
        true_labels = []
        pred_labels = []

        v_acc = []
        entropy = []
        acc_mean = []
        total = 0
        cons_acc = 0.0
        cons_total = 0.0
        attns = []

        all_probs = []
        video_names = []


        for (i_iter, input) in enumerate(loader):
            
            video_model.eval()
            
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)

            total = total + video.size(0)

            border = input.get('duration').cuda(non_blocking=True).float()

            with autocast():
                if(args.border):
                    y_v = video_model(video, border)
                    logits = video_model(video, border)                                           
                else:
                    y_v = video_model(video)   
                    logits = video_model(video)                                        

            # Calculate softmax probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            names_cpu = [name_item.cpu().numpy() for name_item in label]  # Convert each tensor to CPU
            names_str = [str(name_item) for name_item in names_cpu]       # Convert each numpy array to string
            video_names.extend(names_str)



            true_labels.extend(label.cpu().numpy().tolist())
            pred_labels.extend(y_v.argmax(-1).cpu().numpy().tolist())

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if(i_iter % 10 == 0):  
                msg = ''              
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())                
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(loader)-i_iter)/3600.0)
                                
                print(msg)           
            input_fvcore = video,border

        
        df_probs = pd.DataFrame(all_probs)
        df_probs['video_name'] = video_names
        df_probs['true_label'] = true_labels
        df_probs['pred_label'] = pred_labels

        # Save to CSV
        df_probs.to_csv('video_class_probabilities.csv', index=False)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)

        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')
        f1 = f1_score(true_labels, pred_labels, average='macro')
        pytorch_total_params = sum(p.numel() for p in video_model.parameters())

        print(f"Total number of parameters: {pytorch_total_params/1e6}")
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        return acc, msg                                 

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def train():            


    if (args.focal_loss_weight):
        print("Calculating class weights...")
        train_labels = []
        total_batches = len(loader)
        start_time = time.time()
        for i, batch in enumerate(loader):
            labels = batch.get('label')  # Assumes that 'label' is the key for the labels
            train_labels.extend(labels.tolist())

            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (i + 1)) * (total_batches - (i + 1))
            
            print(f"Processed {i + 1}/{total_batches} batches. ETA: {eta:.2f} seconds.")        


        train_labels = np.array(train_labels)
        weights = calculate_class_weights(train_labels)
    else:
        weights = None



    max_epoch = args.max_epoch    

    
    ce = nn.CrossEntropyLoss()

    tot_iter = 0
    # best_acc = 0.0
    best_before = 0.0
    adjust_lr_count = 0
    alpha = 0.2
    epsilon = 1e-7
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    scaler = GradScaler()           

    # mode_list = ['easy', 'medium', 'hard']

    mode_list = [None]
    patience = 10
    temporary_patience = 0
    chk_epoch = 0
    lr_tune = args.lr
    for mode_var in mode_list:

        # lr = args.lr/30
        # optim_video = Lion(video_model.parameters(), lr = lr, weight_decay=1e-3)
        lr = args.lr
        optim_video = optim2.AdaBound(
           video_model.parameters(),
           lr= lr,
           betas= (0.9, 0.999),
           final_lr = 0.1,
           gamma=1e-3,
           eps= 1e-8,
           weight_decay=1e-4,
           # weight_decay=1e-2,
           amsbound=False,
       )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max = args.max_epoch, eta_min=5e-6)


        dataset = Dataset('train', args, mode=mode_var)

        with open('label_sorted.txt') as myfile:
            temp_labels = myfile.read().splitlines()
        args.n_class = len(temp_labels)
        # print(args.n_class)
        
        best_acc = 0.0

        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=True)
        print('Start Training, Data Length:',len(dataset),"Mode:",str(mode_var),"Number of Classes:",str(args.n_class))

        for epoch in range(0, max_epoch):

            total = 0.0
            v_acc = 0.0
            total = 0.0               
            
            lsr = LSR()
            train_acc = []
            train_loss = []
            # softmax for focal loss
            if (args.focal_loss):
                m = nn.Softmax(dim=-1)

            for (i_iter, input) in enumerate(tqdm(loader)):
                tic = time.time()           
                
                video_model.train()

                video = input.get('video').cuda(non_blocking=True)
                label = input.get('label').cuda(non_blocking=True).long()  
                if args.border:   
                    border = input.get('duration').cuda(non_blocking=True).float()
                
                loss = {}
                
                if(args.focal_loss):
                    var_gamma = 2
                    if (args.focal_loss_weight):
                        loss_fn = FocalLoss(gamma=var_gamma, weights=weights)

                    else:
                        loss_fn = FocalLoss(gamma=var_gamma)
                    label_print = "focalloss"
                else:
                    if(args.label_smooth):
                        loss_fn = lsr
                        label_print = "lsr"
                    else:
                        loss_fn = nn.CrossEntropyLoss()
                        label_print = "CE"
                with autocast():
                    if(args.mixup):
                        lambda_ = np.random.beta(alpha, alpha)
                        index = torch.randperm(video.size(0)).cuda(non_blocking=True)
                        
                        mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                        if(args.border):
                            mix_border = lambda_ * border + (1 - lambda_) * border[index, :]
                            
                        label_a, label_b = label, label[index]            
                        if(args.border):
                            y_v = video_model(mix_video, mix_border)       
                        else:                
                            y_v = video_model(mix_video)       

                        
                        if args.focal_loss:
                            y_v = m(y_v)

                        try:
                            loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)
                        except AssertionError:
                            print("Error during loss calculation:", e)
                            print("y_v:", y_v)
                            print("label_a:", label_a)
                            print("label_b:", label_b)

                        var_acc = (y_v.argmax(-1) == label).cpu().numpy().tolist()
                        var_acc = float(np.array(var_acc).reshape(-1).mean())
                        var_acc = '{:.5f}'.format(var_acc)

                    else:
                        if(args.border):
                            y_v = video_model(video, border)       
                        else:                
                            y_v = video_model(video)    
                            
                        if args.focal_loss:
                            loss_bp = loss_fn(m(y_v), label)
                        else:
                            loss_bp = loss_fn(y_v, label)



                        var_acc = (y_v.argmax(-1) == label).cpu().numpy().tolist()
                        var_acc = float(np.array(var_acc).reshape(-1).mean())
                        var_acc = '{:.5f}'.format(var_acc)
                                        
                loss[label_print] = loss_bp

                train_acc.append(float(var_acc))
                train_loss.append(float(loss_bp.detach().cpu().numpy()))
 

                optim_video.zero_grad()   
                torch.nn.utils.clip_grad_norm_(video_model.parameters(), max_norm=1.0)
                scaler.scale(loss_bp).backward()  
                scaler.step(optim_video)
                scaler.update()
                
                toc = time.time()
                
                msg = 'mode={},epoch={},train_iter={},eta={:.5f}'.format(mode_var,epoch, tot_iter, (toc-tic)*(len(loader)-i_iter)/3600.0)
                for k, v in loss.items():                                                
                    msg += ',{}={:.5f}'.format(k, v)
                msg = msg + str(',lr=' + str(showLR(optim_video))) 
                msg = msg + str(',train_acc=' + str(var_acc))                   
                msg = msg + str(',best_val_acc={:2f}'.format(best_acc))
                           
                msg_last = msg
                if(i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0)):

                    acc, msg = val(mode_var)
                    writer.add_scalar("Acc/val", acc, tot_iter)
                    mean_train_acc = np.average(train_acc)
                    mean_train_loss = np.average(train_loss)
                    writer.add_scalar("Acc/train", mean_train_acc, tot_iter)
                    writer.add_scalar("Loss/train", mean_train_loss, tot_iter)



                    if(acc > best_acc):
                        savename = '{}_mode_{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, mode_var, tot_iter, epoch, msg)
                        # savename2 = "best_curriculum_learning_"+mode_var+".pt"
                        savename2 = "best_abl.pt"
                        temp = os.path.split(savename)[0]
                        if(not os.path.exists(temp)):
                            os.makedirs(temp)                    
                        torch.save(
                            {
                                'video_model': video_model.module.state_dict(),
                            }, savename2)         

                    if(tot_iter != 0):
                        best_before = best_acc
                        best_acc = max(acc, best_acc)    
                        
                tot_iter += 1        

            scheduler.step(epoch+1)   

            print("mode:",mode_var," epoch:",epoch," acc:",acc," best_acc:",best_acc," best_before:",best_before," temporary_patience:",temporary_patience," chk_epoch:",chk_epoch)

            if(acc <= best_before):
                if epoch >= chk_epoch:
                    temporary_patience += 1
                    if(temporary_patience > patience):
                        chk_epoch = epoch 
                        temporary_patience = 0
                        break
            else:
                temporary_patience = 0

        if chk_epoch == 0:
            chk_epoch = max_epoch
        
         
        
if(__name__ == '__main__'):
    print("Start")
    if(args.test):
        acc, msg = test()
        print(f'acc={acc}')
        writer.add_scalar("Acc/test", acc)
        exit()
    train()
