import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict 
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import Dataset, DataLoader, Subset
import glob
from PIL import Image
import torch
import numpy as np
import random
np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)


root = './data'
path2jpg = os.path.join(root,'train_jpg')
path2jpg_test = os.path.join(root,'test_jpg') 

def get_vids(path2jpg):
    listOfCats = os.listdir(path2jpg)
    ids = []
    labels = []
    for cat in listOfCats:
        path2cat = os.path.join(path2jpg, cat)
        listOfSubCats = os.listdir(path2cat)
        path2subCats= [os.path.join(path2cat,los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([cat]*len(listOfSubCats))
    return ids, labels, listOfCats 


def get_train_test(phase ='train'):
    all_vids, all_labels, cats = get_vids(path2jpg)
    dic = defaultdict(int)
    i=0
    for cat in cats:
        dic[cat]=i
        i+=1    
    n_classes = 5
    
    if phase =='train':
        unique_vids = [vid for vid, label in zip(all_vids,all_labels) if dic[label]<n_classes]
        unique_labels = [label for vid, label in zip(all_vids,all_labels) if dic[label]<n_classes]

        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
        train_indx, test_indx = next(sss.split(unique_vids, unique_labels))

        train_ids = [unique_vids[ind] for ind in train_indx]
        train_labels = [unique_labels[ind] for ind in train_indx]
        print(len(train_ids), len(train_labels)) 
        test_ids = [unique_vids[ind] for ind in test_indx]
        test_labels = [unique_labels[ind] for ind in test_indx]
        print(len(test_ids), len(test_labels))
        return(train_ids, train_labels, test_ids, test_labels, dic)
    
    elif phase =='test':
        all_vids_test, all_labels_test, _ = get_vids(path2jpg_test)
        unique_vids_test = [vid for vid, label in zip(all_vids_test,all_labels_test) if dic[label]<n_classes]
        unique_labels_test = [label for vid, label in zip(all_vids_test,all_labels_test) if dic[label]<n_classes]
        return(unique_vids_test, unique_labels_test, dic)
        

class VideoDataset(Dataset):
    def __init__(self, vids, labels, transform, dic):      
        self.transform = transform
        self.vids = vids
        self.labels = labels
        self.dic = dic
    
    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, idx):
        # It captures all images in the vids[idx]
        path2imgs=glob.glob(self.vids[idx]+"/*.jpg")
        label = self.dic[self.labels[idx]]
        frames = []
        for img in path2imgs:
            frame = Image.open(img)
            frames.append(frame)
        
        seed = np.random.randint(1e9)        
        frames_tr = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        return frames_tr, label


def get_transform(model_type, phase):
    
    if model_type == "cnn_rnn":
        h, w =224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_type == "3dcnn":
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        
    if phase=='train':
        transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
    elif phase=='test':
        transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ]) 
    return(transformer)
    
def collate_fn_r3d_18(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs[:16] if len(imgs) >= 16 else imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

def collate_fn_cnn_rnn(batch):
    imgs_batch, label_batch = list(zip(*batch))    
    imgs_batch = [imgs[:16] if len(imgs) >= 16 else imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor









