import torch

import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

import albumentations as albu_transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import numpy as np
import cv2 as cv
# from PIL import Image

import os
import glob
from os.path import splitext


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_mode, class_labellist,coco_json_filename="", expansion_data_num = 1, transform_mode = 'aug', dataset_format = 'npy', batch_size=32,img_size=[128,128], data_dir='./data', ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.img_size = img_size
        self.dataset_mode = dataset_mode
        self.dataset_format = dataset_format
        self.transform_mode = transform_mode
        self.class_labellist = class_labellist
        self.transform_method = ''
        self.expansion_data_num = expansion_data_num
        self.coco_json_filename = coco_json_filename

    def setup(self, stage=None , validation_split=0.2):        
        
        if stage is None:
            
            if( self.transform_mode == 'aug'):
                transform = albu_transforms.Compose([                    
                    albu_transforms.RandomRotate90(p = 0.7 ),
                    albu_transforms.Resize(height=self.img_size[0], width=self.img_size[1]),
                    albu_transforms.HorizontalFlip(),
                    albu_transforms.RandomResizedCrop(height=self.img_size[0], width=self.img_size[1], scale=(0.5, 1.0)),
                    # albu_transforms.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
                    albu_transforms.augmentations.transforms.ChannelShuffle(),
                    albu_transforms.augmentations.transforms.Downscale(scale_min=0.5,scale_max=0.9),
                    albu_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])    
                self.transform_method = 'albumentations'            
            else:
                transform = transforms.Compose([
                    transforms.Resize((self.img_size[0], self.img_size[1])),
                    # transforms.Grayscale(num_output_channels=3),
                    # transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                self.transform_method = 'basic' 
            
            if(self.dataset_mode == 'Classification'):
                self.dataset = CustomClassificationDataset(root=self.data_dir, expansion_data = self.expansion_data_num, transform_method = self.transform_method ,labellist=self.class_labellist, dataformat=self.dataset_format, transform=transform)
            elif(self.dataset_mode == 'ObjectDetection'):
                self.dataset = CustomObjectDetectionDataset(coco_root=self.data_dir, coco_json_filename=self.coco_json_filename ,expansion_data = self.expansion_data_num, transform_method = self.transform_method ,labellist=self.class_labellist, dataformat=self.dataset_format, transform=transform)
                        
            dataset_size = len(self.dataset)
            val_size = int(validation_split * dataset_size)
            train_size = dataset_size - val_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
       
    
class CustomClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform_method, labellist , dataformat, expansion_data = 1,transform=None):
        self.root = root
        self.transform = transform
        self.dataformat = dataformat
        self.transform_method = transform_method
        
        # 데이터셋을 불러오는 코드를 여기에 추가        
        self.labels = [splitext(file)[0] for file in os.listdir(self.root)
                    if not file.startswith('.')]
        
        self.labeldatas = []
        self.totalimages = []
        
        for ii in range(0,expansion_data):
        
            for i in range(0,len(self.labels)):
                data_list = glob.glob(self.root +'/' +self.labels[i] +'/*.'+self.dataformat)
                        
                self.totalimages = self.totalimages + data_list
                for tx in range(0,len(data_list)):
                    
                    indices = [name_idx for name_idx, name in enumerate(labellist) if name == self.labels[i]]
                    
                    self.labeldatas.append(indices[0]) 
                
        
        

    def __len__(self):
        
        return len(self.totalimages)  

    def __getitem__(self, idx):
        
        image_filepath = self.totalimages[idx]
        
        ###################### Image Load ########################
        
        if(self.dataformat == 'npy'):
            input_image = np.load(image_filepath)
        else:
            input_image = cv.imread(image_filepath)
            input_image = cv.cvtColor(input_image, cv.COLOR_RGB2BGR)
        
        label = self.labeldatas[idx]
        
        ###################### Augmentation ########################
        
        if(self.transform_method == 'basic'):
            ## normalize & permute
            tensor_image = torch.from_numpy(input_image)
            tensor_image = tensor_image.permute(2,0,1)
            tensor_image = tensor_image.float()
            tensor_image = tensor_image/255.
            
            if self.transform is not None:
                tensor_image = self.transform(tensor_image)    
        else:            
            
            input_image = np.array(input_image,np.float32)
            
            if self.transform is not None:
                temp_input_image = self.transform(image = input_image)             
                tensor_image = torch.from_numpy(temp_input_image['image'])
            else:
                tensor_image = torch.from_numpy(input_image)
            
            tensor_image = tensor_image.permute(2,0,1)
            tensor_image = tensor_image.float()
            
        # 특정 인덱스의 데이터를 반환하는 코드를 여기에 추가해야 합니다.
        return tensor_image,label


class CustomObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_root, transform_method, labellist , dataformat, coco_json_filename = "instances_default.json",expansion_data = 1, transform=None):
        self.root = coco_root
        self.transform = transform
        self.dataformat = dataformat
        self.transform_method = transform_method
        
        self.coco_format_data = COCO(coco_root+'/' +coco_json_filename)
        self.ids = list(sorted(self.coco_format_data.imgs.keys()))
        
        
        # # coco data transform 추후 depth channel 추가를 위해 놔둠.      
        # self.labels = [splitext(file)[0] for file in os.listdir(self.root)
        #             if not file.startswith('.')]
        
        # self.labeldatas = []
        # self.totalimages = []
        
        # for ii in range(0,expansion_data):
        
        #     for i in range(0,len(self.labels)):
        #         data_list = glob.glob(self.root +'/' +self.labels[i] +'/*.'+self.dataformat)
                        
        #         self.totalimages = self.totalimages + data_list
        #         for tx in range(0,len(data_list)):
                    
        #             indices = [name_idx for name_idx, name in enumerate(labellist) if name == self.labels[i]]
                    
        #             self.labeldatas.append(indices[0])

    def __len__(self):
        
        return len(self.ids)  

    def __getitem__(self, idx):
        
        coco = self.coco_format_data
        img_id = self.ids[idx]
        
        target = coco.loadAnns(img_id)
        ###################### Data Load ########################
        
        image_filepath = self.root + '/' + coco.loadImgs(img_id)[0]['file_name']
        if(self.dataformat == 'npy'):
            input_image = np.load(image_filepath)
        else:
            input_image = cv.imread(image_filepath)
            input_image = cv.cvtColor(input_image, cv.COLOR_RGB2BGR)
        
        ## Detection GT        
        # label = self.labeldatas[idx]

        ###################### Augmentation ########################
        
        if(self.transform_method == 'basic'):
            ## normalize & permute
            tensor_image = torch.from_numpy(input_image)
            tensor_image = tensor_image.permute(2,0,1)
            tensor_image = tensor_image.float()
            tensor_image = tensor_image/255.
            
            if self.transform is not None:
                tensor_image = self.transform(tensor_image)
        else:            
            
            input_image = np.array(input_image,np.float32)
            
            if self.transform is not None:
                temp_input_image = self.transform(image = input_image)             
                tensor_image = torch.from_numpy(temp_input_image['image'])
            else:
                tensor_image = torch.from_numpy(input_image)
            
            tensor_image = tensor_image.permute(2,0,1)
            tensor_image = tensor_image.float()
        
        
        bbox = [target[0]['bbox'][0],target[0]['bbox'][1],target[0]['bbox'][0]+target[0]['bbox'][2],target[0]['bbox'][1]+target[0]['bbox'][3]]
        
        output_target = [{
            'boxes': torch.as_tensor(bbox, dtype=torch.float32),
            'labels': torch.as_tensor(target[0]['category_id'], dtype =torch.int64)
            }]
        
        return tensor_image,output_target  # 
