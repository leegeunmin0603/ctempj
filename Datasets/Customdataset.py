import torch

import torchvision.transforms.v2 as transforms
# from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

import albumentations as albu_transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

import os
import glob
from os.path import splitext




class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_mode, class_labellist,coco_json_filename="", expansion_data_num = 1, transform_mode = 'aug', dataset_format = 'npy', batch_size=32,img_size=[480,480], data_dir='./data', ):
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
                    albu_transforms.RandomRotate90(p = 0.5),
                    albu_transforms.Resize(height=self.img_size[0], width=self.img_size[1]),
                    albu_transforms.HorizontalFlip(p =0.5),
                    albu_transforms.RandomResizedCrop(height=self.img_size[0], width=self.img_size[1], scale=(0.5, 0.99)),
                    # albu_transforms.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
                    albu_transforms.augmentations.transforms.ChannelShuffle(),
                    albu_transforms.augmentations.transforms.Downscale(scale_min=0.5,scale_max=0.9),
                    # albu_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ],bbox_params=albu_transforms.BboxParams(format='pascal_voc', min_area= 1000, label_fields=['class_labels'])) 
                self.transform_method = 'albumentations'            
            else:
                transform = transforms.Compose([
                    # transforms.Resize((self.img_size[0], self.img_size[1])),
                    # transforms.Grayscale(num_output_channels=3),
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.val_dataset.dataset.collate_fn)
    


class CustomObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_root, transform_method, labellist , dataformat, coco_json_filename = "instances_default.json",expansion_data = 1, transform=None):
        self.root = coco_root
        self.transform = transform
        self.dataformat = dataformat
        self.transform_method = transform_method
        
        self.coco_format_data = COCO(coco_root+'/' +coco_json_filename)
        self.ids = list(sorted(self.coco_format_data.imgs.keys()))
        
        
        
    def __len__(self):
        
        return len(self.ids)  

    def collate_fn(self, batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # 이미지의 크기가 다를 수 있으므로 가장 큰 크기로 맞추어줍니다.
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        
        # 이미지를 패딩하여 동일한 크기로 만듭니다.
        images_padded = torch.zeros(len(images), 3, max_size[1], max_size[2])
        for i, img in enumerate(images):
            images_padded[i, :, :img.shape[1], :img.shape[2]] = img
        
        # 각 타겟 정보를 리스트에서 텐서로 변환합니다.
        targets = [{'boxes': t['boxes'], 'labels': t['labels']} for t in targets]

        return images_padded, targets
    
    def __getitem__(self, idx):
        
        coco = self.coco_format_data
        img_id = self.ids[idx]
        
        ann_ids = coco.getAnnIds(imgIds=img_id) #img id, category id를 받아서 해당하는 annotation id 반환        
        target = coco.loadAnns(ann_ids)
        ###################### Data Load ########################
        
        image_filepath = self.root + '/' + coco.loadImgs(img_id)[0]['file_name']
        if(self.dataformat == 'npy'):
            input_image = np.load(image_filepath)
        else:
            input_image = cv.imread(image_filepath)
            input_image = cv.cvtColor(input_image, cv.COLOR_RGB2BGR)
        
        ## Detection GT

        bbox = []
        labels = []
        
        for index in target:
            
            bbox.append([index['bbox'][0],index['bbox'][1],index['bbox'][0]+index['bbox'][2],index['bbox'][1]+index['bbox'][3]])
            labels.append(index['category_id'])
        
        
        
        ###################### Augmentation ########################
        
        if(self.transform_method == 'basic'):
            ## normalize & permute
            tensor_image = torch.from_numpy(input_image)
            tensor_image = tensor_image.permute(2,0,1)
            tensor_image = tensor_image.float()
            tensor_image = tensor_image/255.
            # tensor_image = tensor_image.unsqueeze(0)
            
            if self.transform is not None:
                tensor_image = self.transform(tensor_image)
        else:            
            
            input_image = np.array(input_image,np.float32)
            
            if self.transform is not None:
                transformed = self.transform(image = input_image, bboxes=bbox, class_labels = labels)             
                input_image = np.array(transformed['image'],np.float32)
                
                if 'bboxes' in transformed and 'class_labels' in transformed:
                
                    bbox = torch.from_numpy(np.array(transformed['bboxes']))
                    labels = torch.from_numpy(np.array(transformed['class_labels']))
                else:
                    bbox = torch.from_numpy(np.array([[0,0,0,0]]))
                    labels = torch.from_numpy(np.array([[0]]))
                
            tensor_image = torch.from_numpy(input_image)
            
            tensor_image = tensor_image.permute(2,0,1)
            tensor_image = tensor_image.float()
        
        
        tensor_bbox = torch.as_tensor(bbox, dtype=torch.float32)
        tensor_labels = torch.as_tensor(labels, dtype =torch.int64)
        
        ########################### result view #################################
        # bbox = tensor_bbox.numpy()
        # no_image = tensor_image.permute(1,2,0).numpy()
        # no_image = np.array(no_image,np.uint8)
        # for box in bbox:
        #     no_image = cv.rectangle(no_image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
        
        # plt.subplot(1,1,1),plt.imshow(no_image)
        # plt.show()
        
        
        output_target = {
            'boxes': tensor_bbox,
            'labels': tensor_labels
            }
        
        return tensor_image,output_target  #        

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


