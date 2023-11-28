import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models.detection as models

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import pytorch_lightning as pl

import matplotlib.pyplot as plt

import mlflow

import os


class TorchDetectionModelModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.001, model_name='retinanet_resnet50_fpn'):
        super(TorchDetectionModelModule, self).__init__()
        self.learning_rate = learning_rate
        self.selected_modelname = model_name
        
        if(model_name == 'retinanet_resnet50_fpn'):
            selectedmodel = models.retinanet_resnet50_fpn_v2( num_classes=num_classes,pretrained_backbone=True)
        else:
            selectedmodel = models.fasterrcnn_resnet50_fpn_v2(num_classes=num_classes,pretrained_backbone=True)
            
            in_features = selectedmodel.roi_heads.box_predictor.cls_score.in_features
            selectedmodel.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.model = selectedmodel             
        
    
    def forward(self, image):
        
        result = self.model(image)
        return result
        
    

    def training_step(self, batch, batch_idx):
        imgdata, targets = batch
        # 
        
        if(self.selected_modelname == 'retinanet_resnet50_fpn'):
            output = self(imgdata,targets)
            loss = output['classification'] + output['bbox_regression']
        else:
            
            targets = [{k: v for k, v in t.items()} for t in targets]
            loss_dict = self.model(imgdata, targets)
            loss = sum(loss for loss in loss_dict.values())
        
        self.log('train_loss', loss, prog_bar=True)
        
        generate_image_path = self.generate_image(imgdata,'train')        
        mlflow.log_artifact(generate_image_path, artifact_path="images")
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgdata, targets = batch
        
        self.model = self.model.train()
        
        
        
        if(self.selected_modelname == 'retinanet_resnet50_fpn'):
            output = self(imgdata,targets)
            loss = output['classification'] + output['bbox_regression']
        else:            
            targets = [{k: v for k, v in t.items()} for t in targets]
            loss_dict = self.model(imgdata, targets)
            loss = sum(loss for loss in loss_dict.values())
            
        
        self.log('val_loss', loss, prog_bar=True)
        
        generate_image_path = self.generate_image(imgdata,'valid')        
        mlflow.log_artifact(generate_image_path, artifact_path="images")
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_image(self,imgdata, mode):
                
        os.makedirs("./training_image_result/RetinaNet_image_results",exist_ok=True)        
        
        generate_path = f"./training_image_result/RetinaNet_image_results/"+self.selected_modelname + "_" + mode+"_epoch_"+str(self.current_epoch)+"_image.png"
        # save image result
        img = imgdata[0].permute(1,2,0).numpy()   
    
        plt.figure()
        plt.imshow(img)
        plt.ioff()        
        plt.savefig(generate_path)
        plt.close()
        
        return generate_path