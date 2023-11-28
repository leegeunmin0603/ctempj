import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models.detection as models
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import mlflow

import os

# 기반 자료 : https://github.com/Iywie/pl_YOLO/tree/master ( 파일 구조의 변형이 필요함. )
# 기반 자료 : https://github.com/ruhyadi/yolo3d-lightning ( 보류, 더 나은 방법들이 존재하므로 추가 조사가 필요함. )

# yolo8 : 
# 1) https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/
# 2) 

# 작성중

class YoloModelModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.001, model_name='mobilenetV2'):
        super(YoloModelModule, self).__init__()
        self.learning_rate = learning_rate
        self.selected_modelname = model_name
        
        selectedmodel = models.retinanet_resnet50_fpn_v2( num_classes=num_classes)
        # selectedmodel = models.retinanet_resnet50_fpn_v2( num_classes=num_classes,weights_backbone=)
        
        self.model = selectedmodel       
           
        self.loss_fn = torch.nn.SmoothL1Loss()               
        # loss 
    #     L(x) = 0.5 * (|x| - delta) ^ 2   if |x| < delta
    #    |x| - 0.5 * delta         otherwise
        
    
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgdata, target = batch
        output = self(imgdata)
        loss = nn.CrossEntropyLoss()(output, target)
        
        self.log('train_loss', loss, prog_bar=True)
        
        generate_image_path = self.generate_image(imgdata,'train')        
        mlflow.log_artifact(generate_image_path, artifact_path="images")
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgdata, target = batch
        output = self(imgdata)
        loss = nn.CrossEntropyLoss()(output, target)
        self.log('val_loss', loss, prog_bar=True)
        
        generate_image_path = self.generate_image(imgdata,'valid')        
        mlflow.log_artifact(generate_image_path, artifact_path="images")
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_image(self,imgdata, mode):
                
        os.makedirs("./training_image_result/MobileNetModels_image_results",exist_ok=True)        
        
        generate_path = f"./training_image_result/MobileNetModels_image_results/"+self.selected_modelname + "_" + mode+"_epoch_"+str(self.current_epoch)+"_image.png"
        # save image result
        img = imgdata[0].permute(1,2,0).numpy()   
    
        plt.figure()
        plt.imshow(img)
        plt.ioff()        
        plt.savefig(generate_path)
        plt.close()
        
        return generate_path