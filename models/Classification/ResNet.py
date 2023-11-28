import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import mlflow


class ResNetModels(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.001, model_name='resnet18'):
        super(ResNetModels, self).__init__()
        self.learning_rate = learning_rate
        self.selected_modelname = model_name
        
        if(model_name == 'resnet18'):        
            self.model = models.resnet18(pretrained=True)
        elif(model_name == 'resnet50'):        
            self.model = models.resnet50(pretrained=True)
        elif(model_name == 'resnet101'):        
            self.model = models.resnet101(pretrained=True)
        elif(model_name == 'wide_resnet50'):
            self.model = models.wide_resnet50_2(pretrained=True)
        
        # self.model = selectedmodel
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        
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

## 중간 Image 결과물
    def generate_image(self,imgdata, mode):
                
        os.makedirs("./training_image_result/ResNetModels_image_results",exist_ok=True)        
        
        generate_path = f"./training_image_result/ResNetModels_image_results/"+self.selected_modelname + "_" + mode+"_epoch_"+str(self.current_epoch)+"_image.png"
        # save image result
        img = imgdata[0].permute(1,2,0).numpy()   
    
        plt.figure()
        plt.imshow(img)
        plt.ioff()        
        plt.savefig(generate_path)
        plt.close()
        
        return generate_path