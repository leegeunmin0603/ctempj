import numpy as np
from models.ObjectDetection import TorchDetectionModel
import torchvision
import cv2 as cv
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

classlabellist = ['others','box']


num_classes = len(classlabellist)

ckpt_path = './checkpoints/ObjectDetection_fasterrcnn_resnet50_fpn_v2-epoch=06-val_loss=0.0389.ckpt'


model = TorchDetectionModel.TorchDetectionModelModule.load_from_checkpoint(ckpt_path, num_classes=num_classes, model_name = "fasterrcnn_resnet50_fpn_v2")
# TorchDetectionModel.TorchDetectionModelModule(num_classes=num_classes, model_name="fasterrcnn_resnet50_fpn_v2")

# selectedmodel = models.fasterrcnn_resnet50_fpn_v2(num_classes=num_classes,pretrained_backbone=True)

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained = True)

model = model.eval()


image = cv.imread('./data/Detection/20231120-151423_color.jpg')
# image = cv.imread('./data/Custom/person/image1.jpg')
image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
np_sample_image = np.array(image)

# img_size = [128,128]
transform = transforms.Compose([
        # transforms.Resize((img_size[0], img_size[1])),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

tensor_image = torch.from_numpy(np_sample_image)
tensor_image = tensor_image.permute(2,0,1)
tensor_image = tensor_image.float()
tensor_image = tensor_image/255.


transformed_img = transform(tensor_image)

transformed_img = transformed_img.unsqueeze(0)

full_mask = np.copy(np_sample_image)

with torch.no_grad():  # 그라디언트 계산 비활성화
    result = model(transformed_img)

    confidence_threshold = 0.4
    nms_threshold = 0.4
    
    boxes = result[0]['boxes'].numpy().tolist()
    confidences = result[0]['scores'].numpy().tolist()
    labels = result[0]['labels'].numpy().tolist()
    
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    for index in indices:
        
        nms_box = boxes[index]
        full_mask = cv.rectangle(full_mask,(int(nms_box[0]),int(nms_box[1])),(int(nms_box[2]),int(nms_box[3])),(255,0,0),1)
        print(nms_box)
        
        
   
plt.subplot(1,2,1),plt.imshow(np_sample_image)
plt.subplot(1,2,2),plt.imshow(full_mask)
    
plt.show()


print('')
