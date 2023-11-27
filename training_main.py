import os
import pytorch_lightning as pl

# mlflow log
import mlflow
import mlflow.pytorch

# utils
from utils.mlflowUtil import make_signature

##### model structure load #####
# Classification
from models.Classification import ResNet
from models.Classification import MobileNet
# Object Detection
from models.ObjectDetection import TorchDetectionModel


# datasets
from Datasets import MnistDataset
from Datasets import Customdataset

classlabellist = ['others','box']


def train_main(training_purpose, model_name):
    
    
    # 데이터 모듈 초기화    
    # Dataset log
    example_data = "MNist"    
    dataset_path = "./Datasets/training_dataset_log.txt"
    with open(dataset_path, "w") as f:
        f.writelines(example_data)    
    
    ## Dataset Setting

    batch_size = 1
    # data_module = MnistDataset.MNISTDataModule(batch_size=batch_size)  ## Sample for MNIST
    
    # data_module = Customdataset.CustomDataModule(
    #     data_dir = './data/Custom', 
    #     dataset_mode=training_purpose, # dataset 의 모드 선택 : classification , object detection
    #     expansion_data_num = 3000, # data 개수 임의 확장
    #     transform_mode = 'aug', # augmentation 사용여부 aug : 다양한 모드 /  basic : normalization만
    #     class_labellist= classlabellist, 
    #     batch_size=batch_size, 
    #     dataset_format='jpg' # numpy 도 읽는 것이 가능        
    #     )
    
    data_module = Customdataset.CustomDataModule(
        data_dir = './data/Detection', 
        dataset_mode=training_purpose, # dataset 의 모드 선택 : classification , object detection
        expansion_data_num = 3000, # data 개수 임의 확장
        transform_mode = 'aug', # augmentation 사용여부 aug : 다양한 모드 /  basic : normalization만
        class_labellist= classlabellist, 
        batch_size=batch_size, 
        coco_json_filename='instances_default.json',
        dataset_format='jpg' # numpy 도 읽는 것이 가능                
        )  
     
    data_module.setup(validation_split=0.2)
    
    # 학습 모델 초기화
    num_classes = len(classlabellist)
    learning_rate = 0.001
    
    if ( training_purpose == "Classification"):
        if(model_name == "mobilenetV2"):        
            model = MobileNet.MobileNetModels(num_classes=num_classes, model_name = model_name, learning_rate=learning_rate)
        else:
            model = ResNet.ResNetModels(num_classes = num_classes, model_name = model_name, learning_rate=learning_rate)
    elif ( training_purpose == "ObjectDetection"):
        if(model_name == "retinanet_resnet50_fpn"):
            model = TorchDetectionModel.TorchDetectionModelModule(num_classes=num_classes, learning_rate=learning_rate, model_name = "retinanet_resnet50_fpn")
        else:
            model = TorchDetectionModel.TorchDetectionModelModule(num_classes=num_classes, learning_rate=learning_rate, model_name = "mask")
    
    # mlflow 새로운 저장소 경로(중앙 집중형) 설정
    new_local_path = "file:D:/mlruns/"
    mlflow.set_tracking_uri(new_local_path)
    mlflow.set_experiment(training_purpose +"_" + model_name+"result")

    
    # Trainer Set - 학습 루프 및 기본 log 저장 방법
    ## Basic
    checkpoint_callback_set = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', 
        save_top_k=3, 
        dirpath="./checkpoints",    
        filename=training_purpose + "_" + model_name + "-{epoch:02d}-{val_loss:.4f}",
        mode='min'
        )
    
    ## resume training using ckpt  
    trainer = pl.Trainer(max_epochs=7, callbacks = [checkpoint_callback_set])    
    mlflow.pytorch.autolog(registered_model_name = model_name)    

    
    
    with mlflow.start_run(run_name=training_purpose + "_" + model_name + '_trainingLog') as run:
        
        ## dataset log
        mlflow.log_params({"user_name": "사용자 이름"})
        
        
        
        # DataLoader를 사용하여 데이터 한 개 추출하고 signature 정보 획득
        single_data, pred_signature = make_signature(data_module,model,training_purpose)
            
        ## Resume Training
        resume_training = False
        select_checkpoint = './checkpoints/'+ training_purpose + '_' + model_name + '-epoch=04-val_loss=0.1339.ckpt'
        
        # training image result saved folder
        os.makedirs("./training_image_result",exist_ok=True) 
        
        # Do Training         
        if(resume_training):
            trainer.fit(model, data_module, ckpt_path=select_checkpoint)   
        else:
            trainer.fit(model, data_module)   
        
        
        mlflow.pytorch.log_model(model,"model",input_example=single_data)
        
        print('training done')



    
    mlflow.end_run()
    print('all done')
    
if __name__ == '__main__':
    # training_purpose = "Classification"
    # model_name = "resnet18"
    
    training_purpose = "ObjectDetection"
    model_name = "fasterrcnn_resnet50_fpn_v2"
    train_main(training_purpose, model_name)
