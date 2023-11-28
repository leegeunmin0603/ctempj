
import numpy as np
import mlflow
from mlflow.models import infer_signature


def make_signature(data_module,model,training_purpose = 'Classification'):
    
    data_module.prepare_data()    
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    
    model = model.eval()
    
    single_data = batch[0]
    
    if(training_purpose == 'Classification'):
        output_data = model(batch[0])  
        pred_signature = infer_signature(single_data.numpy(), output_data.detach().numpy())
    elif(training_purpose == 'ObjectDetection'):
        output_data = model(batch[0],batch[1])  
        pred_signature = []
        num = 0
        
        for comp in output_data[0]:
            num = num+1
            if(num == 1):
                pred_signature = infer_signature(single_data.numpy(), output_data[0][comp].detach().numpy())            
            else:
                pred_signature_temp = infer_signature(single_data.numpy(), output_data[0][comp].detach().numpy())   
                pred_signature.outputs.inputs.append(pred_signature_temp.outputs.inputs)
           
       
    return single_data.numpy(),pred_signature



def make_DatamoduleToNumpy(data_module):
    # 데이터를 NumPy 배열로 변환
    
    data_module.prepare_data()    
    train_loader = data_module.train_dataloader()
    
    data_list = []
    for data, target in train_loader:
        data_list.append(data.numpy())

    # NumPy 배열을 MLflow 데이터로 변환
    mlflow_data = mlflow.data.from_numpy(np.concatenate(data_list))
    
    return mlflow_data


