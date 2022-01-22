import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import tarfile
import os
import logging
import sys
import copy
import json
import pathlib
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True # Otherwise it throws the error "OSError: image file is truncated (150 bytes not processed)"

def net():
    '''
    Initialize a pretrained model.
    '''
    
    model = models.resnet34(pretrained=False)

    #for param in model.parameters():
    #    param.requires_grad = False   

    num_features=model.fc.in_features # 1000 
    model.fc = nn.Sequential(nn.Linear(num_features, 500), # No need for a softmax, it is included in the "nn.CrossEntropyLoss()"
                                        nn.Linear(500, 250),
                                        nn.Linear(250, 5))
    return model

    
if __name__=='__main__':
    # printing environment variables
    print(os.environ)
    
    # First, let's load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    with open("model.pth", "rb") as f:
        print("Loading the model")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
    model.eval()
    
    
    
    # Secondly, we will create our datalogger
    batch_size = 16
    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])
    
    test_path = "/opt/ml/processing/test"
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    
    # Now, let's evaluate our model   
    running_corrects=0 
    
    for inputs, labels in test_loader:
        # Pass inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs=model(inputs)
        _, preds = torch.max(outputs, 1)         
        running_corrects += torch.sum(preds == labels.data)    
     
    total_acc = (running_corrects / len(test_loader.dataset)).item()
    print(type(total_acc))
    print(total_acc)
    # Finally, let's save the result in a json file 
    report_dict = {"accuracy": total_acc}
    print(type(report_dict))
    print(report_dict)

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as j:
        j.write(json.dumps(report_dict))
    
    # Let's also import the current_json file 
    json_path = "/opt/ml/processing/accuracy/current_accuracy.json"
    with open(json_path, "r") as f:
        current_json = json.load(f)
    evaluation_path = f"{output_dir}/current_accuracy.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(current_json))
    