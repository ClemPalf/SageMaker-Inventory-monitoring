import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import logging
import sys
import copy
import smdebug.pytorch as smd
#import smdistributed.dataparallel.torch.distributed as dist
#from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True # Otherwise it throws the error "OSError: image file is truncated (150 bytes not processed)"
dist.init_process_group()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, loader, criterion, val_or_test, hook, device):
    '''
    Take a model and a testing data loader and determine the test accuray/loss of the model.
    '''
    model.eval()        # Let's save some compute ressources by not tracking the gradients.
    hook.set_mode(smd.modes.EVAL) # Set the SMDebug hook for validation phase.
    
    running_loss=0      
    running_corrects=0 
    
    for inputs, labels in loader:
        # Pass inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)           
        running_corrects += torch.sum(preds == labels.data)    

    total_loss = running_loss / len(loader)       
    total_acc = running_corrects / len(loader)
    
    if val_or_test == "val":
        logger.info("\nVal set: Average loss: {:.2f}, Accuracy: {:.2f}\n".format(total_loss, total_acc))
    else:   
        logger.info("\nTest set: Average loss: {:.2f}, Accuracy: {:.2f}\n".format(total_loss, total_acc)) 
    
    return total_loss
    
def train(model, train_loader, val_loader, criterion, optimizer, hook, device):
    '''
    Train the model.
    '''
    hook.set_mode(smd.modes.TRAIN)  # Set the SMDebug hook for the training phase.

    epochs = 10       
    
    # To keep track of the best performing model (if we end up overfitting, it won't be a problem).
    best_model_wts = copy.deepcopy(model.state_dict())   
    smallest_val_loss = float("inf")

    for epoch in range(epochs):
    
        model.train()
        running_loss = 0
        correct_pred = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()                    # Reset gradients.
            
            # Pass inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)                   # Forward pass.
            loss = criterion(output, labels)         # Calculate loss.
            loss.backward()                          # Backpropagation.
            optimizer.step()                         # Gradient descent.
            
            # Training Loss
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * inputs.size(0)
            correct_pred += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_pred / len(train_loader.dataset)
        
        logger.info("\nEpoch: {}/{}.. ".format(epoch+1, epochs))
        logger.info("\nTraining set: Average loss: {:.2f}, Accuracy: {:.2f}\n".format(epoch_loss, epoch_acc))                                                                                                                       
        
        # Now, let's calculate the validation loss, if it is an "all-time-low", we will save the model.
        
        current_val_loss = test(model, val_loader, criterion, "val", hook, device)    # Pass the SMDebug hook to the test function
        if current_val_loss < smallest_val_loss:
            smallest_val_loss = current_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Let's return the best model!
    model.load_state_dict(best_model_wts)  
    
    return model 
    
def net():
    '''
    Initialize a pretrained model.
    '''
    # We will train the whole model, but still start from the pre-trained weights
    model = models.vgg11_bn(pretrained=True)

    #for param in model.parameters():
    #    param.requires_grad = False   

    num_features=model.classifier[6].in_features # 1000 
    model.classifier[6] = nn.Sequential(nn.Linear(num_features, 500), # No need for a softmax, it is included in the "nn.CrossEntropyLoss()"
                                        nn.Linear(250, 5))
    return model

def create_data_loaders(data, batch_size):
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                   # To improve training, let's add a 50% chance of horizontal flip.
        transforms.Resize((224, 224)),                            # The VGG model requires a 224*224 input dimension.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Targeted mean for each color channel.
                             std=[0.229, 0.224, 0.225])])  # Targeted std for each color channel.

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])
    
    
    # The dataset.ImageFolder function will automatically assign label to the images according to their subdirectories.
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data, 'train'), transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler)
    
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data, 'validation'), transform=testing_transform)
    val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size) 

    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data, 'test'), transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def main(args):
    '''
    Initialize a model by calling the net function.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net().to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    '''
    Create loss and optimizer.
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    Call the train function to start training the model.
    '''
    
    
    # Prepare batch_size for distributed training
    
    train_loader, val_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    logger.info("Starting model training...")
    model=train(model, train_loader, val_loader, criterion, optimizer, hook, device)  # Pass the SMDebug hook to the train function
    
    '''
    Test the model to see its accuracy
    '''
    #logger.info("Starting model evaluation...")
    test(model, test_loader, criterion, "test", hook, device) # Pass the SMDebug hook to the test function
    
    '''
    Save the trained model
    '''
    logger.info("Saving the model...")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

# I put this function here just as a potential helper for futur cases.
def model_fn(model_dir): # That would be args.model_dir
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model    
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify all the hyperparameters you need to use to train your model.
    '''
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Hyperparameters
    parser.add_argument("--batch-size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)")
    parser.add_argument("--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)")
    
    args=parser.parse_args()
    
    main(args)