# Project Overview: Inventory Monitoring at Distribution Centers
Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. In this project, we investigated the potential of CNN model to count the number of objects in a bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.  
  
This repository contains the notebooks and python scripts that were used to 1) download and inspect the dataset, 2) investigate different model architectures and hyperparameters, 3) deploy the model cost-effectively, and 4) automate the whole process via a sagemaker pipeline.  
  
The project was undertaken esclusively within the aws environment and more specificaly, sagemaker studio. 

## 1) Data preparation
To complete this project we will be using the <a href="https://registry.opendata.aws/amazon-bin-imagery/" target="_blank">Amazon Bin Image Dataset</a>. The dataset contains 500,000 images of bins containing one or more objects. For each image there is a metadata file containing information about the image like the number of objects, it's dimension and the type of object.

The notebook "Data_preparation" contains the code that download a subset of this dataset (~10k images) and split them into train, validation and test sets. 
A few images were displayed to estimate their average quality.  
Ultimatly, the images are uploaded to a specific s3 bucket.

## 2) Model choice & hpo
In the second notebook "Model_choice & hpo", the sagemaker hyperparameter tuner function was used to investigate the potential of different model architectures and learning hyperparameters (learning_rate and batch_size).  

## 3) Model training
The best performing configuration found by the hyperparameter tuner (a vgg net with a batch-size of 16 and a learning rate of 0.002) was used to train the entire model for a higher number of epochs (the model weights were pretrained but no layer were frozen). 

As the model did not train properly, a second model architecture was investigated. 

## 4) 2nd model training
This time, a resnet34 was train from scratch (no transfer learning at all). The model manage to train correctly, although the accuracy obtained was poor.

## 5) Deployement 
The "Model deployement" notebook contains the code used to deploy the model and set up autoscalling.  
Elastic inference was attached to the endpoint to reduce prediction time.  
A data capture configuration was also setup to monitor the endpoint usage.  
The "scripts/inference.py" file was attached to the endpoint to permit 2 types of input: the image as bytes, and the S3 link to which the image is stored.

A lambda function was created as an intermediate to receive a link, invoke the endpoint, and return the prediction. Its content was copy pasted inside the scripts/lambda.py file. 

## 6) Sagemaker pipeline
Finally, a sagemaker pipeline was created to automate the process of retraining a model (given the data in a specific S3 folder), assess if this model perform better than the one currently deploy (given a test dataset), and update the model endpoint if it's the case.

The pipeline was built within the "SageMaker pipeline" notebook. All the code associated to the different steps are inside the pipeline folder.  


For a better understanding of the project and the choices made, have a look to the report.pdf file!
