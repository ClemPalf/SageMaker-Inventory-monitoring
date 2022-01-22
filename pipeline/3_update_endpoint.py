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
import sagemaker
from sagemaker import get_execution_role
import time

import boto3


if __name__=='__main__':
    print(os.environ)
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    session = sagemaker.Session()

    bucket = session.default_bucket()
    print("Default Bucket: {}".format(bucket))

    region = session.boto_region_name
    print("AWS Region: {}".format(region))
    
    role = get_execution_role()
    print("RoleArn: {}".format(role))

    
    # Print and parse environment variables
    print(os.environ)
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-data', type=str, default=os.environ['MODEL_DATA'])
    args=parser.parse_args()
    
    # Register model
    pytorch_container = sagemaker.image_uris.retrieve(framework="pytorch",
                                                  region=session.boto_region_name,
                                                  version="1.5",
                                                  instance_type="ml.m5.xlarge",
                                                  accelerator_type='ml.eia2.medium')   
    
    # Remember that a model needs to have a unique name
    model_name = "capstone-inventory-monitoring-model-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    primary_container = {
        "Image": pytorch_container,
        "ModelDataUrl": args.model_data
    }
    # Construct the SageMaker model
    model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)
    
    # Create endpointconfig
    # As before, we need to give our endpoint configuration a name which should be unique
    endpoint_config_name = "capstone-endpoint-config-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    # And then we ask SageMaker to construct the endpoint configuration
    endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m5.xlarge",
                                "AcceleratorType": "ml.eia2.medium",
                                "InitialInstanceCount": 1,
                                "ModelName": model_name,
                                "VariantName": "Pytorch-model"
                            }],
                            DataCaptureConfig = { 
                                "CaptureOptions": [
                                    {
                                        "CaptureMode": "Input"
                                    },
                                    {
                                        "CaptureMode": "Output"
                                    }],
                            "DestinationS3Uri": "s3://sagemaker-us-east-1-646714458109/capstone-inventory-project/data_capture",
                            "EnableCapture": True,
                            "InitialSamplingPercentage": 100})
    
    # Finally, let's update our endpoint
    session.sagemaker_client.update_endpoint(EndpointName="pytorch-inference-2022-01-22-14-40-51-906", EndpointConfigName=endpoint_config_name)
    
    # Aslo, let's update our accuracy file
    s3_client = boto3.client('s3')
    response = s3_client.upload_file("/opt/ml/processing/accuracy/evaluation.json", "sagemaker-us-east-1-646714458109", "capstone-inventory-project/current_accuracy.json")

    
    
    
    