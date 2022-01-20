# Remember that a model needs to have a unique name
xgb_model_name = "boston-update-xgboost-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference and the model artifacts come from the previous call to fit.
xgb_primary_container = {
    "Image": xgb_container,
    "ModelDataUrl": "s3://sagemaker-us-east-1-646714458109/capstone-inventory-project/main_training/pytorch-training-2022-01-17-10-08-10-837/output/model.tar.gz"
}

# And lastly we construct the SageMaker model
xgb_model_info = session.sagemaker_client.create_model(
                                ModelName = xgb_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = xgb_primary_container)


# We will also create an endpoint configuration that we will use later
# We need to give our endpoint configuration a name which should be unique
endpoint_config = "inventory-monitoring-endpoint-config"
# And then we ask SageMaker to construct the endpoint configuration
endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config,
                            ProductionVariants = [{
                                instance_type='ml.m5.large',
                                accelerator_type='ml.eia2.medium', # Low cost GPU
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": xgb_model_name,
                                "VariantName": "XGB-Model"
                            }])