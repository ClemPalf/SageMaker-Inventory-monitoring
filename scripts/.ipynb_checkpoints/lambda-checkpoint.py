import json
import base64
import boto3

ENDPOINT = "pytorch-inference-eia-2022-01-18-12-59-02-036"

def lambda_handler(event, context):
    
    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')
    
    
    response = runtime.invoke_endpoint(EndpointName = ENDPOINT,      # The name of the endpoint we created
                                      ContentType = 'string/url',    # The data format that is expected
                                      Body = event["url"])           # The image url
                                         
    inferences = json.loads(response['Body'].read().decode())

    # We return the data back to the Step Function    
    #event["inferences"] = inferences
    
    return {
        'statusCode': 200,
        'body': inferences
    }