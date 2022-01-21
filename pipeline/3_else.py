import json
import pathlib
import datetime
import logging
import boto3
import os


with open(evaluation_path_, "w") as f:
    f.write(json.dumps(report_dict))

report_dict = {"Accuracy_of_deployed_model": 0.1}

#output_dir = "/opt/ml/processing/evaluation"
#pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

#evaluation_path = f"{output_dir}/evaluation.json"
uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')[:8]
evaluation_path = "evaluation.json"
with open(evaluation_path, "w") as f:
    f.write(json.dumps(report_dict))
    
    


s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)