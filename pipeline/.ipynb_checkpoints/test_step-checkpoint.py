import json
import pathlib
import datetime
import logging
import boto3
import os

report_dict = {"Did_it_work": "Yessss!"}
with open("youpi.json", "w") as f:
    f.write(json.dumps(report_dict))


s3_client = boto3.client('s3')
response = s3_client.upload_file("youpi.json", "sagemaker-us-east-1-646714458109", "capstone-inventory-project/youpi.json")