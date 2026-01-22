import threading
import traceback
import json
import base64
import boto3
import sys
import os
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
import io
import requests
import numpy as np
import awsiot.greengrasscoreipc.clientv2 as clientV2
import logging
from awsiot.greengrasscoreipc.model import QOS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'facenet_pytorch'))
from facenet_pytorch import MTCNN

def on_stream_event(event):
    print('inside on stream event')

    body = json.loads(event.message.payload)
    body_content = base64.b64decode(body['encoded'])
    body_request_id = body['request_id']
    body_filename = body['filename']
    print(f'message filename {body_filename}')
    print(f'request id {body_request_id}')

    print('converting to np array')
    image = Image.open(io.BytesIO(body_content)).convert("RGB")
    image = np.array(image)
    image = Image.fromarray(image)

    print('run detector')
    face, prob = detector(image, return_prob=True, save_path=None)

    if face != None:
                    face_img = face - face.min()  
                    face_img = face_img / face_img.max()  
                    face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
                    face_pil = Image.fromarray(face_img, mode="RGB")

                    buffer = io.BytesIO()
                    face_pil.save(buffer, format="JPEG")  
                    img_bytes = buffer.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    print('create message body with detected face')
                    message_body = {
                    "request_id": body_request_id,
                    "filename": body_filename,
                    "content": img_base64,
                                    }
                    
                    print('sending detected image to sqs queue')
                    sqs_client.send_message(
                    QueueUrl='1231755904-req-queue',
                    MessageBody=json.dumps(message_body)
                )
    else:
                    message_body = {
                    "request_id": body_request_id,
                    "result": "No-Face"}            

                    print('sending to sqs queue')
                    sqs_client.send_message(
                                    QueueUrl='1231755904-resp-queue',
                                    MessageBody=json.dumps(message_body)
                                )

def on_stream_error():
    print('inside on stream error')

print('started fd_component.py')
topic = "clients/1231755904-IoTThing"
qos = QOS.AT_LEAST_ONCE

print('initializing MTCNN detector')
detector = MTCNN(image_size=240, margin=0, min_face_size=20)

print('calling IPC client')
ipc_client = clientV2.GreengrassCoreIPCClientV2()
resp, operation = ipc_client.subscribe_to_iot_core(
    topic_name=topic,
    qos=qos, 
    on_stream_event=on_stream_event,
    on_stream_error=on_stream_error,
)

print('init boto3 for sqs')
sqs_client = boto3.client('sqs')

event = threading.Event()
event.wait()

operation.close()
ipc_client.close()
