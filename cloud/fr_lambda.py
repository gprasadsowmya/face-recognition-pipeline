import json
import boto3
import numpy as np
import time 
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
import torch
import base64
import io
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('sqs boto3 init')
sqs = boto3.client('sqs')
logger.info('get model_path')

model_path = "resnetV1.pt"
logger.info('get video weights path')
model_weights_path = "resnetV1_video_weights.pt"

logger.info("loading weights")
saved_data = torch.load(model_weights_path)
logger.info("loading model")
resnet = torch.jit.load(model_path)

def handler(event, context):
    try:
        logger.info(f"Received {len(event['Records'])} records")
        for record in event['Records']:
            body = json.loads(record['body'])
            content_body = base64.b64decode(body['content'])
            request_id = body['request_id']
            logger.info(f"req id {request_id}")
            filename = body['filename']
            logger.info(f"filename {filename}")


            logger.info('starting transformation pil')
            face_pil = Image.open(io.BytesIO(content_body)).convert("RGB")
            face_numpy = np.array(face_pil, dtype=np.float32)
            face_numpy /= 255.0
            face_numpy = np.transpose(face_numpy, (2, 0, 1))
            face_tensor = torch.tensor(face_numpy, dtype=torch.float32)
            result = ''

            logger.info(f"Processing request {request_id}")
            emb             = resnet(face_tensor.unsqueeze(0)).detach()
            logger.info("getting embedding list..")
            embedding_list  = saved_data[0]
            logger.info("getting name list")
            name_list       = saved_data[1]  
            dist_list       = []
            logger.info("entering for loop")
            for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

            logger.info('dist list')
            idx_min = dist_list.index(min(dist_list))
            logger.info('before result extraction')
            result = name_list[idx_min]
           
            
            logger.info(f"Sending result- {result} for {request_id}")
            sqs.send_message(
                QueueUrl='https://sqs.us-east-1.amazonaws.com/692859916841/1231755904-resp-queue',
                MessageBody=json.dumps({
                    'request_id': request_id,
                    'result': result
                })
            )
        #rough logging   
        return {'statusCode': 200, 'body': json.dumps({'message': 'Pushed to resp queue'})}
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

