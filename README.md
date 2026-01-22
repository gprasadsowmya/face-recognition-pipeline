# Edge-to-Cloud Face Recognition Pipeline (AWS IoT Greengrass + SQS + Lambda)

A distributed edge/cloud ML pipeline that performs face detection at the edge (AWS IoT Greengrass) and face recognition in the cloud (AWS Lambda), connected through decoupled SQS request/response queues.

## Architecture
![Architecture Diagram]
- IoT device publishes video frames (base64) to an MQTT topic in AWS IoT Core.
- Greengrass Core runs a **Face Detection** component:
  - Subscribes to the MQTT topic
  - Runs MTCNN to detect a face
  - Sends detected face crops to an SQS **request** queue (or sends a “No-Face” result to the response queue)
- AWS Lambda **Face Recognition** function:
  - Triggered by SQS request messages
  - Decodes the face crop
  - Computes embeddings with a TorchScript model
  - Matches against a stored embedding database (nearest distance)
  - Publishes `{ request_id, result }` to the SQS **response** queue
- Client consumes results from the response queue.

## Highlights
- Edge inference to reduce bandwidth and avoid sending full frames to the cloud.
- Asynchronous, scalable request/response pattern using SQS.
- Production-style separation of responsibilities: edge detection → cloud recognition → async result delivery.

## Results (end-to-end validation)
Validated the full pipeline using a 100-frame workload (MQTT → Greengrass face detection → SQS → Lambda recognition → SQS response). 

- Requests processed: 100
- Requests success rate: 100%
- Total test duration: 76.23s
- Average end-to-end latency: 0.7623s

## Repo structure
```text
.
├── edge/
│   ├── fd_component.py
├── cloud/
│   ├── fr_lambda.py               
├── docs/
│   ├── architecture.png
└── README.md
```




