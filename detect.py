import numpy as np
import pandas as pd
import albumentations as A
from ultralytics import YOLO

from plot_bboxes import render

training = False

if training:
    model = YOLO('yolov8n.pt')
    # Training
    results = model.train(
        data='plates.yaml',
        imgsz=640,
        epochs=10,
        # batch=8,
        # name='yolov8n_5_adam',
        # optimizer='AdamW',
        # lr0 = 1e-2,
        # device = [0,1,2,3]
        )
else:
    model = YOLO('runs/detect/train2/weights/best.pt')
    
results = model.predict('cars165.png')

names = model.names
class_id = results[0].boxes.cls.item()
names[class_id]

boxes, masks, probs = None, None, None
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmenation masks outputs
    probs = result.probs  # Class probabilities for classification outputs

boxes = boxes.xyxy

render('cars165.png', boxes, './')
