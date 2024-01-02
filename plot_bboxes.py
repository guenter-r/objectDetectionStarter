import torch
import os
import cv2, numpy as np

import pandas as pd

def render(img_path, bboxes, name='Unknown', target_path=None):
        
    img = cv2.imread(img_path)
    img_h,img_w, _ = img.shape   
        
    name = os.path.basename(img_path.split('.jpg')[0])
    color = (105, 163, 179) # Georgia Tech GOLD in BGR
    color = (0, 0, 255)
    
    if bboxes.max() > 1:
        for box in bboxes:
            x_min, y_min, x_max, y_max = [int(el) for el in  box]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
    else:
        
        for box in bboxes:
            x, y, w, h = box.tolist()
            x_min = int((x - w / 2) * img_w)
            y_min = int((y - h / 2) * img_h)
            x_max = int((x + w / 2) * img_w)
            y_max = int((y + h / 2) * img_h)
            
            # Create rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color , 3)

    cv2.imwrite(f'{target_path}/{name}_pred_label.jpg', img)
    
    return None