import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np


# Just an example transformation.
# the p=0.n parameters simply introduce randomness

transform = A.Compose([
    # List the augmentations you want to apply here
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.Resize(640, 640),
    ToTensorV2(),],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])) # THIS IS ESSENTIAL

# define some transformations
def apply_transform(image, bboxes, transform):
    # function uses the transformation as shown above
    augmented = transform(image=image, bboxes=bboxes.tolist())
    augmented_img = augmented['image']
    augmented_bboxes = torch.tensor(augmented['bboxes'])
    return augmented_img, augmented_bboxes

# Path to your images and labels
image_folder = './'
label_folder = './'

# path to the output folder for augmented images and labels
output_folder = 'augmented'
os.makedirs(output_folder, exist_ok=True)

# List all image files in the image folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))
    
    # Load the image and labels
    img = cv2.imread(image_path)
    ## index of 1: because the first column is the label (0 = license plate)
    bboxes = torch.tensor(pd.read_csv(label_path, sep='\s', header=None, engine='python').iloc[:, 1:].to_numpy())
    
    augmented_images = []
    augmented_labels = []

    # create 5 augmentations per image (random outcome has to be specified in "transform")
    for i in range(5):
        # Apply augmentations to image and bounding boxes
        augmented_img, augmented_bboxes = apply_transform(img, bboxes, transform)
        augmented_images.append(augmented_img)
        augmented_labels.append(augmented_bboxes)

        # save and write results
        augmented_image_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_aug{i+1}.jpg')
        augmented_label_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_aug{i+1}.txt')
        cv2.imwrite(augmented_image_path, augmented_img)
        augmented_bboxes_np = augmented_bboxes.numpy()
        np.savetxt(augmented_label_path, augmented_bboxes_np, fmt='%.6f')
