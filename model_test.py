import cv2
import torch
import random
from paddleocr import PaddleOCR
import glob
from ryolo_model import image_preprocess, draw_rotated_box
from OCR import OCR_model
import numpy as np
import ryolo_model
from PIL import Image
import os
import matplotlib.pyplot as plt
# import all the things we need

def load_icdar2015_v3(filepath):
    valid_texts = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            text = ','.join(parts[8:])  # Use join() to concatenate parts
            if "###" not in text:  # Use "not in" to check
                valid_texts.append(text.strip())
    # print("Valid texts: ", v text.replacealid_texts)
    return valid_texts 

def load_icdar2013_v3(filepath):
    valid_texts = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            text = parts[4]  # Use join() to concatenate parts
            ## remove the double quotes
            text =('"', '')
            valid_texts.append(text.strip())
    print("Valid texts: ", valid_texts)
    return valid_texts 

# Configurations
model_config = {'s': 0.2,'m': 0.25, 'l': 0.25}   
model_type = 'm'  # s, m, l
conf_threshold = 0.25
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
model_ryolo = ryolo_model.build_model(model_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_image_path = 'assets\icdar2013\Images\img_110.jpg'

word_boxes, image_preprocess, _ = ryolo_model.model_predict(model_ryolo, test_image_path, conf_threshold=conf_threshold)
image_with_boxes, boxes = ryolo_model.draw_rotated_box(np.array(image_preprocess), word_boxes=word_boxes, gt_boxes=None, gt_label=None)

save_image_path = 'result_image/icdar_2013/' + os.path.basename(test_image_path)

plt.imsave(save_image_path, image_with_boxes)
plt.imshow(image_with_boxes)
plt.show()
