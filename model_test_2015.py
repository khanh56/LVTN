import cv2
import torch
import random
from paddleocr import PaddleOCR
import glob
from ryolo_model import image_preprocess, draw_rotated_box
from OCR import OCR_model
import numpy as np
import ryolo_model
from PIL import Image, ImageFont, ImageDraw
import os
import matplotlib.pyplot as plt
# import all the things we need


def draw_text_on_image(image_with_boxes, recognized_texts, boxes_raw):
    draw = ImageDraw.Draw(image_with_boxes)
    boxes = [tuple(boxes_raw[j][i] for j in range(5)) for i in range(len(boxes_raw[0]))]

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()
        print("Custom font not found, using default font.")

    for text, box in zip(recognized_texts, boxes):
        if text != "No text found or OCR failed.":
            x_center, y_center, width, height, angle = box
            x = x_center - width / 2
            y = y_center + height / 2

            draw.text((x, y), text, font=font, fill=(0, 255, 0))  # Red color for the text

    return image_with_boxes


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
model_config = {'s': 0.2,'m': 0.15, 'l': 0.2}   
model_type = 'l'  # s, m, l
conf_threshold = 0.2
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
model_ryolo = ryolo_model.build_model(model_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_image_path = 'assets/ch8_test_images/ts_img_00277.jpg'

word_boxes, image_preprocess, _ = ryolo_model.model_predict(model_ryolo, test_image_path, conf_threshold=conf_threshold)
image_with_boxes, boxes = ryolo_model.draw_rotated_box(np.array(image_preprocess), word_boxes=word_boxes, gt_boxes=None, gt_label=None)
# recognized_texts = OCR_model(ocr, np.array(image_preprocess), boxes)
image_with_boxes_pil = Image.fromarray(image_with_boxes.astype(np.uint8), 'RGB')
# image_with_boxes_pil = draw_text_on_image(image_with_boxes_pil, recognized_texts, boxes)

save_image_path = 'result_image/icdar_2017/' + os.path.basename(test_image_path)

image_with_boxes_pil.save(save_image_path)
image_with_boxes_pil.show()

