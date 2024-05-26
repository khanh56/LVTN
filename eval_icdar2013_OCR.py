# import all the thing we need
import cv2
import torch
import time
import random
from paddleocr import PaddleOCR
import glob
from ryolo_model import image_preprocess, draw_rotated_box
from OCR import OCR_model
import numpy as np
import ryolo_model
from PIL import Image
import os
import logging
logger = logging.getLogger()

def load_icdar2015_v3(filepath):
    valid_texts = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            text = ','.join(parts[8:])  # Sử dụng join() để nối các phần
            if "###" not in text:  # Sử dụng "not in" để kiểm tra
                valid_texts.append(text.strip())
    print("Valid texts: ", valid_texts)
    return valid_texts 

def load_icdar2013_v3(filepath):
    valid_texts = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            text = parts[4]  # Sử dụng join() để nối các phần
            ## remove the double quotes
            text = text.replace('"', '')
            valid_texts.append(text.strip())
    print("Valid texts: ", valid_texts)
    return valid_texts 
# file_path = os.path.join(test_gt_dir, 'gt_img_1.txt')
# valid_texts = load_icdar2013_v3(file_path)
# print("Valid texts: ", valid_texts)    

logger.setLevel(logging.INFO)

fh = logging.FileHandler('test_2013_2.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


# Configurations
test_gt_dir = "assets/icdar2013/GT"
image_dir_for_testing = 'assets/icdar2013/Images'
test_image_paths = sorted(glob.glob(image_dir_for_testing + '/*'))
model_config = {'s': 0.2,'m': 0.25, 'l': 0.25}   
for model_type, conf_threshold in model_config.items():
    # model_type = 's'
    # conf_threshold = 0.25
 
    logger.info("Start evaluating ICDAR 2013 dataset")
    logger.info(f"Model type: {model_type}, confidence threshold: {conf_threshold}")   
    logger.info(f"Use angle classification: False")

    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    count_gt = 0
    count_pred_true = 0

        
    model_ryolo = ryolo_model.build_model(model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count = 0
    for image_path in test_image_paths:
        count_pred_true += 1
        print(image_path)
        count += 1
        word_boxes, processed_image, _ = ryolo_model.model_predict(model_ryolo, image_path, conf_threshold=conf_threshold)
        image_with_boxes, boxes = ryolo_model.draw_rotated_box(np.array(processed_image), word_boxes=word_boxes, gt_boxes=None, gt_label=None)
        image_with_boxes_pil = Image.fromarray(image_with_boxes.astype(np.uint8), 'RGB')
        recognized_texts = OCR_model(ocr, np.array(processed_image), boxes)
        print(f"Recognized texts: {recognized_texts}")
        filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
        test_gt_path = os.path.join(test_gt_dir, 'gt_' + filename_without_extension + '.txt')
        gts = load_icdar2013_v3(test_gt_path)
        count_gt += len(gts)
        for gt in gts:
            if gt in recognized_texts:
                count_pred_true += 1
        if count_gt == 0:
            print("accuracy: 0")        
        else:
            print("accuracy: ", count_pred_true/count_gt)
        print("count: ", count)
        logger.info(f"Runtime {count} Image path: {image_path}, GT: {gts}, Predicted: {recognized_texts}") 
        logger.info(f"True predictions: {count_pred_true}, Total GT: {count_gt}")

    logger.info(f"Final accuracy: {count_pred_true/count_gt}")
    print("Final accuracy: ", count_pred_true/count_gt)