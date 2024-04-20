

from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict as edict
import os
import glob
import random
import string
from math import sin, cos, radians, sqrt
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

"""# **Config**"""

__C = edict()
cfg = __C

__C.GENERATE_FAKE_IMAGE = edict()
__C.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR =  r"D:\Khanh\testFlask\images\hinh_anh.jpg"
__C.GENERATE_FAKE_IMAGE.FONT_DIR = r"Fonts"
__C.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN = 20
__C.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX = 100
__C.GENERATE_FAKE_IMAGE.TEXT_COLOR = None       # format = (R,G,B); None = random
__C.GENERATE_FAKE_IMAGE.IMAGE_SIZE = (640, 640) # format = (w, h)
__C.GENERATE_FAKE_IMAGE.WORD_COUNT = 20
__C.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN = 2
__C.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX = 5
__C.GENERATE_FAKE_IMAGE.ANGLE_MIN = -90         # degree
__C.GENERATE_FAKE_IMAGE.ANGLE_MAX =  90

"""# **utils**"""

def rect_polygon(xc, yc, w, h, angle, center_x, center_y):
    xc_rotated, yc_rotated = rotated_point(xc, yc, angle, center_x, center_y)
    angle_rad = angle * math.pi/180
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    x1 = int(xc_rotated - w/2 * cos_a - h/2 * sin_a)
    y1 = int(yc_rotated - w/2 * sin_a + h/2 * cos_a)
    x2 = int(xc_rotated + w/2 * cos_a - h/2 * sin_a)
    y2 = int(yc_rotated + w/2 * sin_a + h/2 * cos_a)
    x3 = int(xc_rotated + w/2 * cos_a + h/2 * sin_a)
    y3 = int(yc_rotated + w/2 * sin_a - h/2 * cos_a)
    x4 = int(xc_rotated - w/2 * cos_a + h/2 * sin_a)
    y4 = int(yc_rotated - w/2 * sin_a - h/2 * cos_a)
    a = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)])
    return a

def check_IoU(xc1, yc1, w1, h1, angle1, bbox, angle2, center_x, center_y, IoU_threshold=0): #rect_b: angle = 0
    xc2, yc2, w2, h2 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]
    polygon2 = rect_polygon(xc2, yc2, w2, h2, angle2, center_x, center_y)
    for i in range(len(xc1)):
        polygon1 = rect_polygon(xc1[i], yc1[i], w1[i], h1[i], angle1[i], center_x, center_y)
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.area + polygon2.area - intersect
        iou = intersect / union
        if iou > IoU_threshold:
            return False
    return True

def draw_rotated_box(image, xc, yc, w, h, angle_rad, cls, color=(255, 0, 0), thickness=1):
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    x1 = (xc - w/2 * cos_a - h/2 * sin_a)
    y1 = (yc - w/2 * sin_a + h/2 * cos_a)
    x2 = (xc + w/2 * cos_a - h/2 * sin_a)
    y2 = (yc + w/2 * sin_a + h/2 * cos_a)
    x3 = (xc + w/2 * cos_a + h/2 * sin_a)
    y3 = (yc + w/2 * sin_a - h/2 * cos_a)
    x4 = (xc - w/2 * cos_a + h/2 * sin_a)
    y4 = (yc - w/2 * sin_a - h/2 * cos_a)
    try:
        for x1, y1, x2, y2, x3, y3, x4, y4, cls in zip(x1, y1, x2, y2, x3, y3, x4, y4, cls):
            if cls != 62:
                color = (255, 0, 0)
            elif cls == 62:
                color = (0, 255, 0)
            image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
            image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
            image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
            image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
    except:
        if cls != 62:
            color = (255, 0, 0)
        elif cls == 62:
            color = (0, 255, 0)
        image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
        image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
        image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
        image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
    return image


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def rotated_point(xc, yc, angle, center_x, center_y):
    angle_rad = math.radians(angle)
    xp = (xc - center_x) * math.cos(angle_rad) + (yc - center_y) * math.sin(angle_rad) + center_x
    yp =  - (xc - center_x) * math.sin(angle_rad) + (yc - center_y) * math.cos(angle_rad) + center_y
    return xp, yp

def get_pos(font, text, imgsz = __C.GENERATE_FAKE_IMAGE.IMAGE_SIZE):
    width, height = imgsz
    magic_number_1 = (sqrt(2) - 1)/(2*sqrt(2))
    magic_number_2 = 1 - magic_number_1
    text_width, text_height = font.getsize(text)
    pos_x_min = int(width * magic_number_1) + 1
    pos_x_max = int(width * magic_number_2 - text_width)
    pos_y_min = int(height * magic_number_1) + 1
    pos_y_max = int(height * magic_number_2 - text_height)
    print(pos_x_min, pos_x_max, pos_y_min, pos_y_max)
    pos = (random.randint(pos_x_min, pos_x_max), random.randint(pos_y_min, pos_y_max))
    return pos

"""# **Generate Image Function**"""

def generate_fake_image(image_background = __C.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR,
                        font_dir = __C.GENERATE_FAKE_IMAGE.FONT_DIR,
                        font_size_min = __C.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN,
                        font_size_max = __C.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX,
                        word_count = __C.GENERATE_FAKE_IMAGE.WORD_COUNT,
                        text_color = __C.GENERATE_FAKE_IMAGE.TEXT_COLOR,
                        imgsz  = __C.GENERATE_FAKE_IMAGE.IMAGE_SIZE,
                        word_length_min = __C.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN,
                        word_length_max = __C.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX,
                        angle_min = __C.GENERATE_FAKE_IMAGE.ANGLE_MIN,
                        angle_max = __C.GENERATE_FAKE_IMAGE.ANGLE_MAX):
    cls = string.ascii_letters + string.digits
    font_size = random.randint(font_size_min, font_size_max)
    # image_background_paths = sorted(glob.glob(os.path.join(__C.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR + "/*/*")))
    # image_background_path = np.random.choice(image_background_paths)
    font_paths = sorted(glob.glob(os.path.join(__C.GENERATE_FAKE_IMAGE.FONT_DIR + "/*")))
    width, height = imgsz
    center_x, center_y = width / 2, height / 2
    if text_color is None:
        text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    word_length = [random.randint(word_length_min, word_length_max) for i in range(word_count)]
    text = [generate_random_string(word_length[k]) for k in range(word_count)]
    rotated_text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rotated_text_image)

    angle = []
    mu = (angle_min + angle_max) / 2
    sigma = (angle_max - angle_min) / 6
    for i in range(word_count):
        condition = True
        while condition:
            num = random.gauss(mu, sigma)
            if num >= angle_min and num <= angle_max:
                angle.append(num)
                condition = False

    xc = [[] for i in range(word_count)]
    yc = [[] for i in range(word_count)]
    w = [[] for i in range(word_count)]
    h =  [[] for i in range(word_count)]

    xc_word, yc_word, w_word, h_word = [], [], [], []



    result = np.ones((width, height, 3), dtype=np.uint8)
    for j in range(word_count):
        font_path = np.random.choice(font_paths)
        rotated_text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        condition = True
        font = ImageFont.truetype(font_path, font_size)
        while condition:
            try:
                font = ImageFont.truetype(font_path, font_size)
                pos = get_pos(font, text[j])  ## thiếu size ảnh, mặc định là 640 640
                condition = False
            except:
                font_size -= 10
        condition_IoU = True
        patience = 0
        while condition_IoU:
            bbox = draw.textbbox((pos[0], pos[1]), text[j], font=font)
            if check_IoU(xc_word, yc_word, w_word, h_word, angle, bbox, angle[j], center_x, center_y):
                xc_word.append(int((bbox[0] + bbox[2]) / 2))
                yc_word.append(int((bbox[1] + bbox[3]) / 2))
                w_word.append(int(bbox[2] - bbox[0]))
                h_word.append(int(bbox[3] - bbox[1]))
                patience = 0
                condition_IoU = False
            else:
                font_size = font_size_min if font_size - 5 < font_size_min else font_size - 5
                font = ImageFont.truetype(font_path, font_size)
                pos = get_pos(font, text[j])
                patience += 1
                if patience == 5:
                    break
        if patience == 5:
            word_count = j
            break

        draw = ImageDraw.Draw(rotated_text_image)
        draw.text(pos, text[j], font=font, fill='black')
        x, y = pos
        for c in text[j]:
            bbox = draw.textbbox((x, y), c, font=font)
            xc[j].append(int((bbox[0] + bbox[2]) / 2))
            yc[j].append(int((bbox[1] + bbox[3]) / 2))
            w[j].append(int(bbox[2] - bbox[0]))
            h[j].append(int(bbox[3] - bbox[1]))
            x += draw.textlength(c, font=font)

        rotated_text_image = rotated_text_image.rotate(angle[j], expand=False, center=(center_x, center_y))

        image = Image.new('RGB', rotated_text_image.size, (255, 255, 255))
        image.paste(rotated_text_image, mask=rotated_text_image.split()[3])
        image = np.array(image)

        result = result * (image // 255)

    angle_rad = [(angle[j] * np.pi/180)  for j in range(word_count)]
    annot = []
    for i in range(word_count):
        for j in range(word_length[i]):
            xc[i][j], yc[i][j] = rotated_point(xc[i][j], yc[i][j], angle[i], center_x, center_y)
            annot.append([int(xc[i][j]), int(yc[i][j]), w[i][j], h[i][j], angle_rad[i], cls.index(text[i][j])])
        xc_word[i], yc_word[i] = rotated_point(xc_word[i], yc_word[i], angle[i], center_x, center_y)
        annot.append([int(xc_word[i]), int(yc_word[i]), w_word[i], h_word[i], angle_rad[i], 62])
    annot = np.array(annot)

    # image_background = Image.open(image_background_path)
    if image_background.mode != 'RGB':
        image_background = image_background.convert('RGB')
    
    image_background = image_background.resize((image.shape[0], image.shape[1]))
    image_background = np.array(image_background)
    if image_background.ndim == 2:
        image_background = np.concatenate([image_background[..., None], image_background[..., None],
                                           image_background[..., None]], axis=-1)

    a = np.all(result, axis=2)
    for i in range(width):
        for j in range(height):
            if not(a[i][j]):
                image_background[i][j] = np.array(text_color, dtype='uint8')

    return image_background, annot

"""# **Example**

## **Generate Image**
"""

# cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR =  r"D:\Khanh\testFlask\images\hinh_anh.jpg"
# cfg.GENERATE_FAKE_IMAGE.FONT_DIR = r"D:\Khanh\testFlask\Fonts"
# cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN = 10
# cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX = 200
# cfg.GENERATE_FAKE_IMAGE.TEXT_COLOR = None           # format = (R,G,B); None = random
# cfg.GENERATE_FAKE_IMAGE.IMAGE_SIZE = (640, 640)     # format = (w, h)
# cfg.GENERATE_FAKE_IMAGE.WORD_COUNT = 3
# cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN = 1
# cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX = 5
# cfg.GENERATE_FAKE_IMAGE.ANGLE_MIN = -30             # degree
# cfg.GENERATE_FAKE_IMAGE.ANGLE_MAX = 30

# image, annot  = generate_fake_image(cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR,
#                                     cfg.GENERATE_FAKE_IMAGE.FONT_DIR,
#                                     cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN,
#                                     cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX,
#                                     cfg.GENERATE_FAKE_IMAGE.WORD_COUNT,
#                                     cfg.GENERATE_FAKE_IMAGE.TEXT_COLOR,
#                                     cfg.GENERATE_FAKE_IMAGE.IMAGE_SIZE,
#                                     cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN,
#                                     cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX,
#                                     cfg.GENERATE_FAKE_IMAGE.ANGLE_MIN,
#                                     cfg.GENERATE_FAKE_IMAGE.ANGLE_MAX)
# print(annot)
# plt.imshow(image)

# """## **Visualize label**"""

# image_draw = draw_rotated_box(image, annot[:, 0], annot[:, 1], annot[:, 2], annot[:, 3], annot[:, 4], annot[:, 5])
# plt.imshow(image_draw)
# plt.show()
# # print(image_draw)