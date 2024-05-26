from paddleocr import PaddleOCR
import cv2
import numpy as np
import math

# image_path = 'path_to_your_image.jpg'
# image = cv2.imread(image_path)


def resize_image(image, desired_size):
    old_size = image.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    resized_image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

    return new_im

def rotated_point(xc, yc, angle, center_x=320, center_y=320):
    xp = (xc - center_x) * math.cos(-angle) + (yc - center_y) * math.sin(-angle) + center_x
    yp =  - (xc - center_x) * math.sin(-angle) + (yc - center_y) * math.cos(-angle) + center_y
    return xp, yp

def rotate_rectangle(xc, yc, w, h, angle_deg):
    angle_rad = np.radians(angle_deg)
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
    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def extract_text_region(image, box):
    xc, yc, w, h, angle_rad = box
    angle_deg = np.degrees(angle_rad)
    
    CALIB_WIDTH = 0.07
    CALIB_HEIGHT = 0.1
    # SECURE_MARGIN = 0.1
    # w += CALIB_WIDTH * w * (SECURE_MARGIN + 1)
    # h += CALIB_HEIGHT * h *(SECURE_MARGIN + 1)
    w += CALIB_WIDTH * w
    h += CALIB_HEIGHT * h
    
    # Step 1: Calculate the rotation matrix for the center and the inverse angle
    M = cv2.getRotationMatrix2D((xc, yc), -angle_deg, 1)
    
    # Step 2: Perform the rotation
    # rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Step 3: Calculate the new bounding box coordinates after rotation
    # Compute the bounding box of the rectangle
    # bbox = cv2.boxPoints(((xc, yc), (w, h), angle_deg))
    bbox = rotate_rectangle(xc, yc, w, h, angle_deg)
    
    width = int(np.linalg.norm(bbox[0] - bbox[1]))
    height = int(np.linalg.norm(bbox[1] - bbox[2]))

    # Điểm đích cho biến đổi góc nhìn
    dst_points = np.array([[0, 0],
                        [width-1, 0],
                        [width-1, height-1],
                        [0, height-1]], dtype='float32')

    # Tính ma trận biến đổi
    M = cv2.getPerspectiveTransform(bbox.astype('float32'), dst_points)

    # Áp dụng biến đổi góc nhìn
    warped_image = cv2.warpPerspective(image, M, (width, height))
    warped_image = cv2.flip(warped_image, 0)
    # Bây giờ warped_image là ROI đã được "duỗi thẳng"
    # cv2.imshow('Cropped Rotated BBox', warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    # print("Bounding box: ", bbox)
    # bbox = np.int0(bbox)
    
    # # Get the bounding rectangle of the rotated rectangle
    # condition = True
    # patience = 0
    # while(condition):
    #     patience += 1
    #     x_p, y_p, w_p, h_p = cv2.boundingRect(bbox)
    #     if x_p < 0 or y_p < 0 or x_p+w_p > image.shape[1] or y_p+h_p > image.shape[0]:
    #         # Handle the case where the bounding box is outside the image boundaries
    #         w -= 0.5
    #         h -= 0.5
    #         bbox = cv2.boxPoints(((xc, yc), (w, h), angle_deg))
    #         bbox = np.int0(bbox)
    #         if patience > 10:
    #             return np.zeros_like(image)
    #     else:
    #         condition = False
        
        
    # # Step 4: Crop the image
    # w = int(w)
    # h = int(h)
    # print("Bounding box: ", x_p, y_p, w_p, h_p)
    # print("w, h: ", w, h)
    # cropped_image = rotated_image[y_p:y_p+h_p, x_p:x_p+w_p]

    return warped_image

def double_size_with_padding(img, padding_color=(0, 0, 0)):

    # Lấy kích thước của hình ảnh đầu vào
    h, w = img.shape[:2]

    # Tính toán kích thước padding
    pad_width = w // 3
    pad_height = h // 3

    # Thêm padding vào hình ảnh
    padded_img = cv2.copyMakeBorder(img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=padding_color)

    return padded_img


def OCR_model(ocr,  image, boxes_raw):
    boxes = [tuple(boxes_raw[j][i] for j in range(5)) for i in range(len(boxes_raw[0]))]
    reg_text = []

    for box in boxes:
        image_cropped = extract_text_region(image, box)
        # draw the text region
        # print("Size: ", image_cropped.shape)
        image_cropped = resize_image(image_cropped, 300)
        image_cropped = double_size_with_padding(image_cropped)
        # image_cropped = cv2.resize(image_cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # image_cropped_show = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Text region', image_cropped_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows(  )
        result = ocr.ocr(image_cropped, cls=False)
        # print("Result: ", result)
        if result and all(r is not None for r in result):
            text = [line[1][0] for line in result for line in line]
            # print("Text: ", text[0])
            reg_text.append(text[0])
        else:
            reg_text.append("-1")
    return reg_text