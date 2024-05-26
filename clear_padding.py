import cv2
import numpy as np
import os
import glob

# path_to_image = 'result_image\icdar_2013\img_42.jpg'
# path_to_image = 'result_image\icdar_2013\img_110.jpg'
path_to_dir = 'result_image/icdar_2013/ocr'
path_to_dir_save = path_to_dir.replace('/ocr', '/no_padding/ocr')
path_to_images = glob.glob(os.path.join(path_to_dir, '*.jpg'))
for path_to_image in path_to_images:
    image = cv2.imread(path_to_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ở đây, chúng ta giả định rằng màu đen có giá trị pixel từ 0 đến 10 (có thể điều chỉnh tùy theo ảnh)
    threshold = 10

    rows = np.where(np.max(gray, axis=1) > threshold)[0]
    cols = np.where(np.max(gray, axis=0) > threshold)[0]

    cropped_image = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    ### save to the dir + no_padding with the basename +no_padding
    path_to_image_save = os.path.join(path_to_dir_save, os.path.basename(path_to_image).replace('.jpg', '_no_padding.jpg'))
    cv2.imwrite(path_to_image_save, cropped_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
