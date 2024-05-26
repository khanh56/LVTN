from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np

# Khởi tạo đối tượng OCR
ocr = PaddleOCR(use_angle_cls=True, lang='vi')  # use_angle_cls để xử lý văn bản có góc, thay đổi 'en' thành ngôn ngữ cần thiết

# Đọc ảnh
# image_path = 'static\images\img_423.jpg'
image_path = 'static\images\img_424.jpg'
# image_path = 'static\images\img_435.jpg'
# image_path = 'static\images\img_453.jpg'
# image_path = 'static\images\img_978.jpg'
# image_path = 'static\images\img_984.jpg'
# image_path = 'static\images\img_987.jpg'


image = cv2.imread(image_path)

# Thực hiện nhận dạng văn bản
results = ocr.ocr(image, cls=True)
# Hiển thị kết quả
boxes = [line[0] for line in results for line in line]  # Duyệt qua mỗi phần tử trong kết quả và lấy hộp giới hạn
texts = [line[1][0] for line in results for line in line]  # Lấy văn bản đã nhận dạng
scores = [line[1][1] for line in results for line in line]  # Lấy điểm độ tin cậy

# Tạo một bản sao của ảnh để vẽ kết quả
drawn_image = image.copy()
for box, text, score in zip(boxes, texts, scores):
    box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(drawn_image, [box], isClosed=True, color=(255, 0, 0), thickness=2)
    ((text_x, text_y), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_org = (box[0][0][0], box[0][0][1] - 10)
    cv2.putText(drawn_image, f'{text} ({score:.2f})', text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Hiển thị ảnh kết quả
cv2.imshow('Result', drawn_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
