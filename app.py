from flask import Flask, request, jsonify, send_file, render_template, url_for, session
import os
from werkzeug.utils import secure_filename
from generateText import generate_fake_image, draw_rotated_box
from PIL import Image
import io
import numpy as np
import tempfile
import secrets
import ryolo_model
from OCR import OCR_model
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

from pyngrok  import ngrok
port=5000
ngrok.set_auth_token("2fwiKKl7mSGaJ4DqCuyRzhp2X0Z_7vQSNWucBdec1SWd9moCF")
public_url=ngrok.connect(port).public_url
app.secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = app.secret_key  # Replace with your actual secret key

app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder where uploaded images will be saved
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload - 16 MB

model_ryolo = ryolo_model.build_model(model_type='s')
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image, target_size):
    return ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)




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



@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/convert_to_bw', methods=['POST'])
# def convert_to_bw():
#     if 'image' not in request.files:
#         return 'No file part', 400
#     file = request.files['image']
#     if file.filename == '':
#         return 'No selected file', 400
#     if file:
#         image = Image.open(file.stream).convert('L')  # Chuyển ảnh thành trắng đen
#         img_io = io.BytesIO()  # Tạo một BytesIO để lưu ảnh
#         image.save(img_io, 'PNG')  # Lưu ảnh vào img_io
#         img_io.seek(0)
#         return send_file(img_io, mimetype='image/png')


@app.route('/convert_to_bw', methods=['POST'])
def convert_to_bw():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        image = Image.open(file.stream)
        word_length_min = int(request.form['word_length_min'])
        word_length_max = int(request.form['word_length_max'])
        font_size_min = int(request.form['font_size_min'])
        font_size_max = int(request.form['font_size_max'])
        angle_min = int(request.form['angle_min'])
        angle_max = int(request.form['angle_max'])
        word_count = int(request.form['word_count'])
        color = request.form.get('selected_color', None) 
        ## change the color to tuple
        color = tuple(map(int, color.split(',')))
        print(f"Word count: {word_count}")
        print(f"Angle min: {angle_min}, angle max: {angle_max}")
        print(f"Font size min: {font_size_min}, font size max: {font_size_max}")
        print(f"Word length min: {word_length_min}, word length max: {word_length_max}")
        fake_image, annotation = generate_fake_image(image,
                                                     text_color=color,
                                                     word_count=word_count,
                                                     font_size_min=font_size_min, font_size_max=font_size_max, 
                                                     word_length_min=word_length_min, word_length_max=word_length_max,
                                                     angle_min=angle_min, angle_max=angle_max)
        
        fake_image_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        session['fake_image_path'] = fake_image_path  # Convert numpy array to list before storing in session
        session['annotation'] = annotation.tolist()
        fake_image_text = Image.fromarray(fake_image.astype(np.uint8), 'RGB')
        fake_image_text.save(fake_image_path, 'PNG')
        img_io = io.BytesIO()
        fake_image_text.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

@app.route('/draw_label', methods=['POST'])
def draw_label():
    fake_image_path = session.get('fake_image_path')  # Convert list back to numpy array
    if fake_image_path:
        with open(fake_image_path, 'rb') as f:
            fake_image = np.array(Image.open(f))
            annot = np.array(session['annotation'])
            image_draw_label = draw_rotated_box(fake_image, annot[:, 0], annot[:, 1], annot[:, 2], annot[:, 3], annot[:, 4], annot[:, 5])
            image_draw_label = Image.fromarray(image_draw_label.astype(np.uint8), 'RGB')
            img_io = io.BytesIO()
            image_draw_label.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')

@app.route('/model', methods=['POST'])
def model():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))

        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        temp_jpg_path = os.path.splitext(temp_path)[0] + '.jpg'  # Change the extension to .jpg
        image.save(temp_jpg_path, 'JPEG')  # Save the image in JPEG format
        global model_ryolo
        model_type = request.form.get('model_type', 's')
        conf_threshold = float(request.form.get('confidence_threshold', 0.2))
        model_ryolo = ryolo_model.build_model(model_type)
        print(f"Model type: {model_type}, confidence threshold: {conf_threshold}")
        word_boxes, processed_image, _ = ryolo_model.model_predict(model_ryolo, temp_jpg_path, conf_threshold=conf_threshold)
        image_with_boxes, boxes = ryolo_model.draw_rotated_box(np.array(processed_image), word_boxes=word_boxes, gt_boxes=None, gt_label=None)
        image_with_boxes_pil = Image.fromarray(image_with_boxes.astype(np.uint8), 'RGB')
        global ocr
        recognized_texts = OCR_model(ocr, np.array(processed_image), boxes)
        print(f"Recognized texts: {recognized_texts}")
        
        image_with_boxes_pil = draw_text_on_image(image_with_boxes_pil, recognized_texts, boxes)

        img_io = io.BytesIO()
        image_with_boxes_pil.save(img_io, 'PNG')
        img_io.seek(0)
        os.remove(temp_jpg_path)
        os.rmdir(temp_dir)
        return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    print(f"Public URL: {public_url}")
    app.run(port=port)

    # app.run(debug=True)
    