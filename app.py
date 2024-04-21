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


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'the random string'
app.config['SECRET_KEY'] = 'the random string'    # Replace with your actual secret key
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder where uploaded images will be saved
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload - 16 MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        fake_image, annotation = generate_fake_image(image)
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

        
        model = ryolo_model.build_model(model_type='s')
        word_boxes, processed_image, _ = ryolo_model.model_predict(model, temp_jpg_path)
        image_with_boxes = ryolo_model.draw_rotated_box(np.array(processed_image), word_boxes=word_boxes, gt_boxes=None, gt_label=None)
        image_with_boxes = Image.fromarray(image_with_boxes.astype(np.uint8), 'RGB')
        img_io = io.BytesIO()
        image_with_boxes.save(img_io, 'PNG')
        img_io.seek(0)
        os.remove(temp_jpg_path)
        os.rmdir(temp_dir)
        return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
    