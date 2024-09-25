import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder
import pickle
from werkzeug.utils import secure_filename
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình thư mục tải lên và các loại tệp được phép
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Đảm bảo thư mục tải lên tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Hàm kiểm tra loại tệp
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Tải mô hình và LabelEncoder khi khởi động ứng dụng
MODEL_PATH = 'model/my_resnet101_model.h5'
LE_PATH = 'model/label_encoder.pkl'

model = load_model(MODEL_PATH)

with open(LE_PATH, 'rb') as f:
    le = pickle.load(f)

# Hàm để trích xuất nhãn từ tên tệp
def extract_label(filename):
    """
    Giả định rằng nhãn lớp được mã hóa ở phần đầu của tên tệp, ví dụ:
    'class1_image1.jpg' sẽ trả về 'class1'
    """
    return filename.split('_')[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return redirect(request.url)
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            # Lưu tệp vào thư mục uploads
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Trích xuất nhãn thực tế từ tên tệp
            actual_label = extract_label(filename)
            
            # Đọc và xử lý hình ảnh
            img = cv2.imread(file_path)
            if img is None:
                prediction_label = "Không thể đọc được hình ảnh."
            else:
                img_resized = cv2.resize(img, (224, 224))
                img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
                
                # Dự đoán lớp
                prediction = model.predict(img_preprocessed)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                prediction_label = le.classes_[predicted_class_index]
            
            # Thêm kết quả vào danh sách
            results.append({
                'filename': filename,
                'prediction': prediction_label,
                'actual': actual_label
            })
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
