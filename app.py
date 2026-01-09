from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# 1. Load Model
model = tf.keras.models.load_model('model_nematik_mobilenetv2.h5')
class_names = ['WD', 'FWD', 'GP'] 
IMG_SIZE = 224

def preprocess_image(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # ALUR OPENCV (sama seperti training model)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=80)
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 1
    )
    
    # Konversi ke format yang dimengerti MobileNetV2
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(thresh_rgb)
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join('uploads', file.filename)
            file.save(path)
            
            # Prediksi
            processed_img = preprocess_image(path)
            prediction = model.predict(processed_img)
            result = class_names[np.argmax(prediction)]
            
            return f"Hasil Prediksi: {result}"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
