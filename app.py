import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_PATH = 'model_nematik_mobilenetv2.h5'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)

# Sesuaikan urutan ini dengan urutan folder saat training (biasanya alfabetis)
class_names = ['Fluctuating Williams Domain', 'Grid Pattern', 'Williams Domain']

fase_info = {
    'Williams Domain': 'Fase awal di mana pola garis paralel yang stabil muncul setelah melewati ambang batas tegangan tertentu.',
    'Fluctuating Williams Domain': 'Fase transisi di mana pola garis mulai bergetar secara dinamis akibat kenaikan tegangan.',
    'Grid Pattern': 'Fase lanjutan di mana pola berubah menjadi struktur kisi-kisi atau kotak-kotak yang teratur.'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 7, 50, 80)
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 1
    )
    # Ubah kembali ke RGB untuk input MobileNet
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(thresh_rgb)
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return render_template('index.html')
        file = request.files['file']
        if file.filename == '': return render_template('index.html')

        if file:
            # Simpan file ke static/uploads
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            # Prediksi
            processed_img = preprocess_image(path)
            prediction = model.predict(processed_img)[0]
            
            # Susun hasil
            all_predictions = []
            for i in range(len(class_names)):
                all_predictions.append({
                    'name': class_names[i],
                    'prob': round(float(prediction[i]) * 100, 2),
                    'desc': fase_info.get(class_names[i], '')
                })
            
            # Urutkan dari yang tertinggi
            all_predictions = sorted(all_predictions, key=lambda x: x['prob'], reverse=True)
            top_result = all_predictions[0]

            return render_template('index.html', 
                                   top_result=top_result, 
                                   all_predictions=all_predictions,
                                   image_path=filename) # Mengirim nama file

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
