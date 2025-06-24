from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from utils import classify_single_ultrasound_image


app = Flask(__name__)
# model = tf.keras.models.load_model('breast_cancer_mnodel.h5')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        patient_id = request.form['id']
        file = request.files['image']
        mask = request.files['mask']
        if file and name and patient_id and mask:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            mask_filename = secure_filename(mask.filename)
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            mask.save(mask_path)
            
            pred = classify_single_ultrasound_image(filepath, mask_path)
            class_idx = np.argmax(pred)
            classes = ['benign', 'malignant', 'normal']
            result = {
                'name': name,
                'id': patient_id,
                'diagnosis': classes[class_idx],
                'confidence': f"{np.max(pred)*100:.2f}%",
                'image': filename
            }
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
