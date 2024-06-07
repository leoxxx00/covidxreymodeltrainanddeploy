from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model/covid_detector_model.h5')

def predict_covid(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']
    return class_labels[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            upload_folder = 'static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)
            result = predict_covid(filepath)
            return render_template('result.html', result=result, img_path='uploads/' + file.filename)
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

if __name__ == "__main__":
    app.run(debug=True)
