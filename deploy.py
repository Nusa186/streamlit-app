import PIL
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('image_classifier.h5')

class_indices = np.load('class_dataset.npy', allow_pickle=True).item()

class_names = {v: k for k, v in class_indices.items()}

@app.route('/')
def index():
    return 'Hello, world'

@app.route('/predict-single', methods=['GET', 'POST'])
def predict_single():
    imgFile = request.files.get('imgFile[]')

    if imgFile:
        img_path = "./db/" + imgFile.filename
        imgFile.save(img_path)

        img = load_img(img_path, target_size=(150, 150))
        x = img_to_array(img) / 255.
        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x)

        top_class = np.argmax(prediction)
        top_prob = prediction[0][top_class]

        top_class_name = class_names[top_class]

        result = {
            'input': imgFile.filename,
            'category': top_class_name,
            'prediction': f'{top_prob * 100:.2f}'
        }

        return jsonify({'status': 'SUCCESS', 'result': result}), 200
    else:
        return jsonify({'status': 'ERROR', 'message': 'No file provided.'}), 400

@app.route('/predict-multiple', methods=['GET', 'POST'])
def predict():
    files = request.files.getlist('imgFile[]')  # Menggunakan getlist untuk mengambil daftar file

    predictions = []

    for imgFile in files:
        img_path = "./db/" + imgFile.filename
        imgFile.save(img_path)  # Simpan file ke disk

        img = load_img(img_path, target_size=(150, 150))

        x = img_to_array(img) / 255.
        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x)

        top_class = np.argmax(prediction)
        top_prob = prediction[0][top_class]
        
        top_class_name = class_names[top_class]

        predictions.append({
            'input': imgFile.filename,
            'category': top_class_name,
            'prediction': f'{top_prob * 100:.2f}'
        })

    return jsonify({'status': 'SUCCESS', 'predictions': predictions}), 200  # Format JSON untuk respons
if __name__ == '__main__':
    app.run(debug=True)
