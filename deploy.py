import PIL
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('image_classifier.h5')
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(r"C:\streamlit\Image_classify\train",
                                                    batch_size = 30,
                                                    class_mode = 'categorical', 
                                                    target_size = (150, 150))

@app.route('/')
def index():
    return 'Hello, world'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    files = request.files.getlist('imgFile')  # Menggunakan getlist untuk mengambil daftar file

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

        class_names = train_generator.class_indices
        class_names = dict((v, k) for k, v in class_names.items())
        top_class_name = class_names[top_class]

        predictions.append({
            'input': imgFile.filename,
            'category': top_class_name,
            'prediction': f'{top_prob * 100:.2f}'
        })

    return jsonify({'status': 'SUCCESS', 'predictions': predictions}), 200  # Format JSON untuk respons
if __name__ == '__main__':
    app.run(debug=True)
