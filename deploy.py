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
    imgFile = request.files['imgFile'] #postman key
    img_path = "./db/" + imgFile.filename
    imgFile.save(img_path) #save file to disk

    img = load_img(img_path, target_size=(150, 150))

    x = img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)

    top_class = np.argmax(prediction)
    top_prob = prediction[0][top_class]

    class_names = train_generator.class_indices
    class_names = dict((v,k) for k,v in class_names.items())
    top_class_name = class_names[top_class]

    # return render_template('index.html', predict_category=f'The model predicts that the image is of class {top_class_name}')
    return jsonify({'status':'SUCCES', 'input':imgFile.filename, 'category':top_class_name, 'prediction':f'{top_prob*100:.2f}'}), 200 #json format
if __name__ == '__main__':
    app.run(debug=True)