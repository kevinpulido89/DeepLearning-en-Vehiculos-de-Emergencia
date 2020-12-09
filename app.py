from flask import Flask, jsonify, render_template, request, redirect, url_for
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import shutil
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/target/img"

@app.route('/predict')
def predict(clases):

    ImgGenerator = ImageDataGenerator(rescale = 1./255)

    classificador = load_model('best_cls.h5')

    target_datagen = ImgGenerator.flow_from_directory(batch_size=1,
                                                      directory='static/target',
                                                      target_size=(256, 256),
                                                      class_mode=None)

    result = classificador.predict(next(target_datagen))

    return f'{clases[np.argmax(result[0])]}: {max(result[0]*100):0.2f} %'

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']

        if img_file:
            img_location = os.path.join(UPLOAD_FOLDER, img_file.filename)
            img_file.save(img_location)
            start = time.time()
            pred = predict(categorias)
            print(f'Tiempo de ejecucion: {time.time()-start} -- Resultado: {pred}')
            
            # Move a file from the directory d1 to d2
            new_location = os.path.join("static/", img_file.filename)
            shutil.move(img_location, new_location)

            return render_template('index.html', prediction=pred, image_loc=img_file.filename)

    return render_template('index.html', prediction='', image_loc=None)

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    categorias = {0: 'Ambulancia',
                  1: 'Bomberos',
                  2: 'Patrulla',
                  3: 'Otro tipo'}

    app.run(debug=True)