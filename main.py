# import sys
# print(sys.version_info)
# sudo chown -R keanp /home/keanp/* tambien cambiar kevinp8001

import os
import cv2
import numpy as np
import shutil
import time
from flask import Flask, jsonify, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

def YOLO_detector(net, img):

    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (512, 512), (0, 0, 0), True, crop=False) # (416, 416)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    coordsXY = []
    etiquetas = []

    for n, i in enumerate(range(len(boxes))):
        if i in indexes:
            print('n: ', n)
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in ['car', 'truck', 'bus']:
                color = colors[i]
                coordsXY.append([x,y])
                etiquetas.append(label)
                ymin = y
                ymax = y+h
                xmin = x
                xmax = x+w
                cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                print(label)
                img_name = label + '_' + str(i) + '.jpg'
                
                path = './static/target/img/'
                img_path = os.path.join(path, img_name)
                cv2.imwrite(img_path, cropped_img)

                cv2.rectangle(img, (x, y), (xmax, ymax), color, 2)
                
            else:
                continue
    
    return img, etiquetas, coordsXY


@app.route('/predict')
def predict(clases, num_images:int =1):

    ImgGenerator = ImageDataGenerator(rescale = 1./255)

    classificador = load_model('best_cls.h5')

    target_datagen = ImgGenerator.flow_from_directory(batch_size=num_images,
                                                      directory='/home/keanp/static/target',
                                                      target_size=(256, 256),
                                                      class_mode=None)

    result = classificador.predict(next(target_datagen))

    return f'{clases[np.argmax(result[0])]}: {max(result[0]*100):0.2f} %'

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']

        if img_file:
            start = time.time()
            img_location = os.path.join(UPLOAD_FOLDER, img_file.filename)
            img_file.save(img_location)

            IMAGE = cv2.imread(img_location)
 
            img, clases_yolo, coordsXY= YOLO_detector(net, IMAGE)

            # Saving the image  
            filename = 'DondeGuarda.jpg'
            cv2.imwrite(filename, img)
            print('\nImage Successfully saved\n')

            if ('car' or 'truck' or 'bus') in clases_yolo:
                print(True)

                pred = predict(categorias, len(clases_yolo))
                print(f'Tiempo de ejecucion: {time.time()-start} -- Resultado: {pred}')

                # Move a file from the directory d1 to d2
                new_location = os.path.join("/home/keanp/static/", img_file.filename)
                shutil.move(img_location, new_location)

                # cv2.putText(img, label, (x, y + 30), font, 1, color, 2)

                return render_template('index.html', prediction=pred, image_loc=img_file.filename)

            else:
                print(False)
                return render_template('index.html', prediction='No hay un vehículo en la imagen', image_loc='ghost.jpg')

    #  PONER ACA EL MODELO DE DECODER??? PUEDE AHORRAR TIEMPO
    return render_template('index.html', prediction='', image_loc=None)

if __name__ == '__main__':
    categorias = {0: 'Ambulancia',
                  1: 'Bomberos',
                  2: 'Patrulla',
                  3: 'Otro tipo'}
    print("IP: 35.215.234.57")
    print("Puerto: 8082")

    UPLOAD_FOLDER = "/home/keanp/static/IN"

    net = cv2.dnn.readNet('yolov4.weights','yolov4.cfg')

    print("=======================> INICIO <=======================")

    app.run(host='0.0.0.0', port=8082, debug=True)
