from flask import Flask, render_template, request, send_from_directory
from flask_ngrok import run_with_ngrok
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil

assets_dir = 'assets'
if not os.path.exists(assets_dir):
    os.mkdir(assets_dir)
else:
    shutil.rmtree(assets_dir)
    os.mkdir(assets_dir)

os.mkdir(os.path.join(assets_dir, 'oct'))
os.mkdir(os.path.join(assets_dir, 'cxr'))

oct_model = load_model('models/oct_model.h5')
cxr_model = load_model('models/cxr_model.h5')

print("***** models loaded")

cnt_oct = 0
cnt_cxr = 0

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
run_with_ngrok(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/oct')
def oct_index():
    return render_template('oct_index.html')

@app.route('/oct_prediction', methods=['POST'])
def oct_prediction():
    global cnt_oct
    img = request.files['image']
    file = 'assets/oct/{}.jpg'.format(cnt_oct)
    img.save(file)
    
    # pre-process
    img_cv = cv2.imread(file)
    res = cv2.resize(img_cv, (128,128))
    cv2.imwrite(file, res)

    # img to array
    img_arr = cv2.imread(file)
    img_arr = cv2.resize(img_arr, (128,128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128,128,3)

    prediction = oct_model.predict(img_arr)

    cnv = round(prediction[0,0], 2)
    dme = round(prediction[0,1], 2)
    drusen = round(prediction[0,2], 2)
    normal = round(prediction[0,3], 2)
    
    preds = np.array([cnv, dme, drusen, normal])
    cnt_oct += 1
    return render_template('oct_prediction.html', data=preds)

@app.route('/cxr')
def cxr_index():
    return render_template('cxr_index.html')

@app.route('/cxr_prediction', methods=['POST'])
def cxr_prediction():
    global cnt_cxr
    img = request.files['image']
    file = 'assets/cxr/{}.jpg'.format(cnt_cxr)
    img.save(file)
    
    # pre-process
    img_cv = cv2.imread(file)
    res = cv2.resize(img_cv, (128,128))
    cv2.imwrite(file, res)

    # img to array
    img_arr = cv2.imread(file)
    img_arr = cv2.resize(img_arr, (128,128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128,128,3)

    prediction = cxr_model.predict(img_arr)

    pneumonia = round(prediction[0,0], 2)
    covid = round(prediction[0,1], 2)
    normal = round(prediction[0,2], 2)
    
    preds = np.array([pneumonia, covid, normal])
    cnt_cxr += 1
    return render_template('cxr_prediction.html', data=preds)


@app.route('/load_img_oct')
def load_img_oct():
    global cnt_oct
    return send_from_directory('assets/oct/', "{}.jpg".format(cnt_oct-1))

@app.route('/load_img_cxr')
def load_img_cxr():
    global cnt_cxr
    return send_from_directory('assets/cxr/', "{}.jpg".format(cnt_cxr-1))


if __name__ == '__main__':
    app.run()




