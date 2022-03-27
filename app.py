import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
from keras.models import load_model
import numpy as np
import tensorflow as tf
import pickle
#from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.metrics import accuracy_score
import sys
import sounddevice as sd
import wavio
import time
import librosa
import pandas as pd
#import serial #untuk arduino
import socket


app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'
# Path to the folder where we store the defferent pickles
PICKLE_FOLDER = STATIC_FOLDER + '/pickles'
# Path to the folder where we store the record before predictrion
RECORD_FOLDER = STATIC_FOLDER + '/record'


def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    global model
    #model = tf.keras.models.load_model(MODEL_FOLDER + '/catsVSdogs.h5')
    #model = load_model(MODEL_FOLDER + '/cat_dog_classifier.h5')
    #model = load_model(MODEL_FOLDER + '/time.h5')
    #model = load_model(MODEL_FOLDER + '/time.model')
    model = load_model("time.h5")
    print(model)
    global graph
    #graph = tf.compat.v1.get_default_graph()
    graph = tf.get_default_graph()
    print('[INFO] : Model loaded')


def predict(fullpath):
    # Ambil data 
    data = image.load_img(fullpath, target_size=(128, 128, 3))
    # (150,150,3) ==> (1,150,150,3)
    data = np.expand_dims(data, axis=0)
    # Scaling
    data = data.astype('float') / 255

    # Prediction

    with graph.as_default():
        result = model.predict(data)

    return result

def build_predictions(fullpath):
    global y_pred
#    y_true = []
    y_pred = []
    fn_prob = {}
    y_prob = []
#    nilai_prediksi = []
    
    nfilt = 26
    nfeat = 13
    nfft = 512
    rate = 16000
    step = int(rate/10)
    
    rate, wav = wavfile.read(fullpath)
    print('[INFO] : Loading File..........')
    _min, _max = float('inf'), -float('inf')
    #label = fn2class[fn]
    #print("Label : ", label)
    #c= classes.index(label)
    #print('Classes :', c)
        
        
    for i in range(0, wav.shape[0]-step, step):
        sample = wav[i:i+step]
        x = mfcc(sample, rate, numcep=nfeat, nfilt=nfilt, nfft=nfft)
        _min = min(np.amin(x), _min)
        _max = max(np.amax(x), _max)
        x = (x - _min) / (_max - _min)   
        x = np.expand_dims(x, axis=0)
        #print('data X : ',x)

        with graph.as_default():
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            #y_true.append(c)
            y_pred.append(np.argmax(y_hat))
            fn_prob = np.mean(y_prob, axis=0).flatten(0)
#       nilai_prediksi = fn_prob[fn]
#       print("ini adalah fn_prob : ",fn_prob)
#       print()
#       print(y_hat)

    return y_pred, fn_prob

def save_wav(wavfilename):
    global file_wav
    file_wav = wavfilename
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    #start record
    client_socket.sendto(b'1', ("192.168.43.183",4321))
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait(timecountdown(seconds))  # Wait until recording is finished
    client_socket.sendto(b'2', ("192.168.43.183",4321))
    #stop record
    
    #print('[INFO] : Audio Play......')
    #sd.play(myrecording)
    # write audio file
    final_save = wavio.write("static/record/"+file_wav+'.wav', myrecording,fs,sampwidth=2)
    print('[INFO] : Audio Saved.....')
    #fullname = os.path.join(RECORD_FOLDER, final_save)
    #file_wav.save(fullname)
    #print("fullname : ", fullname)
    signal, rate = librosa.load("static/record/"+file_wav+'.wav',sr = 16000)
    mask = envelope(signal, rate, 0.0005)
    namafile = file_wav+'Clean'+'.wav'
    wavfile.write("static/record/CleanRecord/"+namafile, rate=rate, data=signal[mask])
    #rate, dataMask = wavfile.read(file_wav+'.wav')
    #print(data)
    print('[INFO] : Audio Mask Saved.....')
    return file_wav

def timecountdown(uin):
    when_to_stop = abs(int(uin))
    while when_to_stop > 0:
        m, s = divmod(when_to_stop, 60)
        h, m = divmod(m, 60)
        time_left = str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)
        print(time_left + "\r", end="")
        time.sleep(1)
        when_to_stop -= 1

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# Arduino Config

#@app.before_first_request
#def setupSerial():
#    global arSerial
    #arSerial=arduinoSerial.Arduino(57600,'*',0)
#    arSerial = serial.Serial('\\\\.\\COM9',9600)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['audio']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)
        print("[Audio] : ", fullname)
        labeldata = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        fn_prob = build_predictions(fullname)
        value1 = fn_prob[1][0]
        value2 = fn_prob[1][1]
        
            
        if value1 < value2:
            label = 'GOOD'
            acc_score = ("%.0f%%" % (accuracy_score(y_true=labeldata, y_pred=y_pred) * 100))
            #acc_score = 100
            print('[INFO] : Class Egg Is Good')
        elif value1 > value2:
            label = 'BAD'
            acc_score = ("%.0f%%" % (accuracy_score(y_true=labeldata, y_pred=y_pred) * 100))
            #acc_score = 100
            print('[INFO] : Class Egg Is Bad')

        return render_template('predict.html', audio_file_name=file.filename, label=label, accuracy=acc_score)


# Record Page
@app.route('/recordpage')
def recordpage():
    return render_template('recordwav.html')

@app.route('/recordwav', methods=['GET', 'POST'])
def record_audio():
    if request.method == 'GET':
        render_template('recordwav.html')
    else:
        filename_wav = request.form['audio_name']
        print('[Name File] : ', filename_wav)
        
        filename = save_wav(filename_wav)
        #-------------------------------------     

    return render_template('recordwav.html')
        
@app.route('/recordpredict', methods=['GET', 'POST'])
def record_predict():
    if request.method == 'GET':
        render_template('recordwav.html')
    else:
        pathRecord = ('static/record')
        indexRecord = len(os.listdir(pathRecord)) + 1
        
        filename_wav = request.form['Predict']
        filename_wav = filename_wav + str(indexRecord)
        print('[Name File] : ', filename_wav)

        
        filename = save_wav(filename_wav)
        file = file_wav+'Clean'+'.wav'
        print(file)
        filePath = "static/record/CleanRecord/"+file
        print("filePath :",filePath)
        
        #fn_prob = build_predictions(filePath)
        #print("Predict Berhasil")
        #labeldata = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        #fn_prob = build_predictions(fullname)
        #value1 = fn_prob[1][0]
        #value2 = fn_prob[1][1]
        
            
        #if value1 < value2:
        #    label = 'GOOD'
        #    acc_score = ("%.0f%%" % (accuracy_score(y_true=labeldata, y_pred=y_pred) * 100))
        #    print('[INFO] : Class Egg Is Good')
        #elif value1 > value2:
        #    label = 'BAD'
        #    acc_score = ("%.0f%%" % (accuracy_score(y_true=labeldata, y_pred=y_pred) * 100))
        #    print('[INFO] : Class Egg Is Bad')

        return render_template('recordpredict.html', audio_file_name=filename_wav+'.wav')
        #return render_template('predict.html', audio_file_name=file.filename, label=label, accuracy=acc_score)

        
@app.route('/raspberrypi', methods=['GET', 'POST'])
def raspberrypi():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if request.form['buttonAr'] == 'START-ENGINE':
        inputdata = (1)
        data = bytes(inputdata)
        client_socket.sendto(b'1', ("192.168.43.183",4321))
        #arSerial.write(b'1')
        print('Motor Berjalan')
    elif request.form['buttonAr'] == 'STOP':
        client_socket.sendto(b'2', ("192.168.43.183",4321))
        #arSerial.write(b'3')
        print('Motor & Servo Stop') 

    return render_template('recordwav.html')
    
#@app.route('/arduino', methods=['GET', 'POST'])
#def arduino():
#    if request.form['buttonAr'] == 'START-ENGINE':
#        arSerial.write(b'1')
#        print('Motor Berjalan')
#    elif request.form['buttonAr'] == '2':
#        arSerial.write(b'2')
#        print('Servo On')
#    elif request.form['buttonAr'] == 'STOP':
#        arSerial.write(b'3')
#        print('Motor & Servo Stop') 

#    return render_template('recordwav.html')


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/record/<filename>')
def record_file(filename):
    return send_from_directory(RECORD_FOLDER, filename)

def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
