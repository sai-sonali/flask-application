from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
#from mdl import *
import numpy as np
import pandas as pd
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
# def modl(audio_file,csvfile):
#     signal, sr = librosa.load(audio_file)
#     #extracting MFCC's
#     mfccs = librosa.feature.mfcc(y=signal, n_mfcc=39, sr=sr)
#     fr=len(mfccs[0])
#     df=pd.read_csv(csvfile)

#     GXmfcc=[]
#     mfcc=mfccs.transpose()
#     mfcc=mfcc.tolist()
#     #print(len(mfcc[38]))
#     GXmfcc.extend(mfcc)


#     stime=list(df['Start Time'])
#     etime=list(df['End Time '])
#     code=list(df['Class'])
#     print(len(code))
#     for i in range(len(code)):
#         if(code[i]>0):
#             code[i]=1
#     # ad=11.610045
#     ad=8.77960
#     t=ad/fr
#     t1=0
#     t2=ad/fr
#     y=[]
#     l=len(stime)
#     i=0
#     cnt=0
#     while(t2<ad and t1<=t2):
#         if(i<l-1):
#             while(t1<stime[i] and t2<stime[i]):
#                 y.append(0)
#                 t1+=t
#                 t2+=t
#             while((t1>=stime[i] and t2<=etime[i]) or (t1<stime[i] and t2>=stime[i]) or (t1<etime[i] and t2>etime[i] and t2<stime[i+1])):
#                 y.append(code[i])
#                 t1+=t
#                 t2+=t
#             i+=1
#         else:
#             y.append(0)
#             t1+=t
#             t2+=t
#     X=pd.DataFrame(GXmfcc)
#     Y=pd.DataFrame(y)

#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#     clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None)
#     clf.fit(X_train,Y_train)
#     # Use the trained classifier to predict the labels of the test set
#     y_pred = clf.predict(X_test)
#     #audio_file=path
#     x1,sr1= librosa.load(audio_file)
#     mfccs1=librosa.feature.mfcc(y=x1, n_mfcc=39, sr=sr1)
#     mfcc1=mfccs1.transpose()
#     mfcc1=mfcc1.tolist()
#     y1=[]
#     for i in mfcc1:
#         i=[i]
#         y_pred=clf.predict(i)
#         y_pred=y_pred.tolist()
#         y1.append(y_pred[0])
#     print(y1.count(1))
#     if(y1.count(1)>1):
#         return ("Stuttered speech")
#     else:
#         return ("Normal speech")
#modl("static/files\GPS _school_voice_samples srinivas_g22.wav ","static/files\srinivas_g22.csv")
def generate(path):
    audio_path=path
    with open('model_saved' , 'rb') as f:
        clf = pickle.load(f)
    x1,sr1= librosa.load(audio_path)
    mfccs1=librosa.feature.mfcc(y=x1,sr=sr1,n_mfcc=39)
    mfcc1=mfccs1.transpose()
    mfcc1=mfcc1.tolist()
    y1=[]
    for i in mfcc1:
        i=[i]
        y_pred=clf.predict(i)
        y_pred=y_pred.tolist()
        y1.append(y_pred[0])
    print(y1.count(1))
    if(y1.count(1)>1):
        return ("Uploaded voice clip is Stuttered speech")
    else:
        return ("Uploaded voice clip is Normal speech")
app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER


# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('sample.html')
@app.route('/record')
def index1():
    # Set The upload HTML template '\templates\index.html'
    return render_template('recorder.html')

# Get the uploaded files
@app.route("/", methods=['POST'])
def uploadFiles():
      # get the uploaded file
      uploaded_file = request.files['file']
      
      file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
      uploaded_file.save(file_path)
      
          # save the file
      #print(file_path,file_path2)
    #   result=modl(file_path,file_path2)
      #print(uploaded_file)
      result=generate(file_path)
      return render_template('sample.html',result=result)
      #return redirect(url_for('index'))

if (__name__ == "__main__"):
     app.run(debug=True)