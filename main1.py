import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
audio_files='/Users/sarvagnagudlavalleti/Desktop/GPS _school_voice_samples'
audio=os.listdir(audio_files)
audio.pop(0)
print(audio)
csv_files='/Users/sarvagnagudlavalleti/Desktop/csv files'
csv=os.listdir(csv_files)
csv.pop(4)
print(csv)
for i in range(len(audio)):
    audio[i]='/Users/sarvagnagudlavalleti/Desktop/GPS _school_voice_samples/'+audio[i]
for i in range(len(csv)):
    csv[i]='/Users/sarvagnagudlavalleti/Desktop/csv files/'+csv[i]
import librosa
import librosa.display
import pandas as pd
import numpy as np

GXmfcc = []
y = []
ad = [15.95, 14.85, 15.10, 15.89, 14.35, 10.78, 25.12, 25.17, 10.83, 10.31, 8.18, 14.17, 15.23, 10.85, 15.12, 20.33,
      16.95, 9.03, 13.86, 11.60]
for k in range(len(audio)):
    audio_path = audio[k]
    x, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=39)
    fr = len(mfccs[0])
    mfcc = mfccs.transpose()
    mfcc = mfcc.tolist()
    # print(len(mfcc[38]))
    GXmfcc.extend(mfcc)
    data = pd.read_csv(csv[k])
    stime = list(data['Start Time'])
    etime = list(data['End Time '])
    code = list(data['Class'])
    for m in range(len(code)):
        if (code[m] > 0):
            code[m] = 1
    adi = ad[k]
    t = adi / fr
    t1 = 0
    t2 = adi / fr
    l = len(stime)
    i = 0
    while (t2 < adi and t1 <= t2):
        if (i < l - 1):
            while (t1 < stime[i] and t2 < stime[i]):
                y.append(0)
                t1 += t
                t2 += t
            while ((t1 >= stime[i] and t2 <= etime[i]) or (t1 < stime[i] and t2 >= stime[i]) or (
                    t1 < etime[i] and t2 > etime[i] and t2 < stime[i + 1])):
                y.append(code[i])
                t1 += t
                t2 += t
            i += 1
        else:
            y.append(0)
            t1 += t
            t2 += t
print(len(y))
print(y)
y=np.array(y)
print(y)
y=y[:12661]
print(len(y))
X=pd.DataFrame(GXmfcc)
print(X)
Y=pd.DataFrame(y)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,random_state=42)
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None)
clf.fit(X_train,Y_train)
# Use the trained classifier to predict the labels of the test set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)

# Calculate precision
precision = precision_score(Y_test, y_pred)

# Calculate recall
recall = recall_score(Y_test, y_pred)

# Calculate F1 score
f1 = f1_score(Y_test, y_pred)

# Print the results
print("Accuracy: ", accuracy*100)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

print(y_pred)
# Use the trained classifier to predict the class probabilities of the test set
y_proba = clf.predict_proba(X_test)
y_pred=y_pred.reshape(-1,1)
# Print the class probabilities
pickle.dump(clf,open('/Users/sarvagnagudlavalleti/Downloads/A 2/model_saved','wb'))