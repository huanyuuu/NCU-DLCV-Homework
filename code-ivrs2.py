import numpy as np
import cv2
import os
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
from numpy import expand_dims
import numpy as np
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras import backend as K
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
import math
from sklearn.model_selection import StratifiedKFold
from keras.layers import Conv2D

def load_images_from_folder(folder_path,label_path):
    y_all = pd.read_csv(label_path)
    images = []
    labels = []
    check_order=[]
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename),0)
        lb = y_all[y_all['image']==filename]['label2_p'].values[0]
        img = cv2.resize(img, (704, 576), interpolation=cv2.INTER_AREA)
        if img is not None:
            images.append(img)
            labels.append(lb)
            check_order.append(filename)
    return images,labels,check_order

def load_real_samples():
    x,y,o = load_images_from_folder('../input/stonecnn/train/train','../input/stonecnn/train.csv')
    x = expand_dims(x, axis=-1)
    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32')
    x = x/255
    x = np.repeat(x, 3, -1)
    print(x.shape, y.shape)
    return [x, y]

allx,ally = load_real_samples()

x,y,o=load_images_from_folder('../input/stonecnn/train/train','../input/stonecnn/train.csv')
x_t,y_t,o_t=load_images_from_folder('../input/stonecnn/test (1)/test','../input/stonecnn/test.csv')
x = expand_dims(x, axis=-1)
x = np.array(x)
x_t = np.array(x_t)
x = x/255
x_test = x_t/255
x = np.repeat(x, 3, -1)
in_shape =  x.shape[1:4]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = np.array(y_train).reshape(-1,1)
y_valid = np.array(y_valid).reshape(-1,1)
y_test = np.array(y_t).reshape(-1,1)
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)


input_tensor = Input(shape=in_shape) 
model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False
                                               ,input_tensor = input_tensor)
m = model.output
m = GlobalAveragePooling2D()(m)
m = Dense(50, activation='relu')(m)
predictions = Dense(3, activation='softmax')(m)
model = Model(inputs=model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.001), 
              loss='categorical_crossentropy',metrics = 'accuracy')


checkpoint_cb = keras.callbacks.ModelCheckpoint("iv3_m.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 4, restore_best_weights=True)
model.fit(x_train,
               y_train,
               batch_size=6,
               epochs=50,
               callbacks=[checkpoint_cb, early_stopping_cb],
               validation_data=(x_valid,y_valid),
               verbose=1)

model_m = keras.models.load_model('iv3_m.h5')
prediction = model_m.predict(allx)
prediction_c = []
for i in range(0,len(ally)):
    m = prediction[i].max()
    rs = [i for i, j in enumerate(prediction[i]) if j == m][0]
    prediction_c.append(rs)

val_res = pd.DataFrame(prediction_c,columns=['pred'])
val_real = pd.DataFrame(ally,columns=['real'])
val_all = val_res.join(val_real)

acc = accuracy_score(val_all['real'],val_all['pred'])
rec = recall_score(val_all['real'],val_all['pred'],average='macro')
prec = precision_score(val_all['real'],val_all['pred'],average='macro')

print('Acc: '+str(acc)+' Rec: '+str(rec)+' Prec: '+str(prec))

