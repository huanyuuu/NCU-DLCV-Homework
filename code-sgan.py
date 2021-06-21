# %% [code] {"execution":{"iopub.status.busy":"2021-06-21T06:11:42.901355Z","iopub.execute_input":"2021-06-21T06:11:42.902020Z","iopub.status.idle":"2021-06-21T06:11:50.931069Z","shell.execute_reply.started":"2021-06-21T06:11:42.901901Z","shell.execute_reply":"2021-06-21T06:11:50.930001Z"}}
import numpy as np
import cv2
import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score
import tensorflow.keras as keras

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

input_img_size = (576, 704, 3)
class_type = 3

def load_images_from_folder(folder_path,label_path):
    y_all = pd.read_csv(label_path)
    images = []
    labels = []
    check_order=[]
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename),0)
        lb = y_all[y_all['image']==filename]['label2_j'].values[0]
        img = cv2.resize(img, (704, 576), interpolation=cv2.INTER_AREA)
        if img is not None:
            images.append(img)
            labels.append(lb)
            check_order.append(filename)
    return images,labels,check_order

def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

def load_real_samples():
    x,y,o = load_images_from_folder('../input/stonecnn/train/train','../input/stonecnn/train.csv')
    x = expand_dims(x, axis=-1)
    x = np.array(x)
    y = np.array(y)
    x = x.astype('float32')
    x = (x - 127.5) / 127.5
    return [x, y]

class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding
        })
        return config

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def define_discriminator(in_shape=input_img_size, n_classes=3):
 
    in_image = Input(shape=in_shape)

    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Flatten()(fe)

    fe = Dropout(0.4)(fe)

    fe = Dense(n_classes)(fe)

    c_out_layer = Activation('softmax')(fe)

    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    d_out_layer = Lambda(custom_activation)(fe)

    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model

def define_generator(filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None):

    img_input = layers.Input(shape=input_img_size)
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)
    

    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))


    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))


    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))
   
 
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)   
    return model

def select_supervised_samples(dataset, n_samples=15, n_classes=3):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):       
        X_with_class = X[y == i]       
        ix = randint(0, len(X_with_class), n_per_class)       
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)

def generate_real_samples(dataset, n_samples):   
    images, labels = dataset   
    ix = randint(0, images.shape[0], n_samples)    
    X, labels = images[ix], labels[ix]  
    y = ones((n_samples, 1))
    return [X, labels], y

def generate_latent_points2(n_samples):
    z_input,_ = load_real_samples()
    index = np.random.choice(z_input.shape[0], n_samples, replace=False)  
    X = z_input[index]
    return X

def generate_latent_points(n_samples):
    z_input,_ = load_real_samples()
    index1 = np.random.choice(z_input.shape[0], n_samples, replace=False)
    index2 = np.random.choice(z_input.shape[0], n_samples, replace=False)
    m_input = z_input[index1]
    n_input = z_input[index2]
    a_input = m_input+n_input
    a_input = (a_input - a_input.min()) / (a_input.max() - a_input.min())
    a_input = a_input*2 - 1
    return a_input
    

def generate_fake_samples(generator,n_samples):
    z_input = generate_latent_points(n_samples)  
    images = generator.predict(z_input)
    y = zeros((z_input.shape[0], 1))
    return images, y

def generate_fake_samples_perf(generator,index):
    z_input,_ = load_real_samples()  
    X = z_input[index]
    images = generator.predict(X)
    y = zeros((X.shape[0], 1))
    return images, y


from matplotlib.pyplot import figure
def summarize_performance(step, g_model, c_model, dataset,index):
    X, _ = generate_fake_samples_perf(g_model,index)
    X = X * 127.5 + 127.5
    pyplot.figure(figsize=(20,20))
    for i in range(4):
        pyplot.subplot(2, 2, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    X, y = dataset
    _, acc = c_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    filename2 = 'g_model_%04d.h5' % (step+1)
    filename3 = 'c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

def train(g_model, d_model, c_model, gan_model, dataset,index ,n_epochs=60, n_batch=4):
    X_sup, y_sup = select_supervised_samples(dataset)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    loss_df = pd.DataFrame(columns=['step','c_loss','c_acc','d_loss_real','d_loss_fake','gan_loss'])
    for i in range(n_steps):
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], n_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        X_gan, y_gan = generate_latent_points(n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        loss_df = loss_df.append({'step':i+1,'c_loss':c_loss,'c_acc':c_acc*100,'d_loss_real':d_loss1,'d_loss_fake':d_loss2,'gan_loss':g_loss},ignore_index=True)
        if (i+1) % (bat_per_epo * 3) == 0:
            summarize_performance(i, g_model, c_model, dataset,index)
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
    return loss_df

dataset = load_real_samples()

## Plot 4 images
index = np.random.choice(47, 4, replace=False)
d_model, c_model = define_discriminator()
g_model = define_generator()
gan_model = define_gan(g_model, d_model)

loss_table = train(g_model, d_model, c_model, gan_model, dataset,index)
loss_table.to_csv('loss_table.csv',index=False)

x,y,o = load_images_from_folder('../input/stonecnn/test (1)/test','../input/stonecnn/test.csv')

x = expand_dims(x, axis=-1)
x = np.array(x)
x = x.astype('float32')
x = (x - 127.5) / 127.5

finalmodel = keras.models.load_model('c_model_0660.h5')
prediction = finalmodel.predict(x)

result = []
for i in range(0,len(prediction)):
    m = prediction[i].max()
    rs = [i for i, j in enumerate(prediction[i]) if j == m]
    result.append(rs)

val_res = pd.DataFrame(result,columns=['pred'])
val_real = pd.DataFrame(y,columns=['real'])
val_all = val_res.join(val_real)

acc = accuracy_score(val_all['real'],val_all['pred'])
rec = recall_score(val_all['real'],val_all['pred'],average='macro')
prec = precision_score(val_all['real'],val_all['pred'],average='macro')

z_input,_ = load_real_samples()  
X = z_input[index]
pyplot.figure(figsize=(20,20))
for i in range(4):
    pyplot.subplot(2, 2, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
filename1 = 'real_plot.png'
pyplot.savefig(filename1)
pyplot.close()

rec
prec
acc
