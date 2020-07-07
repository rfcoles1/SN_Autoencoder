import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from torch.utils.data import DataLoader
from torch.autograd import Variable

from Payne import *


def Conv1DTrans(inp, filters, kernel_size, strides, activation=tf.nn.relu, padding='valid'):

    x = layers.Lambda(lambda x: K.expand_dims(x, axis=2))(inp)
    x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), \
        strides=(strides, 1), kernel_initializer='glorot_normal', \
        activation=activation, padding=padding)(x)
    x = layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)

    return x

"""
def loss_ae(y_true, y_pred):
    return tf.math.reduce_mean((y_pred - y_true)**2)
"""
def loss_ae_mask(y_true, y_pred, mask, error):
    return tf.math.reduce_mean(((y_pred-y_true)*mask/error)**2)

"""
def loss_enc(y_true, y_pred):
    return tf.keras.losses.MSE(y_pred, y_true)
    
def loss_dec(y_true, y_pred):
    return tf.math.reduce_mean((y_pred - y_true)**2)
"""
def loss_mse(y_true, y_pred):
    return tf.math.reduce_mean((y_pred - y_true)**2)


class Network():
    def __init__(self, num_z=10):
        self.batch_size = 4
        self.num_z = num_z
        self.checkpoint = 1000
        self.lr_enc = 0.0001
        self.lr_dec = 0.0001
        self.lr_ae = 0.0001

        self.losses_path = './records/losses_tmp.pickle' 
        self.checkpoint_path = './records/cp_tmp'

        self.en_input = layers.Input(shape=(7167,1), name='encoder_input')
        self.encoder = tf.keras.models.Model(self.en_input, self.Encoder(self.en_input), name='encoder')

        self.de_input = layers.Input(shape=(25+self.num_z), name='dececoder_input')
        self.decoder = tf.keras.models.Model(self.de_input, self.Decoder(self.de_input), name='decoder')
   
        self.mask = tf.keras.layers.Input(shape=(7167,1), name='mask_input')
        self.err = tf.keras.layers.Input(shape=(7167,1), name='err_input')

        self.op_enc = tf.keras.optimizers.Adam(learning_rate=self.lr_enc)
        self.op_dec = tf.keras.optimizers.Adam(learning_rate=self.lr_dec)
        self.op_ae = tf.keras.optimizers.Adam(learning_rate=self.lr_ae)

        self.labels_payne = np.load('./data/mock_all_spectra_no_noise_resample_prior_large.npz')['labels'].T
        self.perturbations = [100., 0.1, 0.2, *np.repeat(0.1, 20), 5., 2.]
        self.emulator, self.y_min, self.y_max = build_emulator('./data/PAYNE.pth.tar')

        self.norm_data = np.load('./data/normalization_data.npz')
        self.x_mean = torch.Tensor(self.norm_data['x_mean'].astype(np.float32))
        self.x_std = torch.Tensor(self.norm_data['x_std'].astype(np.float32))

        self.obs_dataset = PayneObservedDataset('./data/aspcapStar_dr14.h5', obs_domain='APOGEE',\
                dataset='train', x_mean=self.x_mean, x_std=self.x_std, collect_x_mask=True)
        self.obs_train_dataloader = DataLoader(self.obs_dataset, batch_size=self.batch_size,\
                        shuffle=True, drop_last=True)
        
        self.reset()

    def reset(self):
        self.losses = defaultdict(list)
        self.curr_epoch = 0


    def save(self):
        f = open(self.losses_path, 'wb')
        pickle.dump(self.losses,f)
        f.close()
        self.encoder.save_weights(self.checkpoint_path + 'enc')
        self.decoder.save_weights(self.checkpoint_path + 'dec')

    def load(self):
        f = open(self.losses_path, 'rb')
        self.losses = pickle.load(f)
        f.close()
        self.curr_epoch = self.losses['iterations'][-1]
        self.encoder.load_weights(self.checkpoint_path + 'enc')
        self.decoder.load_weights(self.checkpoint_path + 'dec')


    def Encoder(self,y):
        x = layers.Conv1D(filters=32, kernel_size=7, strides=4, \
            kernel_initializer='glorot_normal', activation=tf.nn.relu)(y)
        x = layers.Conv1D(filters=64, kernel_size=7, strides=4, \
            kernel_initializer='glorot_normal', activation=tf.nn.relu)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, strides=4, \
            kernel_initializer='glorot_normal', activation=tf.nn.relu)(x)
        x = layers.Conv1D(filters=256, kernel_size=7, strides=4, \
            kernel_initializer='glorot_normal', activation=tf.nn.relu)(x)
        x = layers.Conv1D(filters=512, kernel_size=7, strides=4, \
            kernel_initializer='glorot_normal', activation=tf.nn.relu)(x)

        x = layers.Flatten()(x)
        outy = layers.Dense(25)(x)
        outz = layers.Dense(self.num_z)(x)
        return outy,outz

    def Decoder(self,x):
        y = layers.Dense(3072)(x)
        y = tf.reshape(y,[-1,6,512])

        y = Conv1DTrans(y,filters=512, kernel_size=7, strides=4)
        y = Conv1DTrans(y,filters=256, kernel_size=7, strides=4)
        y = Conv1DTrans(y,filters=128, kernel_size=7, strides=4)
        y = Conv1DTrans(y,filters=64, kernel_size=7, strides=4)
        y = Conv1DTrans(y,filters=32, kernel_size=7, strides=4)

        outx = Conv1DTrans(y,filters=1, kernel_size=1, strides=1, activation=None)
        return outx


    def get_batch_synth(self, N=4, norm_param=True, norm_spec=True):
        y = np.copy(self.labels_payne[np.random.randint(len(self.labels_payne), size=N)])
        y += np.array([np.random.uniform(-1*p, p, size=N) for p in self.perturbations]).T
        for i in [2,23,24]:
            y[y[:,i]<np.min(self.labels_payne[:,i]),i] = np.min(self.labels_payne[:,i])

        y = Variable(torch.Tensor(y.astype(np.float32)))
        if norm_param:
            y = (y - self.y_min)/(self.y_max - self.y_min) - 0.5

        x = self.emulator(y)
        if norm_spec:
            x = (x - self.x_mean) / self.x_std
        x = x[:,47:]

        return x, y

    def get_batch_obs(self, N=4):
        loader = DataLoader(self.obs_dataset, batch_size=N,\
            shuffle=True, drop_last=True)
        batch = next(iter(loader))
        
        inp = np.reshape(batch['x'], (N,7167,1)).numpy()
        mask = np.reshape(batch['x_msk'], (N,7167,1)).numpy()
        err = np.reshape(batch['x_err'], (N,7167,1)).numpy()

        return inp,mask,err
