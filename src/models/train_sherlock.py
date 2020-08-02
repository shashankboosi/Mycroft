import os
import sys

import numpy as np
import tensorflow as tf
from .dataset import WDCDataset, ToTensor
from .model_construction import NNModelConstruction
from .sherlock_model import Sherlock
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

SEED = 13


# Input: X_train and X_val numpy ndarray as returned by build_features,
#        y_train and y_val arrays of labels,
#        nn_id indicating whether to take a retrained model or sherlock
# Output: Stored retrained model
def train_val_model(x_train, y_train, x_val, y_val, nn_id):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    np.save('./models/classes_{}.npy'.format(nn_id), encoder.classes_)

    y_train_int = encoder.transform(y_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    try:
        y_val_int = encoder.transform(y_val)
        y_val_cat = tf.keras.utils.to_categorical(y_val_int)
    except ValueError:
        print('Validation labels should only contain labels that exist in deploy file.')

    lr = 0.0001
    epochs = 100

    train_data = WDCDataset(x_train, y_train_cat, transform=ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    print("The length of the train dataset is {}".format(len(train_data)))

    validation_data = WDCDataset(x_val, y_val_cat, transform=ToTensor())
    validation_loader = DataLoader(dataset=validation_data, batch_size=256, shuffle=False)
    print("The length of the validation dataset is {}".format(len(train_data)))

    m = NNModelConstruction(train_loader, validation_loader, Sherlock(SEED), lr, epochs)
    m.train(save_cp=True)
