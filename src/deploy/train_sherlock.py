import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader

from src.models.dataset import WDCDataset, ToTensor
from src.models.model_construction import NNModelConstruction
from src.models.sherlock_model import Sherlock

SEED = 13


# Input: X_train and X_val numpy ndarray as returned by build_features,
#        y_train and y_val arrays of labels,
#        nn_id indicating whether to take a retrained model or sherlock
# Output: Stored retrained model
def train_sherlock(X_train, y_train, X_val, y_val, nn_id):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    np.save('../src/deploy/classes_{}.npy'.format(nn_id), encoder.classes_)

    y_train_int = encoder.transform(y_train)
    y_train_cat = to_categorical(y_train_int)

    try:
        y_val_int = encoder.transform(y_val)
        y_val_cat = to_categorical(y_val_int)
    except ValueError:
        print('Validation labels should only contain labels that exist in deploy file.')

    lr = 0.0001
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    # Load Sherlock model architecture
    file = open('../src/models/sherlock_model.json', 'r')
    sherlock_file = file.read()
    sherlock = model_from_json(sherlock_file)
    file.close()

    print(sherlock.summary())

    # Compile Sherlock
    sherlock.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])

    print('Successfully loaded and compiled model, now fitting.')

    # Fit Sherlock to new data
    sherlock.fit(X_train, y_train_cat,
                 validation_data=(X_val, y_val_cat),
                 callbacks=callbacks, epochs=100, batch_size=256)

    # Save model and weights
    model_json = sherlock.to_json()
    with open('../src/models/{}_model.json'.format(nn_id), 'w') as json:
        json.write(model_json)

    sherlock.save_weights('../src/models/{}_weights.h5'.format(nn_id))
    print('Retrained Sherlock.')


def train_predict_sherlock(x_train, y_train, x_val, y_val, nn_id):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    np.save('../src/deploy/classes_{}.npy'.format(nn_id), encoder.classes_)

    y_train_int = encoder.transform(y_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    try:
        y_val_int = encoder.transform(y_val)
        y_val_cat = tf.keras.utils.to_categorical(y_val_int)
    except ValueError:
        print('Validation labels should only contain labels that exist in deploy file.')

    lr = 0.0001
    epochs = 100
    results = []

    train_data = WDCDataset(x_train, y_train_cat, transform=ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    print("The length of the train dataset is {}".format(len(train_data)))

    validation_data = WDCDataset(x_val, y_val_cat, transform=ToTensor())
    validation_loader = DataLoader(dataset=validation_data, batch_size=32, shuffle=False)
    print("The length of the validation dataset is {}".format(len(train_data)))

    m = NNModelConstruction(train_loader, validation_loader, Sherlock(10), lr, epochs)
    m.train()

    # accuracies = [0]
    # for e in range(epochs):
    #     m.train_epoch()
    #     accuracy = m.eval()
    #     print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")
    #     accuracies.append(accuracy)
    # results.append(accuracies)
