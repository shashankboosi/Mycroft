import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .dataset import WDCDataset, ToTensor
from .model_construction import NNModelConstruction
from .sherlock_model import Sherlock

sys.path.append(os.getcwd())

SEED = 13


# Input: X and X_val numpy ndarray as returned by build_features,
#        Y and y_val arrays of labels,
#        nn_id indicating whether to take a retrained model or sherlock
# Output: Stored retrained model
def train_val_predict_model(X, Y, nn_id, label_categories):
    encoder = LabelEncoder()
    encoder.fit(Y)
    np.save('./models/classes_{}.npy'.format(nn_id), encoder.classes_)
    y_int = encoder.transform(Y)

    lr = 0.0001
    epochs = 100

    # Divide the dataset into train, validation and test
    train_data = WDCDataset(X, y_int, transform=ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    print("The length of the train dataset is {}".format(len(train_data)))

    validation_data = WDCDataset(X, y_int, transform=ToTensor())
    validation_loader = DataLoader(dataset=validation_data, batch_size=256, shuffle=False)
    print("The length of the validation dataset is {}".format(len(train_data)))

    test_data = WDCDataset(X, y_int, transform=ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=False)
    print("The length of the test dataset is {}".format(len(test_data)))

    m = NNModelConstruction(train_loader, validation_loader, test_loader,
                            Sherlock(SEED, label_categories=label_categories), nn_id, lr, epochs)
    m.train(save_cp=True)
    print('Trained new model.')

    # Predict labels using the model
    predicted_labels = m.predict()
    true_labels = encoder.inverse_transform(y_int)
    print('Predicted labels: ', predicted_labels, 'true labels: ', true_labels)
    print('The number of correct predictions are {}/{}'.format(np.count_nonzero(predicted_labels == true_labels),
                                                               len(test_data)))
    # F1-score of the best model prediction
    print('The final f1-score is {}%'.format(f1_score(true_labels, predicted_labels, average='weighted') * 100))
