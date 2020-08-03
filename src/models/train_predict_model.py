import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

from .dataset import WDCDataset, ToTensor
from .model_construction import NNModelConstruction
from .sherlock_model import Sherlock

sys.path.append(os.getcwd())

SEED = 13


# Input: X and X_val numpy ndarray as returned by build_features,
#        Y and y_val arrays of labels,
#        nn_id indicating whether to take a retrained model or sherlock
# Output: Stored retrained model
def train_val_predict_model(X, Y, nn_id, train_data_split, data_split, label_categories):
    encoder = LabelEncoder()
    encoder.fit(Y)
    np.save('./models/classes_{}.npy'.format(nn_id), encoder.classes_)
    y_int = encoder.transform(Y)

    lr = 0.0001
    epochs = 100

    # Divide the dataset into train, validation and test
    dataset = WDCDataset(X, y_int, transform=ToTensor())

    if data_split:

        # Train, validation and test split
        train_size = int(train_data_split * len(dataset))
        remaining_size = len(dataset) - train_size
        validation_size = int(0.5 * remaining_size)
        test_size = remaining_size - validation_size

        train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
        print('Train dataset', len(train_dataset))
        print('Validation dataset', len(validation_dataset))
        print('Test dataset', len(test_dataset))

        train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    else:
        train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
        validation_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)
        print('Train, validation and test dataset size', len(dataset))

    m = NNModelConstruction(train_loader, validation_loader, test_loader,
                            Sherlock(SEED, label_categories=label_categories), nn_id, lr, epochs)
    m.train(save_cp=True)
    print('Trained new model.')

    # Predict labels using the model
    predicted_labels, true_labels = m.predict()

    # F1-score of the best model prediction
    print('The final f1-score is {}%'.format(f1_score(true_labels, predicted_labels, average='weighted') * 100))
