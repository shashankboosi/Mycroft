import pickle

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

from .dataset import WDCDataset, ToTensor
from .model_construction import NNModelConstruction
from .sherlock_model import Sherlock

SEED = 13


def train_val_predict_model(X, Y, nn_id, train_data_split, data_split, is_sample, no_of_tables, label_categories):
    """This function performs operations on the NN model such as train, validate and test. It also calculates the
    metrics like categorical accuracy and F1-score

    :param X: numpy ndarray as returned by build_features
    :param Y: array of labels
    :param nn_id: class selection between sherlock or mycroft data
    :param train_data_split: percentage of train data to be used
    :param data_split: Condition to split the data or not
    :param label_categories: It it the count of the number of labels available in the dataset (Max: 78)
    :return: Final predictions and calculates the F1-score
    """
    encoder = LabelEncoder()
    encoder.fit(Y)
    np.save('./models/classes_{}.npy'.format(nn_id), encoder.classes_)
    y_int = encoder.transform(Y)

    lr = 0.0001
    epochs = 20

    # Divide the dataset into train, validation and test
    dataset = WDCDataset(X, y_int, transform=ToTensor())

    if data_split:

        # Train, validation and test split
        train_size = int(train_data_split * len(dataset))
        remaining_size = len(dataset) - train_size
        validation_size = int(0.5 * remaining_size)
        test_size = remaining_size - validation_size

        train_dataset, validation_dataset, test_dataset = random_split(dataset,
                                                                       [train_size, validation_size, test_size])
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
    best_acc_list = m.train(save_cp=True)

    if is_sample:
        acc_list_path = "../resources/output/accuracy_list/acc_list_sample.p"
    else:
        acc_list_path = "../resources/output/accuracy_list/acc_list_{}.p".format(no_of_tables)

    with open(acc_list_path, 'wb') as f:
        pickle.dump(best_acc_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Trained new model.')

    # Predict labels using the model
    predicted_labels, true_labels = m.predict()

    # F1-score of the best model prediction
    print('The final f1-score is {}%'.format(f1_score(true_labels, predicted_labels, average='weighted') * 100))

    # Precision Score of the best model prediction
    print('The final precision score is {}%'.format(
        precision_score(true_labels, predicted_labels, average='weighted') * 100))

    # Recall Score of the best model prediction
    print('The final recall score is {}%'.format(recall_score(true_labels, predicted_labels, average='weighted') * 100))
