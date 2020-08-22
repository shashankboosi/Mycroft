import os
import time

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from tqdm import tqdm


class NNModelConstruction:
    def __init__(self, train_loader, validation_loader, test_loader, network, nn_id, learning_rate=0.001, epochs=50):
        """
        Load Data, initialize a given network structure and set learning rate
        """

        self.test_loader = test_loader
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.nn_id = nn_id

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = network.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        if not os.path.exists(os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'output', 'checkpoints'))):
            os.mkdir(os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'output', 'checkpoints')))
        self.model_checkpoint_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'output', 'checkpoints'))

    def train(self, save_cp=False):

        total_step = len(self.train_loader)
        best_acc = 0
        prev_loss = 0
        count = 0
        best_acc_list = []
        for epoch in tqdm(range(self.epochs)):
            print()
            print('-----------------------------------------------------')
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            self.model.train()
            self.train_epoch(epoch, total_step)
            epoch_acc, epoch_loss = self.eval(epoch)

            # Early Stopping of 5
            if epoch == 0:
                prev_loss = epoch_loss
            else:
                if prev_loss < epoch_loss:
                    count += 1
                else:
                    count = 0
                prev_loss = epoch_loss

            if count == 5:
                exit('The model stopped learning and early stopping is applied')

            # Saving the best model based on the best accuracy
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_list.append(best_acc*100)
                if save_cp:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_checkpoint_path, 'CP-{}.pth'.format(self.nn_id)))
                print('Checkpoint at epoch {} saved !'.format(epoch + 1))
                print('Best val accuracy is {}%'.format(best_acc * 100))
            self.model.train()

        return best_acc_list

    def train_epoch(self, epoch, total_step):

        train_history_per_epoch = {'loss': 0, 'acc': 0}
        start = time.time()
        total = 0
        for i, batch in enumerate(self.train_loader):

            x, y, z, w, label = batch
            label = label.type(torch.LongTensor)

            outputs = self.model(x, y, z, w)
            loss = self.criterion(outputs, label)

            train_history_per_epoch['loss'] += loss.item() * label.size(0)
            total += label.size(0)
            train_history_per_epoch['acc'] += (torch.max(outputs.data, 1)[1] == label).sum().item()

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 2 == 0:
                print('Loss at Epoch [{}/{}] and Step [{}/{}] is: {:.4f}'
                      .format(epoch + 1, self.epochs, i + 1, total_step, train_history_per_epoch['loss'] / total))

        time_elapsed = time.time() - start
        print('Time taken for epoch {} is {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
        print('Loss and accuracy of the network on the epoch: {:.4f} & {:.4f}'.format(
            train_history_per_epoch['loss'] / total,
            100 * train_history_per_epoch['acc'] / total))

    def eval(self, epoch):

        # Validate the training data
        print('\nEval..')

        validation_history_per_epoch = {'loss': 0, 'acc': 0}
        total = 0
        self.model.eval()
        for j, val_batch in enumerate(self.validation_loader):
            x, y, z, w, label = val_batch
            label = label.type(torch.LongTensor)

            # Predict
            output_pred = self.model(x, y, z, w)

            # Calculate loss
            val_loss = self.criterion(output_pred, label)

            total += label.size(0)
            validation_history_per_epoch['loss'] += val_loss.item() * label.size(0)
            validation_history_per_epoch['acc'] += (
                    torch.max(output_pred.data, 1)[1] == label).sum().item()

        print('Validation loss and accuracy of the network for epoch [{}/{}] : {:.4f} & {:.4f}%'.format(
            epoch + 1, self.epochs, validation_history_per_epoch['loss'] / total,
            100 * validation_history_per_epoch['acc'] / total))

        return validation_history_per_epoch['acc'] / total, validation_history_per_epoch['loss'] / total

    def predict(self):

        # Load the pytorch file and define model
        print('\nPrediction..')
        net = self.model
        net.load_state_dict(torch.load(os.path.join(self.model_checkpoint_path, 'CP-{}.pth'.format(self.nn_id))))
        net.eval()

        test_history = {'loss': 0, 'acc': 0}

        total = 0
        prediction_labels = []
        true_labels = []
        with torch.no_grad():
            for k, test_batch in enumerate(self.test_loader):
                x, y, z, w, label = test_batch
                label = label.type(torch.LongTensor)

                # Predict test outputs
                output_pred = self.model(x, y, z, w)

                # Calculate test loss
                test_loss = self.criterion(output_pred, label)
                prediction_labels.append(torch.argmax(output_pred, 1))
                true_labels.append(label)

                total += label.size(0)
                test_history['loss'] += test_loss.item() * label.size(0)
                test_history['acc'] += (torch.max(output_pred.data, 1)[1] == label).sum().item()

        print('Prediction loss and accuracy of the network: {:.4f} & {:.4f}%'.format(test_history['loss'] / total,
                                                                                     100 * test_history['acc'] / total))

        encoder = LabelEncoder()
        encoder.classes_ = np.load('./models/classes_{}.npy'.format(self.nn_id), allow_pickle=True)
        y_pred_labels = encoder.inverse_transform(torch.cat(prediction_labels))
        y_true_labels = encoder.inverse_transform(torch.cat(true_labels))

        print('Predicted labels: ', y_pred_labels, 'true labels: ', y_true_labels)
        print('The number of correct predictions are {}/{}'.format(np.count_nonzero(y_pred_labels == y_true_labels),
                                                                   total))

        return y_pred_labels, y_true_labels
