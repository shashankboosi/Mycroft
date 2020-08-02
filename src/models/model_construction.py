import os
import time

import torch
from torch import nn, optim
from tqdm import tqdm


class NNModelConstruction:
    def __init__(self, train_loader, validation_loader, network, learning_rate=0.001, epochs=50):
        """
        Load Data, initialize a given network structure and set learning rate
        """

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = network.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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
        for epoch in tqdm(range(self.epochs)):
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
                if save_cp:
                    torch.save(self.model.state_dict(), os.path.join(self.model_checkpoint_path, 'CP.pth'))
                print('Checkpoint at epoch {} saved !'.format(epoch + 1))
            self.model.train()

        return

    def train_epoch(self, epoch, total_step):

        train_history_per_epoch = {'loss': 0, 'acc': 0}
        start = time.time()
        total = 0
        for i, batch in enumerate(self.train_loader):

            x, y, z, w, label = batch

            outputs = self.model(x, y, z, w)
            loss = self.criterion(outputs, torch.argmax(label, 1))

            train_history_per_epoch['loss'] += loss.item() * label.size(0)
            total += label.size(0)
            train_history_per_epoch['acc'] += (torch.max(outputs.data, 1)[1] == torch.max(label, 1)[1]).sum().item()

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                print('Loss at Epoch [{}/{}] and Step [{}/{}] is: {:.4f}'
                      .format(epoch + 1, self.epochs, i + 1, total_step, train_history_per_epoch['loss'] / total))

        print('Time taken for epoch {} is {}'.format(epoch + 1, time.time() - start))
        print('Loss and accuracy of the network on the epoch: {:.4f} & {:.4f}'.format(
            train_history_per_epoch['loss'] / total,
            100 * train_history_per_epoch['acc'] / total))

    def eval(self, epoch):

        print('\nEval..')
        validation_history_per_epoch = {'loss': 0, 'acc': 0}
        # eval
        self.model.eval()
        total = 0
        for j, val_batch in enumerate(self.validation_loader):
            x, y, z, w, label = val_batch

            # Predict
            output_pred = self.model(x, y, z, w)

            # Calculate loss
            val_loss = self.criterion(output_pred, torch.argmax(label, 1))

            total += label.size(0)
            validation_history_per_epoch['loss'] += val_loss.item() * label.size(0)
            validation_history_per_epoch['acc'] += (
                    torch.max(output_pred.data, 1)[1] == torch.max(label, 1)[1]).sum().item()

        print('Validation loss and accuracy of the network for epoch [{}/{}] : {:.4f} & {:.4f}%'.format(
            epoch + 1, self.epochs, validation_history_per_epoch['loss'] / total,
            100 * validation_history_per_epoch['acc'] / total))

        return validation_history_per_epoch['acc'] / total, validation_history_per_epoch['loss'] / total
