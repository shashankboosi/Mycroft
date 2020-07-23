import time

import numpy as np
import torch
from torch import nn, optim, from_numpy
from tqdm import tqdm

from src.models.metrics import categorical_accuracy, f1_score


class NNModelConstruction:
    def __init__(self, train_loader, validation_loader, network, learning_rate=0.001, epochs=50):
        """
        Load Data, initialize a given network structure and set learning rate
        """

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = network
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):

        total_step = len(self.train_loader)
        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            self.train_epoch(epoch, total_step)

        return

    def train_epoch(self, epoch, total_step):

        train_history_per_epoch = {'loss': 0, 'acc': 0}
        self.model.train()
        start = time.time()
        total = 0
        for i, batch in enumerate(self.train_loader):

            x, y, z, w, label = batch
            del batch

            outputs = self.model(x, y, z, w)
            loss = self.criterion(outputs, torch.max(label, 1)[1])

            train_history_per_epoch['loss'] += loss.item()
            total += label.size(0)
            train_history_per_epoch['acc'] += (torch.max(outputs, 1)[1] == torch.max(label, 1)[1]).sum().item()

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                print('Loss at Epoch [{}/{}] and Step [{}/{}] is: {:.4f}'
                      .format(epoch + 1, self.epochs, i + 1, total_step, train_history_per_epoch['loss'] / 10))
                train_history_per_epoch['loss'] = 0

        print('Time taken for epoch {} is {}'.format(epoch, time.time() - start))
        print('Accuracy of the network on the epoch: %d %%' % (
                100 * train_history_per_epoch['acc'] / total))

    def eval(self):
        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            log_ps = self.model(from_numpy(self.x_test[0]).float(),
                                from_numpy(self.x_test[1]).float(),
                                from_numpy(self.x_test[2]).float(),
                                from_numpy(self.x_test[3]).float()
                                )
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == torch.from_numpy(self.y_test)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        return accuracy / self.num_test_samples
