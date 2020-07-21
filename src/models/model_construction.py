import time

import numpy as np
import torch
from torch import nn, optim, from_numpy
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

        self.model = network
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):

        total_step = len(self.train_loader)
        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))

            train_history_per_epoch = {'loss': [], 'acc': [], 'f1_score': []}
            self.model.train()

            for i, batch in enumerate(self.train_loader):
                start = time.time()
                x, y, z, w, label = batch
                # images = input.type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2)
                # labels = label.type(torch.LongTensor).unsqueeze(1).to(self.device)
                del batch

                outputs = self.model(x, y, z, w)
                loss = self.criterion(outputs, torch.max(label, 1)[1])
                # overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(labels.squeeze(1),
                #                                                                   outputs.argmax(dim=1),
                #                                                                   self.num_of_classes)
                # train_history_per_epoch['acc'].append(overall_acc)
                # train_history_per_epoch['f1_score'].append(avg_per_class_acc)
                train_history_per_epoch['loss'].append(loss.item())

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                end = time.time()
                print('Time taken for the batch is {}'.format(end - start))

                if (i + 1) % 10 == 0:
                    print('Loss at Epoch [{}/{}] and Step [{}/{}] is: {:.4f} and {:.4f}'
                          .format(epoch + 1, self.epochs, i + 1, total_step,
                                  np.mean(train_history_per_epoch['loss']), loss.item()))

        # print(summary(self.model.to(self.device), [(1001, 960), (1001, 201), (1001, 400), (1001, 27)]))
        # loss = self.criterion(outputs, torch.max(from_numpy(self.y_train), 1)[1])
        return

    def train_epoch(self):
        pass

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
