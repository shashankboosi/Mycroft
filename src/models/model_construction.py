import torch

from torch import nn, optim, from_numpy


class NNModelConstruction:
    def __init__(self, x_train, x_test, y_train, y_test, network, learning_rate):
        """
        Load Data, initialize a given network structure and set learning rate
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = network
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_train_samples = len(self.x_train[0][0])
        self.num_test_samples = len(self.x_test[0][0])

    def train_epoch(self):
        self.model.train()
        outputs = self.model(from_numpy(self.x_train[0]).float(),
                             from_numpy(self.x_train[1]).float(),
                             from_numpy(self.x_train[2]).float(),
                             from_numpy(self.x_train[3]).float()
                             )

        # print(summary(self.model.to(self.device), [(1001, 960), (1001, 201), (1001, 400), (1001, 27)]))
        loss = self.criterion(outputs, torch.max(from_numpy(self.y_train), 1)[1])
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

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
