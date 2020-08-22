import torch
import torch.nn as nn
import torch.nn.functional as F

from .sherlock_model import SherlockThinSlice


class MycroftBiLSTM(nn.Module):
    def __init__(self, seed, label_categories=78):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.label_categories = label_categories
        self.input_data_slice1 = SherlockThinSlice(960, 300)
        self.input_data_slice2 = SherlockThinSlice(201, 200)
        self.input_data_slice3 = SherlockThinSlice(400, 400)
        self.bn1 = nn.BatchNorm1d(27, momentum=0.99, eps=0.001)
        self.bn2 = nn.BatchNorm1d(927, momentum=0.99, eps=0.001)
        self.bn3 = nn.BatchNorm1d(400, momentum=0.99, eps=0.001)
        self.dense1 = nn.Linear(927, 600)

        self.lstm = nn.LSTM(input_size=600, hidden_size=400, num_layers=1, batch_first=True, bidirectional=True)

        self.dense2 = nn.Linear(400 * 2, 400)
        self.dense3 = nn.Linear(400, self.label_categories)

    def forward(self, x, y, z, w):
        """
            4 inputs - input_data[0], input_data[1], input_data[2], input_data[3] (4 inputs)
             -> input[0] -> batch_norm -> dense -> dropout -> dense
             -> input[1] -> batch_norm -> dense -> dropout -> dense
             -> input[2] -> batch_norm -> dense -> dropout -> dense
             -> input[3] -> batch_norm
             -> concatenate -> batch_norm -> dense -> dropout -> dense -> dense
             -> CategoricalCrossEntropyLoss
             -> loss
        """
        input_data_1 = self.input_data_slice1(x)  # 960
        input_data_2 = self.input_data_slice2(y)  # 201
        input_data_3 = self.input_data_slice3(z)  # 400
        input_data_4 = self.bn1(w)  # 27
        x = torch.cat((input_data_1, input_data_2, input_data_3, input_data_4), 1)

        x = self.bn2(x)
        x = F.dropout(F.relu(self.dense1(x)), 0.35)
        x = x.unsqueeze(1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(1 * 2, x.size(0), 400).to(device)
        c0 = torch.zeros(1 * 2, x.size(0), 400).to(device)

        output, _ = self.lstm(x, (h0, c0))

        output = F.relu(self.dense2(output[:, -1, :]))
        output = self.dense3(self.bn3(output))

        return output
