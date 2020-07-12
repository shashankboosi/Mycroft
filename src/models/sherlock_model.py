import torch
import torch.nn as nn
import torch.nn.functional as F


class SherlockThinSlice(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.dense1 = nn.Linear(in_channel, out_channel)
        self.dense2 = nn.Linear(out_channel, out_channel)

    def forward(self, x):
        x = self.bn1(x)
        x = F.dropout(self.dense1(x), 0.35)
        x = self.dense2(x)
        return x


class Sherlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_data_slice1 = SherlockThinSlice(960, 300)
        self.input_data_slice2 = SherlockThinSlice(201, 200)
        self.input_data_slice3 = SherlockThinSlice(400, 400)
        self.bn1 = nn.BatchNorm1d(27)
        self.bn2 = nn.BatchNorm1d(927)
        self.dense1 = nn.Linear(927, 500)
        self.dense2 = nn.Linear(500, 500)
        self.dense3 = nn.Linear(500, 23)

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
        x = F.dropout(self.dense1(x), 0.35)
        x = self.dense2(x)
        x = self.dense3(x)

        return F.softmax(x, dim=1)
