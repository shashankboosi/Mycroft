import torch
from torch.utils.data import Dataset


class WDCDataset(Dataset):
    """
    PyTorch Dataset for the Semantic Type Detection model. To be used by a PyTorch DataLoader to feed batches to the model.
    """

    def __init__(self, features, labels, transform=None):
        """
        :param features: data that is used to be trained or tested
        :param labels: labels that are predicted
        """

        self.features = features
        self.labels = labels

        self.transform = transform

    def __getitem__(self, index):
        batch_sample = self.features[0][index], self.features[1][index], self.features[2][index], \
                       self.features[3][index], self.labels[index]  # 960, 201, 400, 27

        if self.transform:
            batch_sample = self.transform(batch_sample)

        return batch_sample

    def __len__(self):
        return len(self.features[0])


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, batch_sample):
        x, y, z, w, labels = batch_sample
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float(), \
               torch.from_numpy(w).float(), torch.from_numpy(labels)
