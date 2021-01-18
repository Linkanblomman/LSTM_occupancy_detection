import torch
from torch.utils.data import Dataset, DataLoader

class ToTensor:
  def __call__(self, batch):
    inputs, targets = batch
    return torch.tensor(inputs, dtype=torch.float), torch.tensor(targets, dtype=torch.int)

class OccDecDataset(Dataset):

  def __init__(self, x_features, y_features, transform=None):
    # Data loading
    self.x = x_features
    self.y = y_features
    self.n_samples = x_features.shape[0]
    self.transform = transform

  def __getitem__(self, index):
    batch = self.x[index], self.y[index] # [x , y]

    if self.transform:
      batch = self.transform(batch)
    else:
      print("Use transform ToTensor and NormalizeToStandardScaler")

    return batch

  def __len__(self):
    # len(dataset)
    return self.n_samples