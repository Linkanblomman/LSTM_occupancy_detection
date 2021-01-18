import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import time

# PyTorch
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


from dataloader_from_training import OccDecDataset, ToTensor
from model_architecture import RNN 

print("torch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)

seed_value = 42
torch.manual_seed(seed_value)

# Parameters
num_classes = 2 # Temperature and Humidity
learning_rate = 0.001
look_back = 20 # Seq_len
input_size = 2 # Temperature and Humidity
hidden_size = 40
num_layers = 1

DATASET = os.path.join(os.path.dirname(__file__), './raspberry_pi_files/datatest.txt')
DATALOADER = os.path.join(os.path.dirname(__file__), './raspberry_pi_files/dataloader.pth')
LOADED_MODEL = os.path.join(os.path.dirname(__file__), './raspberry_pi_files/LSTM_model.pth')

print('Test dataset file path: ', DATASET)
print('Test dataloader file path: ', DATALOADER)
print('Model file path: ', LOADED_MODEL)

rawdata_file = DATASET

index_col = 'Time (s)'
test_occupancy_df = pd.read_csv(rawdata_file)
test_occupancy_df.index.names = [index_col]

# Load the dataloader from training
#dataloader = torch.jit.load(DATALOADER)
dataloader = torch.load(DATALOADER)

print("Dataloader size:", len(dataloader))

# Load the model from training
loaded_model = RNN(input_size, hidden_size, num_classes)
loaded_model.load_state_dict(torch.load(LOADED_MODEL, map_location='cpu'))
loaded_model.eval()

test_loss , label_values, pred_plt = [], [], []

test_running_loss = 0.0
test_running_correct = 0 

start = time.time()

with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):
    features, labels = data[0], data[1]
    labels = labels.type(torch.int64)
    label_values.append(labels)

    outputs = loaded_model(features)
    _, preds = torch.max(input=outputs.data, dim=1)
    pred_plt.append(preds.item())
    test_running_loss += preds.item()
    test_running_correct += (preds == labels).sum().item()
    n_samples += labels.size(0)
    n_correct += (preds == labels).sum().item()

  test_loss = test_running_loss/len(dataloader.dataset)
  test_accuracy = 100. * test_running_correct/len(dataloader.dataset)
  print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}')

end = time.time()
print(f"Time for model to evaluate: {(end-start)/60:.2f} minutes")

series = np.hstack((np.zeros(look_back + 1, dtype=int), pred_plt))

# Plot the model predictions vs the sensor values
test_occupancy_df['LSTM'] = pd.Series(series, index=test_occupancy_df.index)
test_occupancy_df['Occupancy'].plot(figsize=[20, 10], legend=True)
test_occupancy_df['LSTM'].apply(lambda x: x - 1.1).plot(legend=True)
test_occupancy_df['Temperature'].apply(lambda x: x -15).plot(legend=True)
test_occupancy_df['Humidity'].apply(lambda x: x / 5. ).plot(legend=True)
plt.show()