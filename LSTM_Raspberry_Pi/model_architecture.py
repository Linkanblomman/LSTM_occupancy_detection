import torch
import torch.nn as nn

class RNN(nn.Module):
  def __init__(self, input_size, hidden_layer_size, num_classes): # num_classes == output_size
    super(RNN, self).__init__()
    self.hidden_layer_size = hidden_layer_size
    self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True) # input_seq has to be -> (batch_size, seq_length, input_size)
    self.fc = nn.Linear(hidden_layer_size, num_classes)

  def forward(self, input_seq):
    h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)
    c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)

    out, (h_out, _) = self.lstm(input_seq, (h0,c0)) # output of shape (seq_len, batch, num_directions * hidden_size)

    h_out = h_out.view(-1, self.hidden_layer_size)
    out = self.fc(h_out)

    return out