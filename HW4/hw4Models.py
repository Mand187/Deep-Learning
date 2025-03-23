import torch
import torch.nn as nn

class Encoder(nn.Module):
    """The Encoder part of the seq2seq model."""
    def __init__(self, inputSize, hiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize)

    def forward(self, inputTensor, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(inputTensor).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        # Initializes hidden state
        return torch.zeros(1, 1, self.hiddenSize, device=device)
    
class Decoder(nn.Module):
    """The Decoder part of the seq2seq model."""
    def __init__(self, hiddenSize, outputSize):
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.softmax = nn.LogSoftmax(dim=1)
                             
    def forward(self, inputTensor, hidden):
        embedded = self.embedding(inputTensor).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hiddenSize, device=device)
