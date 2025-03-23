import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TranslationDataset(Dataset):
    def __init__(self, dataset, word_to_index):
        self.dataset = dataset
        self.word_to_index = word_to_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_sentence, target_sentence = self.dataset[idx]
        input_indices = [self.word_to_index[word] for word in input_sentence.split()] + [EOS_token]
        target_indices = [self.word_to_index[word] for word in target_sentence.split()] + [EOS_token]
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU layer

    def forward(self, input, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        # Initializes hidden state
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU layer
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=14, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)  # Embedding layer
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # Attention layer
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)  # Combining layer
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)  # GRU layer
        self.out = nn.Linear(self.hidden_size, output_size)  # Output layer

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculating attention weights
        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # Combining embedded input with attention output
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
def create_vocab_mappings(dataset, SOS_token, EOS_token):
    word_to_index = {"SOS": SOS_token, "EOS": EOS_token}
    
    for pair in dataset:
        for word in pair[0].split() + pair[1].split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
    
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    return word_to_index, index_to_word

def translate(encoder, decoder, sentence, input_lang, output_lang, device, max_length=10):
    with torch.no_grad():
        input_tensor = torch.tensor([input_lang.word2index[word] for word in sentence.split(' ')], device=device).view(-1, 1)
        encoder_hidden = encoder.initHidden()
        input_length = input_tensor.size(0)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        
        return ' '.join(decoded_words)