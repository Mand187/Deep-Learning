import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output

class TransformerII(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(TransformerII, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.output_embedding = nn.Embedding(output_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.output_embedding(tgt)
        memory = self.transformer_encoder(src)
        transformer_output = self.transformer_decoder(tgt, memory)
        output = self.fc(transformer_output)
        return output