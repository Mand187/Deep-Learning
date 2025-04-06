import torch
from torch.utils.data import DataLoader

def generate_translation(model, fr_sentence, max_length, eng_vocab, fr_vocab, device):
    # Tokenize and pad the English sentence
    fr_indices = [fr_vocab.word2index.get(word, fr_vocab.word2index['<PAD>']) for word in fr_sentence.split()]
    fr_indices += [fr_vocab.word2index['<EOS>']]  # Add <EOS> token
    fr_indices = torch.tensor(fr_indices, dtype=torch.long).unsqueeze(0).to(device)  # Move to model's device

    # Initialize English translation with <SOS> token
    eng_indices = [eng_vocab.word2index['<SOS>']]
    
    # Disable gradient computation
    with torch.no_grad():
        model.eval()
        # Loop to generate English translation
        for _ in range(max_length):
            # Convert English translation indices to tensor
            eng_tensor = torch.tensor(eng_indices, dtype=torch.long).unsqueeze(0).to(device)  # Move to model's device
            
            # Predict the next token in the English translation
            output = model(fr_indices, eng_tensor)
            next_token_index = torch.argmax(output[:, -1, :], dim=-1).item()
            
            # If the predicted token is <EOS>, stop generating
            if next_token_index == eng_vocab.word2index['<EOS>']:
                break
            
            # Add the predicted token to the French translation after <SOS>
            eng_indices.append(next_token_index)
    
    # Convert French translation indices to words
    eng_sentence = ' '.join([eng_vocab.index2word[index] for index in eng_indices[1:]])  # Exclude <SOS>
    return eng_sentence
