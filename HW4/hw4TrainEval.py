import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(encoder, decoder, dataloader, criterion, n_epochs, encoder_optimizer, decoder_optimizer, device):
    for epoch in range(n_epochs):
        total_loss = 0
        
        for input_tensor, target_tensor in dataloader:
            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            # Initialize encoder hidden state
            encoder_hidden = encoder.initHidden()

            # Clear gradients for optimizers
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Calculate the length of input and target tensors
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            loss = 0

            # Encoding
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

            # Decoder initialization
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            # Decoding loop
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            # Backpropagation
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item() / target_length

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')
    
def evaluate_and_show_examples(index_to_word, encoder, decoder, dataloader, criterion, n_examples=10):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_predictions = 0

    # No gradient calculation
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            loss = 0

            # Encoding step
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

            # Decoding step
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            # Optionally, print some examples
            if i < n_examples:
                predicted_sentence = ' '.join([index_to_word[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_sentence = ' '.join([index_to_word[index.item()] for index in target_tensor if index.item() not in (SOS_token, EOS_token)])
                input_sentence = ' '.join([index_to_word[index.item()] for index in input_tensor if index.item() not in (SOS_token, EOS_token)])

                print(f'Input: {input_sentence}, Target: {target_sentence}, Predicted: {predicted_sentence}')

        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')
