from config import SOSToken, EOSToken
import torch

def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, n_examples=10, device = 'None'):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()

    fr_vocab = dataloader.dataset.fr_vocab
    eng_vocab = dataloader.dataset.eng_vocab
    
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
            decoder_input = torch.tensor([[SOSToken]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOSToken:
                    break

            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            # Optionally, print some examples
            if i < n_examples:
                predicted_sentence = ' '.join([fr_vocab.index2word[index] for index in predicted_indices if index not in (SOSToken, EOSToken)])
                target_sentence = ' '.join([fr_vocab.index2word[index.item()] for index in target_tensor if index.item() not in (SOSToken, EOSToken)])
                input_sentence = ' '.join([eng_vocab.index2word[index.item()] for index in input_tensor if index.item() not in (SOSToken, EOSToken)])

                print(f'Input: {input_sentence}, Target: {target_sentence}, Predicted: {predicted_sentence}')

        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss:.4f}, Accuracy: {100*accuracy:.2f}%')