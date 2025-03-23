import time
import torch

from config import SOSToken, EOSToken
from tqdm import tqdm  

def encode(inputTensor, encoder, encoderHidden, device):
    for ei in range(inputTensor.size(0)):
        encoderOutput, encoderHidden = encoder(inputTensor[ei].unsqueeze(0), encoderHidden)
    return encoderHidden


def decode(encoderHidden, decoder, targetLength, device):
    decoderInput = torch.tensor([[SOSToken]], device=device)
    decoderHidden = encoderHidden
    decoderOutputs = []
    for di in range(targetLength):
        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
        topv, topi = decoderOutput.topk(1)
        decoderInput = topi.squeeze().detach()
        decoderOutputs.append(decoderOutput)
        if decoderInput.item() == EOSToken:  # Stop if EOS token is generated
            break
    return decoderOutputs


def calculate_loss(decoderOutputs, targetTensor, criterion):
    loss = 0
    for di, decoderOutput in enumerate(decoderOutputs):
        loss += criterion(decoderOutput, targetTensor[di].unsqueeze(0))
    return loss


def calculate_accuracy(decoderOutputs, targetTensor):
    correctPredictions = 0
    totalWords = 0
    for di, decoderOutput in enumerate(decoderOutputs):
        topv, topi = decoderOutput.topk(1)
        if topi.item() == targetTensor[di].item():
            correctPredictions += 1
        totalWords += 1
    accuracy = (correctPredictions / totalWords) * 100
    return accuracy


def train_fn(inputTensor, targetTensor, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion, device):
    encoderHidden = encoder.initHidden(device)

    # Clear gradients for optimizers
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    encoderHidden = encode(inputTensor, encoder, encoderHidden, device)

    decoderOutputs = decode(encoderHidden, decoder, targetTensor.size(0), device)

    loss = calculate_loss(decoderOutputs, targetTensor, criterion)

    loss.backward()

    encoderOptimizer.step()
    decoderOptimizer.step()

    return loss.item() / targetTensor.size(0)


def evaluate(encoder, decoder, criterion, valDataloader, device):
    encoder.eval()
    decoder.eval()

    totalLoss = 0
    correctPredictions = 0
    totalWords = 0

    with torch.no_grad():
        for inputTensor, targetTensor in valDataloader:
            inputTensor = inputTensor[0].to(device)
            targetTensor = targetTensor[0].to(device)

            encoderHidden = encoder.initHidden()
            encoderHidden = encode(inputTensor, encoder, encoderHidden, device)

            decoderOutputs = decode(encoderHidden, decoder, targetTensor.size(0), device)

            # Calculate loss and accuracy
            totalLoss += calculate_loss(decoderOutputs, targetTensor, criterion)
            accuracy = calculate_accuracy(decoderOutputs, targetTensor)

            correctPredictions += accuracy * targetTensor.size(0) / 100
            totalWords += targetTensor.size(0)

    avgLoss = totalLoss / len(valDataloader)
    accuracy = correctPredictions / totalWords * 100

    return avgLoss, accuracy


def log_progress(epoch, totalEpochs, epochLoss, valLoss, valAccuracy, epochTime):
    print(f'Epoch {epoch + 1}/{totalEpochs}, Loss: {epochLoss:.4f}, Validation Loss: {valLoss:.4f}, Validation Accuracy: {valAccuracy:.2f}%, Time: {epochTime:.2f}s')


def update_progress_bar(progressBar, epoch, totalEpochs, epochLoss, valLoss, valAccuracy, epochTime):
    progressBar.update(1)
    progressBar.set_description(f"Epoch {epoch + 1}/{totalEpochs}")
    progressBar.set_postfix(
        loss=f"{epochLoss:.4f}",
        valLoss=f"{valLoss:.4f}",
        valAccuracy=f"{valAccuracy:.2f}%",
        time=f"{epochTime:.2f}s"
    )


def training(encoder, decoder, encoderOptimizer, decoderOptimizer, criterion, dataloader, valDataloader, epochs, device):
    progressBar = tqdm(total=epochs, desc="Training Progress", position=0)

    for epoch in range(epochs):
        startTime = time.time()
        totalLoss = 0

        for i, (inputTensor, targetTensor) in enumerate(dataloader):
            inputTensor = inputTensor[0].to(device)
            targetTensor = targetTensor[0].to(device)

            loss = train_fn(inputTensor, targetTensor, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion, device)
            totalLoss += loss

        epochLoss = totalLoss / len(dataloader)

        # Validation phase
        valLoss, valAccuracy = evaluate(encoder, decoder, criterion, valDataloader, device)

        epochTime = time.time() - startTime

        update_progress_bar(progressBar, epoch, epochs, epochLoss, valLoss, valAccuracy, epochTime)

        # Print detailed loss stats every 10 epochs
        if (epoch + 1) % 10 == 0:
            log_progress(epoch, epochs, epochLoss, valLoss, valAccuracy, epochTime)

    progressBar.close()
    print("\nTraining complete!")

