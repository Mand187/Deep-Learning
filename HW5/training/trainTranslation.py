import time
import torch
import torch.nn as nn
import torch.optim as optim

def trainModel(model, trainingLoader, validationLoader, epochs=10, learningRate=0.001, updateInterval=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Training on device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    trainLosses = []
    valLosses = []
    trainAccuracies = []
    valAccuracies = []
    epochTimes = []
    trueLabelsList = []
    predictedLabelsList = []

    totalTrainingStart = time.time()

    for epoch in range(epochs):
        model.train()
        runningLoss = 0.0
        correctTrain = 0
        totalTrain = 0
        epochStartTime = time.time()

        for src, tgt in trainingLoader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(src, tgt[:, :-1])  # Pass target sequence (excluding <EOS> token)

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))  # Exclude <SOS> token from targets

            # Backward pass
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            # Calculate training accuracy
            predicted = torch.argmax(outputs, dim=-1)
            correctTrain += (predicted == tgt[:, 1:]).sum().item()
            totalTrain += tgt[:, 1:].nelement()

        trainLoss = runningLoss / len(trainingLoader)
        trainAccuracy = (correctTrain / totalTrain) * 100
        trainLosses.append(trainLoss)
        trainAccuracies.append(trainAccuracy)

        epochEndTime = time.time()
        epochTime = epochEndTime - epochStartTime
        epochTimes.append(epochTime)

        # Validation phase
        model.eval()
        runningValLoss = 0.0
        correctVal = 0
        totalVal = 0
        trueLabels = []
        predictedLabels = []

        with torch.no_grad():
            for src, tgt in validationLoader:
                src, tgt = src.to(device), tgt.to(device)

                outputs = model(src, tgt[:, :-1])  # Pass target sequence (excluding <EOS> token)

                # Calculate validation loss
                valLoss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))  # Exclude <SOS> token from targets
                runningValLoss += valLoss.item()

                # Calculate validation accuracy
                predicted = torch.argmax(outputs, dim=-1)
                correctVal += (predicted == tgt[:, 1:]).sum().item()
                totalVal += tgt[:, 1:].nelement()

                # Collect labels for later evaluation
                trueLabels.append(tgt[:, 1:].cpu().numpy())
                predictedLabels.append(predicted.cpu().numpy())

            valLoss = runningValLoss / len(validationLoader)
            valAccuracy = (correctVal / totalVal) * 100
            valLosses.append(valLoss)
            valAccuracies.append(valAccuracy)

        # Print progress at updateInterval or final epoch
        if (epoch + 1) % updateInterval == 0 or (epoch + 1) == epochs:
            inferenceTime = sum(epochTimes) - epochTime  # Time spent on validation/inference
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {trainLoss:.4f} | Train Acc: {trainAccuracy:.2f}% | "
                  f"Val Loss: {valLoss:.4f} | Val Acc: {valAccuracy:.2f}%")
            print(f"  Epoch time: {epochTime:.2f} sec | Inference time: {inferenceTime:.2f} sec")

    totalTrainingTime = time.time() - totalTrainingStart
    avgEpochTime = sum(epochTimes) / len(epochTimes)

    print("\nTraining complete.")
    print(f"Total training time: {totalTrainingTime:.2f} seconds")
    print(f"Average time per epoch: {avgEpochTime:.2f} seconds")

    return trainLosses, valLosses, trainAccuracies, valAccuracies, epochTimes, trueLabelsList, predictedLabelsList
