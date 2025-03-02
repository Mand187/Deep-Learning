import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm  # Import TQDM

def trainModel(model, trainLoader, testLoader, numEpochs=50, learningRate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Training on {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    trainLosses, valLosses = [], []
    trainAccuracies, valAccuracies = [], []
    epochTimes = []

    model = model.to(device)

    # Determine update interval for tqdm
    if numEpochs > 500:
        update_interval = 100
    elif numEpochs > 200:
        update_interval = 50
    elif numEpochs > 50:
        update_interval = 10
    else:
        update_interval = 1

    for epoch in range(1, numEpochs + 1):
        startTime = time.time()
        model.train()
        runningLoss = 0.0
        correctTrain, totalTrain = 0, 0

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM returns (hidden, cell_state)
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            totalTrain += labels.size(0)
            correctTrain += predicted.eq(labels).sum().item()

        epochLoss = runningLoss / len(trainLoader)
        trainLosses.append(epochLoss)
        trainAccuracy = 100. * correctTrain / totalTrain
        trainAccuracies.append(trainAccuracy)

        # Validation
        model.eval()
        valLoss = 0.0
        correct, total = 0, 0
        trueLabelsList, predictedLabelsList = [], []

        with torch.no_grad():
            for inputs, labels in testLoader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Initialize hidden state for evaluation
                hidden = model.init_hidden(inputs.size(0))
                if isinstance(hidden, tuple):
                    hidden = (hidden[0].to(device), hidden[1].to(device))
                else:
                    hidden = hidden.to(device)

                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, labels)
                valLoss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Collect labels for evaluation
                trueLabelsList.extend(labels.cpu().numpy())
                predictedLabelsList.extend(predicted.cpu().numpy())

        valLoss /= len(testLoader)
        valLosses.append(valLoss)

        accuracy = 100. * correct / total
        valAccuracies.append(accuracy)

        epochTime = time.time() - startTime
        epochTimes.append(epochTime)

        if epoch % update_interval == 0 or epoch == numEpochs:
            print(f'Epoch {epoch}/{numEpochs}, Train Loss: {epochLoss:.4f}, Train Acc: {trainAccuracy:.2f}%, Val Loss: {valLoss:.4f}, Val Acc: {accuracy:.2f}%')

    return trainLosses, valLosses, trainAccuracies, valAccuracies, epochTimes, trueLabelsList, predictedLabelsList

def trainModelTQDM(model, trainLoader, testLoader, numEpochs=50, learningRate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Training on {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    trainLosses, valLosses = [], []
    trainAccuracies, valAccuracies = [], []
    epochTimes = []

    model = model.to(device)

    # Use tqdm for overall progress
    progress_bar = tqdm(total=numEpochs, desc="Training Progress", position=0)
    
    for epoch in range(1, numEpochs + 1):
        startTime = time.time()
        
        # Training phase
        model.train()
        runningLoss = 0.0
        correctTrain, totalTrain = 0, 0
        
        # Process training data
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM returns (hidden, cell_state)
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            totalTrain += labels.size(0)
            correctTrain += predicted.eq(labels).sum().item()

        epochLoss = runningLoss / len(trainLoader)
        trainLosses.append(epochLoss)
        trainAccuracy = 100. * correctTrain / totalTrain
        trainAccuracies.append(trainAccuracy)

        # Validation phase
        model.eval()
        valLoss = 0.0
        correct, total = 0, 0
        trueLabelsList, predictedLabelsList = [], []

        with torch.no_grad():
            for inputs, labels in testLoader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Initialize hidden state for evaluation
                hidden = model.init_hidden(inputs.size(0))
                if isinstance(hidden, tuple):
                    hidden = (hidden[0].to(device), hidden[1].to(device))
                else:
                    hidden = hidden.to(device)

                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, labels)
                valLoss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Collect labels for evaluation
                trueLabelsList.extend(labels.cpu().numpy())
                predictedLabelsList.extend(predicted.cpu().numpy())

        valLoss /= len(testLoader)
        valLosses.append(valLoss)

        accuracy = 100. * correct / total
        valAccuracies.append(accuracy)

        epochTime = time.time() - startTime
        epochTimes.append(epochTime)
        
        # Update overall progress bar with detailed epoch metrics
        progress_bar.update(1)
        progress_bar.set_description(f"Epoch {epoch}/{numEpochs}")
        progress_bar.set_postfix(
            train_loss=f"{epochLoss:.4f}",
            train_acc=f"{trainAccuracy:.2f}%", 
            val_loss=f"{valLoss:.4f}", 
            val_acc=f"{accuracy:.2f}%",
            time=f"{epochTime:.2f}s"
        )

    progress_bar.close()
    print("\nTraining complete!")

    return trainLosses, valLosses, trainAccuracies, valAccuracies, epochTimes, trueLabelsList, predictedLabelsList