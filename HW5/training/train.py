import time
import torch
import torch.nn as nn
import torch.optim as optim

def trainModel(model, trainLoader, testLoader, numEpochs=50, learningRate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Training on device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    trainLosses, valLosses = [], []
    trainAccuracies, valAccuracies = [], []
    epochTimes = []

    totalTrainingStart = time.time()
    model = model.to(device)

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

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            totalTrain += labels.size(0)
            correctTrain += predicted.eq(labels).sum().item()

        epochLoss = runningLoss / len(trainLoader)
        trainAccuracy = 100. * correctTrain / totalTrain
        trainLosses.append(epochLoss)
        trainAccuracies.append(trainAccuracy)

        # Validation
        model.eval()
        valLoss = 0.0
        correct, total = 0, 0
        trueLabelsList, predictedLabelsList = [], []

        inferenceStart = time.time()
        with torch.no_grad():
            for inputs, labels in testLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valLoss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                trueLabelsList.extend(labels.cpu().numpy())
                predictedLabelsList.extend(predicted.cpu().numpy())
        inferenceEnd = time.time()
        inferenceTime = inferenceEnd - inferenceStart

        valLoss /= len(testLoader)
        accuracy = 100. * correct / total
        valLosses.append(valLoss)
        valAccuracies.append(accuracy)

        epochTime = time.time() - startTime
        epochTimes.append(epochTime)

        if epoch % update_interval == 0 or epoch == numEpochs:
            print(f"Epoch {epoch:3d}/{numEpochs} | Train Loss: {epochLoss:.4f} | Train Acc: {trainAccuracy:.2f}% | "
                  f"Val Loss: {valLoss:.4f} | Val Acc: {accuracy:.2f}%")
            print(f"  Epoch time: {epochTime:.2f} sec | Inference time: {inferenceTime:.2f} sec")

    totalTrainingTime = time.time() - totalTrainingStart
    avgEpochTime = sum(epochTimes) / len(epochTimes)

    print("\nTraining complete.")
    print(f"Total training time: {totalTrainingTime:.2f} seconds")
    print(f"Average time per epoch: {avgEpochTime:.2f} seconds")

    return trainLosses, valLosses, trainAccuracies, valAccuracies, epochTimes, trueLabelsList, predictedLabelsList

def trainMultiSequence(name, model, sequences=[], epochs=100, learningRate=0.001, trainingLoader=[], validationLoader=[]):

    names = []
    models = []
    trainingLosesMulti, validationAccuraciesMulti = [], []
    trainAccuraciesMulti, epochTimesMulti = [], []
    validationLossesMulti = []
    trueLabelsListsMulti, predictedLabelsListsMulti = [], []

    for i, sequence in enumerate(sequences):
        print(f'Training model for sequence length {sequence}')
        trainLosses, valLosses, trainAccuracies, valAccuracies, epochTimes, trueLabelsList, predictedLabelsList = trainModel(model,
                                                                                                                     trainingLoader[i],
                                                                                                                     validationLoader[i],
                                                                                                                     numEpochs=epochs,
                                                                                                                     learningRate=learningRate)
        print(f'Model for sequence length {sequence} trained.')

        names.append(f'{name} - {sequence}')
        models.append(model)
        trainingLosesMulti.append(trainLosses)
        trainAccuraciesMulti.append(trainAccuracies)
        validationAccuraciesMulti.append(valAccuracies)
        validationLossesMulti.append(valLosses)
        epochTimesMulti.append(epochTimes)
        trueLabelsListsMulti.append(trueLabelsList)
        predictedLabelsListsMulti.append(predictedLabelsList)
        
    return names, models, trainingLosesMulti, validationAccuraciesMulti, trainAccuraciesMulti, validationLossesMulti, epochTimesMulti, trueLabelsListsMulti, predictedLabelsListsMulti


