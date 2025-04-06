import torch
import requests
from torch.utils.data import Dataset, DataLoader

class CharacterDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def prepareData(text, sequenceLen, k, batchSize):
    chars = sorted(list(set(text)))
    vocabSize = len(chars)  # Changed to camelCase

    charToIdx = {char: i for i, char in enumerate(chars)}
    idxToChar = {i: char for char, i in charToIdx.items()}

    encode = lambda string: [charToIdx[c] for c in string]
    decode = lambda indices: ''.join([idxToChar[idx] for idx in indices])

    encodedText = encode(text)

    sequences, targets = [], []
    for i in range(0, len(encodedText) - sequenceLen[k]):
        sequence = encodedText[i: i + sequenceLen[k]]
        target = encodedText[i + sequenceLen[k]]

        sequences.append(sequence)
        targets.append(target)

    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    dataset = CharacterDataset(sequences, targets)

    trainSize = int(len(dataset) * 0.8)
    testSize = len(dataset) - trainSize
    trainingDataset, validationDataset = torch.utils.data.random_split(
        dataset, [trainSize, testSize]
    )

    trainingLoader = DataLoader(
        dataset=trainingDataset, 
        batch_size=batchSize, 
        shuffle=True
    )
    validationLoader = DataLoader(
        dataset=validationDataset, 
        batch_size=batchSize, 
        shuffle=False
    )

    return trainingLoader, validationLoader, charToIdx, idxToChar, vocabSize

def prepareDataURL(url, sequenceLen, k, batchSize):
    response = requests.get(url)
    text = response.text

    return prepareData(text, sequenceLen, k, batchSize)

def prepareMultiDataURL(url, sequenceLen, batchSize):
    response = requests.get(url)
    text = response.text

    return prepareMultiData(text, sequenceLen, batchSize)

def prepareMultiData(text, sequenceLen, batchSize):
    trainingLoaders = []
    validationLoaders = []
    charToIdxs = []
    idxToChars = []
    vocabSizes = []

    for i in range(len(sequenceLen)):
        trainingLoader, validationLoader, charToIdx, idxToChar, vocabSize = prepareData(text, sequenceLen, i, batchSize)
        trainingLoaders.append(trainingLoader)
        validationLoaders.append(validationLoader)
        charToIdxs.append(charToIdx)
        idxToChars.append(idxToChar)
        vocabSizes.append(vocabSize)

    return trainingLoaders, validationLoaders, charToIdxs, idxToChars, vocabSizes



