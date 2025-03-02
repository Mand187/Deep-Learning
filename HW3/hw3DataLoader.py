import torch
from torch.utils.data import Dataset, DataLoader, random_split
import requests

class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

def prepareDataset(text, sequenceLength):
    chars = sorted(set(text))
    charToIdx = {ch: i for i, ch in enumerate(chars)}
    idxToChar = {i: ch for ch, i in charToIdx.items()}
    
    encodedText = [charToIdx[ch] for ch in text]

    if len(encodedText) <= sequenceLength:
        raise ValueError("Text is too short for the given sequence length.")

    sequences = []
    targets = []
    for i in range(len(encodedText) - sequenceLength):
        sequences.append(encodedText[i:i + sequenceLength])
        targets.append(encodedText[i + sequenceLength])

    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return sequences, targets, chars, charToIdx, idxToChar

def getDataLoaders(url, sequenceLength=20, batchSize=128, trainSplit=0.8):
    response = requests.get(url)
    text = response.text

    sequences, targets, chars, charToIdx, idxToChar = prepareDataset(text, sequenceLength)

    dataset = CharDataset(sequences, targets)
    
    torch.manual_seed(42)
    trainSize = int(len(dataset) * trainSplit)
    testSize = len(dataset) - trainSize
    trainDataset, testDataset = random_split(dataset, [trainSize, testSize])

    trainLoader = DataLoader(trainDataset, shuffle=True, batch_size=batchSize)
    testLoader = DataLoader(testDataset, shuffle=False, batch_size=batchSize)

    return trainLoader, testLoader, chars, charToIdx, idxToChar

def simpleTextDataLoader(sequenceLength=20, batchSize=128, trainSplit=0.8):
    text = "Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text. At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model. One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks. Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time. Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants. In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."

    sequences, targets, chars, charToIdx, idxToChar = prepareDataset(text, sequenceLength)

    dataset = CharDataset(sequences, targets)

    torch.manual_seed(42)
    trainSize = int(len(dataset) * trainSplit)
    testSize = len(dataset) - trainSize
    trainDataset, testDataset = random_split(dataset, [trainSize, testSize])

    trainLoader = DataLoader(trainDataset, shuffle=True, batch_size=batchSize)
    testLoader = DataLoader(testDataset, shuffle=False, batch_size=batchSize)

    return trainLoader, testLoader, chars, charToIdx, idxToChar
