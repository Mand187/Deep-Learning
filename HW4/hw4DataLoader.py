import torch
from torch.utils.data import Dataset, DataLoader
from config import SOSToken, EOSToken

class Vocabulary:
    def __init__(self):
        self.wordToIndex = {"SOS": SOSToken, "EOS": EOSToken}
        self.indexToWord = {SOSToken: "SOS", EOSToken: "EOS"}
        self.wordCount = {}
        self.numWords = 2  # Start counting from 2 (for special tokens)

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.wordToIndex:
            self.wordToIndex[word] = self.numWords
            self.indexToWord[self.numWords] = word
            self.wordCount[word] = 1
            self.numWords += 1
        else:
            self.wordCount[word] += 1

class EngFrDataset(Dataset):
    def __init__(self, pairs, engVocab, frVocab):
        self.engVocab = engVocab
        self.frVocab = frVocab
        self.pairs = []
        
        for eng, fr in pairs:
            self.engVocab.addSentence(eng)
            self.frVocab.addSentence(fr)
            self.pairs.append((eng, fr))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        engSentence, frSentence = self.pairs[idx]
        inputIndices = [self.engVocab.wordToIndex[word] for word in engSentence.split()] + [EOSToken]
        targetIndices = [self.frVocab.wordToIndex[word] for word in frSentence.split()] + [EOSToken]
        
        return torch.tensor(inputIndices, dtype=torch.long), torch.tensor(targetIndices, dtype=torch.long)

def dataLoader(textData, batchSize=1, shuffle=True):
    engVocab = Vocabulary()
    frVocab = Vocabulary()
    dataset = EngFrDataset(textData, engVocab, frVocab)
    
    trainDataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
    validDataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
    
    return trainDataLoader, validDataLoader, engVocab, frVocab