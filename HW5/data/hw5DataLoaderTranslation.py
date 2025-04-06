from torch.utils.data import Dataset
import torch

class Vocabulary:
    def __init__(self):
        # Initialize dictionaries for word to index and index to word mappings
        self.word2index = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2}
        self.index2word = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>'}
        self.word_count = {}  # Keep track of word frequencies
        self.n_words = 3  # Start counting from 2 to account for special tokens

    def add_sentence(self, sentence):
        # Add all words in a sentence to the vocabulary
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        # Add a word to the vocabulary
        if word not in self.word2index:
            # Assign a new index to the word and update mappings
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            # Increment word count if the word already exists in the vocabulary
            self.word_count[word] += 1

# Custom Dataset class for English to French sentences
class EngFrDataset(Dataset):
    def __init__(self, pairs, max_length=10):
        self.eng_vocab = Vocabulary()
        self.fr_vocab = Vocabulary()
        self.pairs = []
        self.max_length = max_length

        # Process each English-French pair
        for eng, fr in pairs:
            self.eng_vocab.add_sentence(eng)
            self.fr_vocab.add_sentence(fr)
            self.pairs.append((eng, fr))

        # Separate English and French sentences
        self.eng_sentences = [pair[0] for pair in self.pairs]
        self.fr_sentences = [pair[1] for pair in self.pairs]
    
    def pad_sentence(self, sentence, max_length):
        # Split the sentence into words
        words = sentence.split()
        # If the sentence is shorter than max_length, pad it with '<PAD>'
        if len(words) < max_length:
            words += ['<PAD>'] * (max_length - len(words))
        # If the sentence is longer than max_length, truncate it
        elif len(words) > max_length:
            words = words[:max_length]
        return ' '.join(words)

    def __len__(self):
        # Return the number of sentence pairs
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get the sentences by index
        eng_sentence = self.pad_sentence(self.eng_sentences[idx], max_length=self.max_length)
        fr_sentence = self.pad_sentence(self.fr_sentences[idx], max_length=self.max_length)
        input_indices = [self.eng_vocab.word2index[word] for word in eng_sentence.split()] + [1]
        target_indices = [self.fr_vocab.word2index[word] for word in fr_sentence.split()] + [1]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)