import torch
from torch.utils.data import TensorDataset, DataLoader
import re
import collections
import random


class TimeMachine():
    def __init__(self,batch_size, num_steps, num_train=0.75, num_val = 0.25):
        self.batch_size = batch_size
        
        self.num_steps = num_steps 
        
        corpus, self.vocab = self.build(self.get_data())
        self.num_val = int(num_val * len(corpus))
        self.num_train = int( num_train * len(corpus))
        array  = torch.tensor( [ corpus[i:i+num_steps] for i in range(len(corpus)-num_steps)])
        self.X,self.Y = array[:, :-1], array[:, 1:]

    def get_data(self):
        fname = 'The Project Gutenberg eBook of The Time .txt'
        with open(fname) as f:
            return f.read()
        
    def process(self, text):
        return re.sub('[^A-Za-z]+',' ',text).lower()
    
    def tokenize(self,text):
        return list(text)
    
    def build(self, raw_text, vocab =None):
        tokens = self.tokenize(self.process(raw_text))
        if vocab is None :
            vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens ]
        return corpus, vocab
    def dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train,self.num_val)
        dataset = TensorDataset(self.X[idx],self.Y[idx])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train)
        return dataloader

    
class Vocab:
    def __init__(self,tokens = [],min_freq=0,reserved_tokens = []):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line ]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(),key=lambda x:x[1], reverse=True)
         # sorted llist of unique letters/items in the corpus
         #helps to get the letter from its index/order in the array
        self.idx_to_token = list(sorted(set(['<UNK>']+ reserved_tokens + [token for token, freq in self.token_freqs if freq
        > min_freq])))
        # a dictionary that of token: index by enumarting (giving each token an index starting from 0) each letter 
        # helps y taking the letter and gives it's index 
        self.token_to_index = {token :idx for idx, token in enumerate(self.idx_to_token)}
       
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        """iterate the dictionary by letter to get the index"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_index.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        """iterrate the array by index to get the letter """
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    @property
    def unk(self):  
        """a fancy way to write :
               token_to_index.get(tokens, self.token_to_index['<UNK>'])
                in __getitem__() """
        return self.token_to_index['<UNK>']
    
data = TimeMachine(batch_size=5,num_steps=5)
i = 1
for x, y in data.dataloader(train = True):
    
    print('x:' ,x,'\n', 'y:',y)
    i+=1
    if i == 3 :
        break





