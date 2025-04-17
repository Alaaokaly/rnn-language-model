# the hidden state doesn't directly affect how far back in time the network can remember 
# but rather affect the richiness of the representation "features" of each time step
#so the number of hiffen units doesn't change the memory span of the network (time_steps)
import math 
import torch 
from torch import nn 
from torch.nn import functional as F
from data import TimeMachine

# num_inputs is the size of vocab : in this case would be number of the letters in the corpus 
# numbder of steps is the distance we have in time :sequence length 
# number of hiddens is the number of features each time step has 
# batch size is the number of sequence the model sees at each time stap and generates an equevlante 
   #number of hidden states 

class RNNScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs  = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        self.W_xh = nn.Parameter(torch.randn(num_inputs,num_hiddens)*self.sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens,num_hiddens)*self.sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))
   
    def forward(self, inputs, state=None):
        if state is None:
            state = torch.zeros((inputs.shape[1], self.num_hiddens),
                                device='cpu')
        outputs = []
        for X in inputs:
            state = torch.tanh(torch.matmul(X, self.W_xh) +
                               torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state


class RNNLMScratch(nn.Module):
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.rnn= rnn
        self.vocab_size = vocab_size
        self.lr = lr 
        self.loss_fn = nn.CrossEntropyLoss()
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def loss(self, outputs, labels):
        return self.loss_fn(outputs.view(-1, self.vocab_size), labels.view(-1))

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=False)

    def one_hot(self, X):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    
    def output_layer(self, rnn_outputs):
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, 1)
    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs) 
    
    def predict(self, prefix, num_preds, vocab, device=None):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
                with open ('predictio.text','w') as f:
                    f.write(f"The Input is :{prefix}")
                    out = ''.join([vocab.idx_to_token[i] for i in outputs])
                    f.write(f"The predicted is {out}")
        return out
        
           



class Trainer :
    def __init__(self ,model, train_loader, val_loader = None, optimizer= None,
                  max_epochs= 10, grad_clip_val =0.0,  
                  device = 'cpu'): 
        self.model = model
        self.trainer_loader = train_loader 
        self.val_loader = val_loader
        self.optimizer = optimizer
       
        self.max_epochs = max_epochs
        self.gradient_clip_val  = grad_clip_val
        self.device = device

    def train_one_epoch (self): # train by epoch 
        self.model.train()
        total_loss = 0 
        for x, y in self.trainer_loader:
            x, y = x.to(self.device), y.to(self.device) # Pytorch method to transfer a tensor, so data and model are on the same device
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, y)
            loss.backward() # compute backward gradients 
            if self.gradient_clip_val > 0:
                # calculate the sum norm of params 
                #if norm >grad_clip_valu 
                # param_grid *= grad_clip_val/ norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip_val)
            self.optimizer.step() #update the new values for params using grad 
            total_loss+= loss.item()
        return total_loss / len(self.trainer_loader) # mean error value 
    
    def predict(self, prefix, num_preds, vocab, device=None):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
    def validate (self):
        self.model.validation_step()
        total_loss = 0 
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.model.loss(outputs, y )
                total_loss += loss.item()
        return total_loss / len(self.trainer_loader) # mean error value 
    
    def fit (self):
        with open('training_results.txt', 'w') as f:
            for epoch in range(1,self.max_epochs):
                train_loss = self.train_one_epoch()
                val_loss = self.validate() if self.val_loader else None
    
                if val_loss is not None:
                    log_message = f"Epoch {epoch}/{self.max_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                    f.write(log_message + '\n')
                   
                    print(log_message)
                else:
                    log_message = f"Epoch {epoch}/{self.max_epochs} - Train Loss: {train_loss:.4f}"
                    f.write(log_message + '\n')
                   
                    print(log_message)
                
                # Flush to ensure results are written immediately
                f.flush()



data = TimeMachine(batch_size=1024, num_steps=32)
train_loader = data.dataloader(train = True)
val_loader = data.dataloader(train=False)

rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trainer = Trainer(max_epochs=300, grad_clip_val=1, model = model, train_loader = train_loader,
                   val_loader = val_loader, optimizer= optimizer)
trainer.fit()

model.predict('it has', 20, data.vocab)
