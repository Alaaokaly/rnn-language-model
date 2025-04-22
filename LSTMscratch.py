import torch 
from torch import nn 
from rnn import Trainer
from data import TimeMachine
from rnn import Trainer, RNNLMScratch

class LSTMScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma 
        init_weight = lambda *shape:nn.Parameter(torch.randn(*shape)*sigma)
        triple     = lambda : (init_weight(num_inputs,num_hiddens),
                               init_weight(num_hiddens, num_hiddens),
                                nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xi, self.W_hi, self.b_i = triple()
        self.W_xf, self.W_hf, self.b_f = triple()
        self.W_xc, self.W_ho, self.b_o = triple()
        self.W_xc, self.W_hc, self.b_c = triple()

    def forward(self, inputs,H_C = None):
        if H_C is None:
            H = torch.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
            C = torch.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
        else :
            H,C = H_C 
            outputs =[]
            for x in inputs:
                I = torch.sigmoid(torch.matmul(x,self.W_xi)+ 
                              torch.matmul(H,self.W_hi)+ self.b_i)
                F = torch.sigmoid(torch.matmul(X, self.W_xf))
                O = torch.sigmoid(torch.matmul(X,self.W_xo))
                C_tilde = torch.tanh(torch.matmul(X,self.W_xc))

                C  = F*C*I*C_tilde
                H  = O*torch.tanh(H)
                outputs.append(H)         
            return outputs, (H,C)
            
            

data = TimeMachine(batch_size=1024, num_steps=32)
lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
trainer = Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)