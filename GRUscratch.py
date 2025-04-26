from torch import nn 
import torch 

from rnn import Trainer
from data import TimeMachine
from rnn import Trainer, RNNLMScratch


class GRUscratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma 
        init_weight = lambda *shape:nn.Parameter(torch.randn(*shape)*sigma)
        triple     = lambda : (init_weight(num_inputs,num_hiddens),
                               init_weight(num_hiddens, num_hiddens),
                                nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xz, self.W_hz, self.b_z = triple()
        self.W_xr, self.W_hr, self.b_r = triple()
        self.W_xh, self.W_hh, self.b_h = triple()
    def forward(self, inputs, H=None):
        if H is None:
            H = torch.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
            outputs = []
            for x in inputs: 
                Z = torch.sigmoid(torch.matmul(x, self.W_xz) +
                        torch.matmul(H, self.W_hz) + self.b_z)
                R = torch.sigmoid(torch.matmul(x, self.W_xr) +
                                torch.matmul(H, self.W_hr) + self.b_r)
                H_tilde = torch.tanh(torch.matmul(x, self.W_xh) +
                                   torch.matmul(R * H, self.W_hh) + self.b_h)
                H = Z * H + (1 - Z) * H_tilde
                outputs.append(H)
                return outputs, H
            
data = TimeMachine(batch_size=1024, num_steps=32)
val_loader = data.dataloader(train=False)
train_loader = data.dataloader(train = True)
gru = GRUscratch(num_inputs=len(data.vocab), num_hiddens=32)

model = RNNLMScratch(gru,vocab_size=len(data.vocab), lr=1)      

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = Trainer(max_epochs=300, grad_clip_val=1, model = model, train_loader = train_loader,
                   val_loader = val_loader, optimizer= optimizer)
trainer.fit()
model.predict('it has', 20, data.vocab)
