from rnn import RNN
import torch

net = RNN()

print(net(torch.Tensor(10,10)))
