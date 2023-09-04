import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
        
class NeuralTSDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, style='ts'):
        self.func = extend(Network(dim, hidden_size=hidden).cuda())
        self.context_list = None
        self.len = 0
        self.reward = None
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        mu = self.func(tensor)
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()
        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
        sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), 0, 0
    
    def train(self, context, reward):
        self.len += 1
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba / self.len)
        if self.context_list is None:
            self.context_list = torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.reward = torch.tensor([reward], device='cuda', dtype=torch.float32)
        else:
            self.context_list = torch.cat((self.context_list, torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.reward = torch.cat((self.reward, torch.tensor([reward], device='cuda', dtype=torch.float32)))
        if self.len % self.delay != 0:
            return 0
        for _ in range(100):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(self.context_list).view(-1)
            loss = self.loss_func(pred, self.reward)
            loss.backward()
            optimizer.step()
        return 0