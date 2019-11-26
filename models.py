import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, dropout=0.2):
        super(Encoder, self).__init__()
        self.p = dropout
        self.conv1 = nn.Conv2d(1, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.max_pool2d(self.drop(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.drop(F.relu(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.relu(self.conv4(x))
        return x


class jointClassifier(nn.Module):

    def __init__(self, input_dim):
        super(jointClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                            nn.Linear(128, 3))
    def forward(self, input):
        return self.classifier(input)



class circleParametrizer(nn.Module):

    def __init__(self, spatial=True, device='cpu'):
        super(circleParametrizer, self).__init__()
        self.encoder = Encoder(dropout = 0.2)
        self.classifier = jointClassifier(64 if spatial else 625)
        self.spatial = spatial
        if self.spatial:
            self.xdim = torch.FloatTensor([i for i in range(25) for j in range(25)]).to(device)
            self.ydim = torch.FloatTensor([i for j in range(25) for i in range(25)]).to(device)
            self.SS = nn.Softmax(dim=2)

    def vectorize(self, x):
        if not self.spatial:
            x = x.mean(dim=1)
            return x.view(x.shape[0], -1)
        else:
            probs = self.SS(x.view(x.shape[0], x.shape[1], -1))
            E_X = (probs * self.xdim).sum(dim=2)
            E_Y = (probs * self.ydim).sum(dim=2)
            return torch.cat((E_X, E_Y), dim=1)

    def forward(self, input, spatial=False):
        return self.classifier(self.vectorize(self.encoder(input)))
