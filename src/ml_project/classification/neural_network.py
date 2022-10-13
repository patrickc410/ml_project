import torch
import torch.nn as nn
from torch.nn import Linear, Sigmoid, Softmax
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm


class NNGenreClassifier(nn.Module):
    def __init__(self, DIM_IN=11, DIM_HID=10, DIM_OUT=13, N_LAYERS=2):
        super(NNGenreClassifier, self).__init__()
        # self.linears = []
        # for i in range(N_LAYERS):
        #     if i == 0:
        #         lin = Linear(DIM_IN, DIM_HID)
        #     elif i == N_LAYERS - 1:
        #         lin = Linear(DIM_IN, DIM_OUT)
        #     else:
        #         lin = Linear(DIM_HID, DIM_HID)
        #     self.linears.append(lin)

        self.lin1 = Linear(DIM_IN, DIM_HID)
        self.lin2 = Linear(DIM_HID, DIM_HID)
        self.lin3 = Linear(DIM_IN, DIM_OUT)

        self.sig = Sigmoid()
        self.softmax = Softmax(dim=DIM_OUT)

    def forward(self, X):
        # for layer in self.linears:
        #     X = layer(X)
        #     X = self.sig(X)
        
        X = self.lin1(X)
        X = self.sig(X)
        X = self.lin2(X)
        X = self.sig(X)
        X = self.lin3(X)

        return self.softmax(X)


def train_network(model: torch.nn.Module, iterator: DataLoader):

    optim = Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    total_loss = 0
    for x, y in tqdm(iterator):
        ### YOUR CODE STARTS HERE (~6 lines of code) ###

        optim.zero_grad()

        output = model(x)
        output = torch.squeeze(output)
        loss = criterion(output, y)
        total_loss += loss

        loss.backward()
        optim.step()

        ### YOUR CODE ENDS HERE ###
    return total_loss
