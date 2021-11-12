import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
torch.set_default_dtype(torch.float64)

class Stock_Classifier(nn.Module):
    def __init__(self, start):
        super(Stock_Classifier, self).__init__()
        print("Number of features: %d" % start)
        self.fc1 = nn.Linear(start, 24)
        self.fc2 = nn.Linear(24,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4,2)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
	batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):

    if running_mode == "train":
        if valid_set:
            n = 0
            cond = True
            loss = []
            accuracy = []
            lossv = []
            accuracyv = []
            while cond:
                train_loader = DataLoader(train_set)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model, l, a, p = _train(model, train_loader, optimizer)
                loss.append(l)
                accuracy.append(a)

                valid_loader = DataLoader(valid_set)
                lv, av, predictions = _test(model, valid_loader)
                lossv.append(lv)
                accuracyv.append(av)

                n += 1

                if lv <= stop_thr or n == n_epochs:
                    cond = False

            rl = {'train':loss, 'valid':lossv}
            ra = {'train':accuracy*100, 'valid':accuracyv*100}

            return model, rl, ra, p

        else:
            loss = []
            accuracy = []
            for i in range(n_epochs):
                train_loader = DataLoader(train_set, batch_size, shuffle=shuffle)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model, l, a, p = _train(model, train_loader, optimizer)
                loss.append(l)
                accuracy.append(a)

            rl = {'train':loss, 'valid':[]}
            ra = {'train':accuracy*100, 'valid':[]}

            return model, rl, ra, p
    else:
        test_loader = DataLoader(valid_set, batch_size, shuffle=shuffle)
        loss, accuracy, predictions = _test(model, test_loader)
        return model, loss, accuracy, predictions

def _train(model, data_loader, optimizer, device=torch.device('cpu')):
    criterion = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []
    predictions = []

    for i, (inputs, targets) in enumerate(data_loader):
        outputs = model(inputs)
        targets = targets.long()

        loss = criterion(outputs, torch.max(targets,1)[1])
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = targets.size(0)
        _, predicted = torch.max(outputs.data, 1) # This line for prediction
        correct = (predicted == targets).sum().item()
        predictions.append(predicted)

        acc_list.append(correct/total)

    return model, sum(loss_list)/len(loss_list), 100*sum(acc_list)/len(acc_list), predictions

def _test(model, data_loader, device=torch.device('cpu')):

    criterion = nn.CrossEntropyLoss()
    loss_list = []
    acc_list = []
    predictions = []

    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.double()
        outputs = model(inputs)
        targets = targets.long()
        loss = criterion(outputs, torch.max(targets,1)[1])
        loss_list.append(loss.item())

        loss.backward()

        total = targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        acc_list.append(correct/total)
        predictions.append(predicted.item())

    return sum(loss_list)/len(loss_list), 100*sum(acc_list)/len(acc_list), predictions
