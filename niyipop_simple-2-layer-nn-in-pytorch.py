import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
print(os.listdir("../input"))


# Read the training and test[](http://) datasets


%%time
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)


# Split the training dataset for training and validation


y = train['target'].values
X = train.drop(['ID_code', 'target'], axis=1).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


print(len(X_train), len(X_val))
print(len(y_train), len(y_val))


# Construct a 2-Layer NN


#Seed
torch.manual_seed(1234)

#hyperparameters
hl = 10
lr = 0.01
num_epoch = 100

#Model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(200, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


# Train the NN


%%time
#train
for epoch in range(num_epoch):
    X = Variable(torch.Tensor(X_train).float())
    Y = Variable(torch.Tensor(y_train).long())

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()

    if (epoch) % 10 == 0:
        print ('Epoch [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epoch, loss.item()))


# Validate the NN


%%time

#Validation
X = Variable(torch.Tensor(X_val).float())
Y = torch.Tensor(y_val).long()
out = net(X)

_, predicted = torch.max(out.data, 1)

#get accuration
print('Accuracy of the network %d %%' % (100 * torch.sum(Y==predicted) / len(y_val)))


# Perform prediction on test dataset


%%time

#Test
X_test = test.drop(['ID_code'], axis=1).values

X = Variable(torch.Tensor(X_test).float())
out = net(X)

_, predicted = torch.max(out.data, 1)


# Output prediction to CSV


ID_code = test['ID_code']
target = predicted.data.numpy()

my_submission = pd.DataFrame({'ID_code': ID_code, 'target': target})
my_submission.to_csv('submission.csv', index=False)

my_submission.head()



