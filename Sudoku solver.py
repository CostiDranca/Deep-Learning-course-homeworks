import torch
import torch.nn as nn #PyTorch's module wrapper
import torch.optim as optim #PyTorch's optimiser
from torch.autograd import Variable #PyTorch's implementer of gradient descent and back propogation
import torch.nn.functional as F
import numpy as np
import random
quizzes = np.zeros((1000000, 81), np.int32)
solutions = np.zeros((1000000, 81), np.int32)
for i, line in enumerate(open('D:\\python projects\\ProiectImagine\\sudoku.csv', 'r').read().splitlines()[1:10000]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        quizzes[i, j] = q
        solutions[i, j] = s

quizz_train= quizzes[:8000]
solution_train = solutions[:8000]
test = quizzes[8000:9000]
test_labels = solutions[8000:9000]

train_var = Variable(torch.FloatTensor(quizz_train), requires_grad = False)
labels_var = Variable(torch.FloatTensor(solution_train), requires_grad = False)
test= Variable(torch.FloatTensor(test), requires_grad = False)
test_labels = Variable(torch.FloatTensor(test_labels), requires_grad = False)

print(train_var[0])
print(labels_var[0])

class Neural_Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Neural_Net, self).__init__()
        self.h1_layer = nn.Linear(in_size, 500)
        self.h2_layer = nn.Linear(500, 200)
        self.o_layer = nn.Linear(200, 81)
    def forward(self,x):
        y1 = F.rrelu(self.h1_layer(x))
        y2 = F.rrelu(self.h2_layer(y1))
        y3 = self.o_layer(y2)
        rezultat = F.rrelu(y3)
        return rezultat



model = Neural_Net(81,81)
loss_function = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.01)
batch_size = 1000
nr_batches = 7

all_losses = []
val_loss = []
for num in range(400):
    #Get random batch from training data
    index = random.choice(np.random.permutation(nr_batches))
    batch_train = train_var[index*batch_size:(index+1)*batch_size]
    batch_labels = labels_var[index*batch_size:(index+1)*batch_size]

    rezultat = model(batch_train)
    loss = loss_function(rezultat, batch_labels)
    all_losses.append(loss.data)
    optim.zero_grad()
    loss.backward()
    optim.step()

    loss = loss_function(model(test),test_labels)
    val_loss.append(loss.data)
    if num % 50 ==0 or num == 0:
        print(loss)

print("REZZULTATUL Pentru primul exemplu: ",model(train_var[0]))

import matplotlib.pyplot as plt
#%matplotlib inline
all_losses = np.array(all_losses, dtype = np.float)

plt.plot(all_losses)
plt.plot(val_loss)
plt.show()





