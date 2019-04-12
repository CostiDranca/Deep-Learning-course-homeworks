# Tema 2 Deep Learning
# Dranca Constantin 334

# Preluare date
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from IPython.core.debugger import set_trace
import numpy as np
from matplotlib import pyplot
# we need google drive access to upload the datasets
from google.colab import drive
drive.mount('/content/gdrive')


# Training settings
kwargs={}
class Args():
  def __init__(self):
      self.batch_size = 64
      self.test_batch_size = 64
      self.epochs = 10#10
      self.lr = 0.01
      self.momentum = 0.9
      self.seed = 1
      self.log_interval = int(10000 / self.batch_size)
      self.cuda = False

args = Args()

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


import pickle
from google.colab import drive
drive.mount('/content/gdrive')
FILE_PATH_T = 'gdrive/My Drive/mnist_count_train.pickle'
FILE_PATH_TS = 'gdrive/My Drive/mnist_count_test.pickle'


def get_large_dataset(path, max_batch_idx=100, shuffle=False, first_k=5000):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    np_dataset_large = np.expand_dims(data['images'], 1)[:first_k]
    np_dataset_coords = data['coords'].astype(np.float32)[:first_k]
    np_dataset_no_count = data['no_count'].astype(np.float32)[:first_k]

    print(f'np_dataset_large shape: {np_dataset_large.shape}')
    for ii in range(5):
        iii = np_dataset_large[10 + ii].reshape((100, 100))
        pyplot.figure()
        pyplot.imshow(iii, cmap="gray")
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    dataset_large, dataset_coords, dataset_no_count = map(torch.tensor,
                                                          (np_dataset_large, np_dataset_coords, np_dataset_no_count))
    dataset_large = dataset_large.to(device)
    dataset_coords = dataset_coords.to(device)
    dataset_no_count = dataset_no_count.to(device)

    large_dataset = TensorDataset(dataset_large, dataset_coords, dataset_no_count)
    large_data_loader = DataLoader(large_dataset,
                                   batch_size=args.batch_size, shuffle=shuffle, drop_last=True)
    return large_data_loader


path_train = FILE_PATH_T
path_test = FILE_PATH_TS

large_data_loader_train = get_large_dataset(path_train, max_batch_idx=50, shuffle=True)
large_data_loader_test = get_large_dataset(path_test, max_batch_idx=50)


# Method 1
# Fully Convolutional Neural Network for digits counting
class CNN_fully_conv_count(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, no_filters1, 5, 1)
        self.conv2 = nn.Conv2d(no_filters1, no_filter2, 5, 1)
        self.fully_conv1 = nn.Conv2d(no_filter2, no_neurons1, 4)
        self.fully_conv2 = nn.Conv2d(no_neurons1, 10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.fully_conv1(x))
        x = self.fully_conv2(x)
        return F.log_softmax(x, dim=1)


def preporcess(data):
    return data.float() / 255.0


PATH = 'conv_net.pt'
torch.save(model.state_dict(), PATH)

model_count = CNN_fully_conv_count()

loaded_state_dict = torch.load(PATH)

model_dict = {}
for key, val in loaded_state_dict.items():
    key = key.replace('fc', 'fully_conv')
    if 'fully_conv1.weight' in key:
        val = val.view(-1, no_filter2, 4, 4)
    if 'fully_conv2.weigh' in key:
        val = val.view(-1, no_neurons1, 1, 1)
    model_dict[key] = val
model_count.load_state_dict(model_dict)
model_count = model_count.to(device)

# Best value: -0.0001  error: 145.96124362945557
prag = -0.0001
all_mse = []
all_acc = 0.0
exemple = 0.0

for batch_idx, (large_imgs, target_coords, digit_counts) in enumerate(large_data_loader_test):

    large_imgs = preporcess(large_imgs)
    out_prob_maps = model_count(large_imgs)
    # Maximul pe fiecare harta
    maxime = torch.max(out_prob_maps.view(args.batch_size, 10, -1), dim=2)[0]

    contor = torch.Tensor([0.0] * len(maxime)).to(device)
    # Primele 5 maxime
    maxime, index = torch.sort(maxime, descending=True)
    maxime = maxime.to(device)
    for i in range(len(maxime)):
        for j in range(5):
            if (maxime[i][j] >= prag):
                contor[i] = contor[i] + 1
    mse = torch.mean(torch.sqrt(torch.sum((contor - digit_counts) * (contor - digit_counts), dim=0))).to(device)
    accuracy = contor.eq(digit_counts.view_as(contor)).sum().item()
    all_acc += accuracy

    exemple += len(contor)
    print("Accuracy: ", accuracy)
    all_mse.append(mse.item())
    print("Eroare: ", mse.item())
print("Eroarea totala medie: ", torch.mean(torch.Tensor(all_mse)).item())
print("Total accuracy: ", 100. * all_acc / exemple)


# Method 2

# Arhitectura retea pentru numarare
class CNN_count_digits(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, no_filters1, 5, 1)
        self.conv2 = nn.Conv2d(no_filters1, no_filter2, 5, 1)
        self.fully_conv1 = nn.Conv2d(no_filter2, no_neurons1, 4)
        self.fully_conv2_new = nn.Conv2d(no_neurons1, 10, 1)

        self.linear_loc = nn.Linear(19 * 19 * 10, 5)

    def forward(self, xb):
        x = xb.view(-1, 1, xb.shape[2], xb.shape[3])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.fully_conv1(x))
        self.conv_act = self.fully_conv2_new(x).view(args.batch_size, -1)
        self.lin = self.linear_loc(self.conv_act)
        return F.log_softmax(self.lin, dim=1)

        # self.SoftMax = nn.LogSoftmax(dim =1)
        # return self.SoftMax(self.lin)


def plot_loss(loss, label, color='blue'):
    pyplot.plot(loss, label=label, color=color)
    pyplot.legend()


def train_counting(args, model, device, train_loader, optimizer, epoch):
    model.train()
    all_losses = []
    for batch_idx, (data, target_coords, digit_counts) in enumerate(large_data_loader_test):
        digit_counts = digit_counts.long()
        digit_counts = digit_counts - 1
        data, target_coords, digit_counts = data.to(device), target_coords.to(device), digit_counts.to(device)
        data = preporcess(data)
        optimizer.zero_grad()
        output = model(data)

        # criteriu = nn.NLLLoss()
        loss = F.nll_loss(output, digit_counts)

        loss.backward()
        all_losses.append(loss.data.cpu().numpy())
        optimizer.step()
        if False and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return np.array(all_losses).mean()


def test_counting(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target_coords, digit_counts in test_loader:
            digit_counts = digit_counts - 1
            data, target_coords, digit_counts = data.to(device), target_coords.to(device), digit_counts.to(device)
            data = preporcess(data)
            output = model(data)
            test_loss += F.nll_loss(output, digit_counts.long(), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred
            correct += pred.eq(digit_counts.long().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}    '.format(test_loss), "Acurracy: ",
          100. * correct / len(test_loader.dataset))
    return test_loss


PATH = 'conv_net.pt'
torch.save(model.state_dict(), PATH)

loc_model_scratch = CNN_count_digits()

# Model pentru transfer Learning
loc_model_pretrained = CNN_count_digits()

loaded_state_dict = torch.load(PATH)
model_dict = {}
for key, val in loaded_state_dict.items():
    key = key.replace('fc', 'fully_conv')
    print(f'key: {key}')
    if 'fully_conv1.weight' in key:
        val = val.view(-1, no_filter2, 4, 4)
    if 'fully_conv2.weigh' in key:
        val = val.view(-1, no_neurons1, 1, 1)
    model_dict[key] = val

loc_model_scratch = loc_model_scratch.to(device)

loc_model_pretrained.load_state_dict(model_dict, strict=False)
loc_model_pretrained = loc_model_pretrained.to(device)

loc_model_pretrained.linear_loc.weight.data

optimizer_loc_scratch = torch.optim.Adam(loc_model_scratch.parameters(), lr=0.001)
optimizer_loc_pretrained = torch.optim.Adam(loc_model_pretrained.parameters(), lr=0.001)

losses_train = []
losses_test = []
print("Training for Scratch Net: ")
for epoch in range(1, args.epochs + 20):
    print("\n", f'Scratch epoch: {epoch}')
    train_loss = train_counting(args, loc_model_scratch, device, large_data_loader_train, optimizer_loc_scratch, epoch)
    test_loss = test_counting(args, loc_model_scratch, device, large_data_loader_test)

    losses_train.append(train_loss)
    losses_test.append(test_loss)

losses_train_pre = []
losses_test_pre = []

print("Training for Pretrained Net: ")
for epoch in range(1, args.epochs + 20):
    print("\n", f'Pretrained epoch: {epoch}')
    train_loss = train_counting(args, loc_model_pretrained, device, large_data_loader_train, optimizer_loc_pretrained,
                                epoch)
    test_loss = test_counting(args, loc_model_pretrained, device, large_data_loader_test)

    losses_train_pre.append(train_loss)
    losses_test_pre.append(test_loss)

plot_loss(losses_train, 'scratch_train_loss', 'red')
plot_loss(losses_test, 'scratch_test_loss')

plot_loss(losses_train_pre, 'pretrained_train_loss', 'pink')
plot_loss(losses_test_pre, 'pretrained_test_loss', 'green')
