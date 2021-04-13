import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# hyperparameters
num_epochs = 10
batch_size = 50
learning_rate = 0.001
# momentum = .9
no_filter1 = 30
no_filter2 = 50
no_neurons = 300
first_k = 5000
log_interval = int(1000 / batch_size)

path_train = './data/mnist_count_train.pickle'
path_test = './data/mnist_count_test.pickle'


def preprocess(data):
    return data.float() / 255.0


def get_dataset(path: str, shuffle: bool = False, first_k=1000) -> DataLoader:
    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    np_dataset_images = np.expand_dims(data['images'], 1)[:first_k]
    np_dataset_no_digits = data['no_count'].astype(np.float32)[:first_k]

    dataset_images, dataset_no_digits = map(
        torch.tensor,
        (np_dataset_images, np_dataset_no_digits)
    )

    dataset_images = dataset_images.to(device)
    dataset_no_digits = dataset_no_digits.to(device)

    dataset_images = TensorDataset(dataset_images, dataset_no_digits)

    dataset_loader = DataLoader(
        dataset=dataset_images,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataset_loader


train_loader = get_dataset(
    path=path_train,
    shuffle=False,
    first_k=first_k
)

test_loader = get_dataset(
    path=path_test
)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=no_filter1,
            kernel_size=5,
            stride=1
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=no_filter1,
            out_channels=no_filter2,
            kernel_size=5,
            stride=1
        )

        self.fully_conv1 = nn.Conv2d(no_filter2, no_neurons, 4)
        self.fully_conv2 = nn.Conv2d(no_neurons, 5, 1)

        self.linear_loc = nn.Linear(19 * 19 * 5, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fully_conv1(x))
        x = self.fully_conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_loc(x)
        return x


def plot_loss(loss, label, color='blue'):
    plt.plot(loss, label=label, color=color)
    plt.legend()


model = ConvNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate
)

n_total_steps = len(train_loader)
losses_train = []
losses_test = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        images = preprocess(images)
        # forward
        outputs = model(images)
        # print(outputs)
        loss = F.mse_loss(outputs, labels)
        losses_train.append(loss.detach().cpu().numpy())
        # loss = F.nll_loss(F.softmax(outputs), labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % log_interval == 0:
            print(
                f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
            )

print('Finished training!')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for (images, labels) in test_loader:
        images = images.to(device).float()
        labels = labels.to(device).view(-1, 1)

        outputs = model(images)

        _, predicted = torch.max(outputs, dim=-1)
        predicted = predicted.view(-1, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc}%')

plot_loss(losses_train, label='loss')
plt.show()
