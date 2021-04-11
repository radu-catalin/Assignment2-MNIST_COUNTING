import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 64
num_epochs = 5
learning_rate = 0.01
momentum = 0.9
cuda = False

# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
device = torch.device('cpu')

no_filter1 = 20
no_filter2 = 50
no_neurons = 500


# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, no_filter1, 5, 1)
        self.conv2 = nn.Conv2d(no_filter1, no_filter2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * no_filter2, no_neurons)
        self.fc2 = nn.Linear(no_neurons, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 4 * no_filter2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def train_model(model, train_loader, optimizer, epoch):
    model.train()
    all_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)
        all_losses.append(loss.detach().cpu().numpy())

        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return np.array(all_losses).mean()


def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        num_iter = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).float().mean().item()
            num_iter += 1
    test_loss /= num_iter
    test_accuracy = 100.0 * correct / num_iter

    print(f'\nAverage loss: {test_loss:.4f}, Accuracy: {test_accuracy:.0f}%\n')
    return test_loss, test_accuracy


def plot_loss(loss, label, color='red'):
    plt.plot(loss, label=label, color=color)
    plt.legend()


model = CNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

losses_train = []
losses_test = []
accuracy_test = []

for epoch in range(num_epochs):
    train_loss = train_model(
        model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
    test_loss, test_accuracy = test_model(model=model, test_loader=test_loader)
    losses_train.append(train_loss)
    losses_test.append(test_loss)
    accuracy_test.append(test_accuracy)

plt.figure(1)
plot_loss(losses_train, 'train_loss', 'red')
plot_loss(losses_test, 'test_loss', 'blue')
plt.show()

plt.figure(2)
plot_loss(accuracy_test, 'test_accuracy')
plt.show()

print('test', losses_test, losses_train)

torch.save(model.state_dict(), 'mnist_cnn.pt')
