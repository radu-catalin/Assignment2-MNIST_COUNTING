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

# hyper parameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001
momentum = .9
no_filter1 = 20
no_filter2 = 50
no_neurons = 500
first_k = 10000
log_interval = int(1000 / batch_size)

path_train = './assignment2/data/mnist_count_train.pickle'
path_test = './assignment2/data/mnist_count_test.pickle'


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
    shuffle=True,
    first_k=first_k
)

test_loader = get_dataset(
    path=path_test
)

examples = iter(train_loader)
sample, labels = examples.next()

print('labels:', labels)


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

        # plt.figure()
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(x[i][0].detach().numpy())

        x = self.pool(F.relu(self.conv2(x)))

        # plt.figure()
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(x[i][0].detach().numpy())

        x = F.relu(self.fully_conv1(x))

        # plt.figure()
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(x[i][0].detach().numpy())

        x = self.fully_conv2(x)

        # plt.figure()
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(x[i][0].detach().numpy())
        # print(x.shape)
        x = x.view(x.size(0), -1)

        # print(x.shape)
        x = self.linear_loc(x)
        # x = torch.sigmoid(x)
        # print(x.shape)
        # plt.show()
        return x


def plot_loss(loss, label, color='blue'):
    plt.plot(loss, label=label, color=color)
    plt.legend()
    plt.show()


def train_localisation(model, device, train_loader, optimizer, epoch):
    model.train()
    all_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = preprocess(data)
        optimizer.zero_grad()
        output = model(data)
        # as we want to predict real numbers, from a continuous space ([0,1])
        # we use mean-square-error loss (L2 loss)
        loss = F.mse_loss(output, target)

        loss.backward()
        all_losses.append(loss.data.cpu().numpy())
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.array(all_losses).mean()


def test_localisation(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        num_iter = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = preprocess(data)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()
            num_iter += 1
    test_loss /= num_iter

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss


model = ConvNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate
)

# losses_train = []
# losses_test = []


# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     all_losses = []
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # put the data on the GPU
#         data, target = data.to(device), target.to(device)
#         # initialize as zeros all the gradients of the model
#         optimizer.zero_grad()
#         data = preprocess(data)
#         # target = target.long()
#         # obtain the predictions in the FORWARD pass of the network
#         output = model(data)
#         # compute average LOSS for the current batch
#         loss = F.nll_loss(F.mse_loss(output, target), target)
#         all_losses.append(loss.detach().cpu().numpy())
#         # BACKPROPAGATE the gradients
#         loss.backward()
#         # use the computed gradients to OPTIMISE the model
#         optimizer.step()
#         # print the training loss of each batch
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#     return np.array(all_losses).mean()


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         num_iter = 0
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             # obtain the prediction by a forward pass
#             data = preprocess(data)
#             output = model(data)
#             # calculate the loss for the current batch and add it across the entire dataset
#             # sum up batch loss
#             test_loss += F.nll_loss(F.mse_loss(output, target), target)
#             # compute the accuracy of the predictions across the entire dataset
#             # get the most probable prediction
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).float().mean().item()
#             num_iter += 1
#     test_loss /= num_iter
#     test_accuracy = 100. * correct / num_iter
#     # print the Accuracy for the entire dataset
#     print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
#         test_loss,
#         test_accuracy))
#     return test_loss, test_accuracy

# for epoch in range(1, num_epochs):

#     print(f'Scratch epoch: {epoch}')
#     train_loss = train_localisation(
#         model, device, train_loader, optimizer, epoch)
#     test_loss = test_localisation(model, device, test_loader)

#     losses_train.append(train_loss)
#     losses_test.append(test_loss)

# plot_loss(losses_train, 'scratch_train_loss', 'red')
# plot_loss(losses_test, 'scratch_test_loss')


# losses_train = []
# losses_test = []
# accuracy_test = []
# for epoch in range(1, num_epochs + 1):
#     # for epoch in range(1, 3):
#     train_loss = train(model, device, train_loader, optimizer, epoch)
#     test_loss, test_accuracy = test(model, device, test_loader)
#     losses_train.append(train_loss)
#     losses_test.append(test_loss)
#     accuracy_test.append(test_accuracy)


###########################################
# examples = iter(train_loader)
# samples, labels = examples.next()
# samples.to(device)
# outputs = model(samples)

# for i, (images, labels) in enumerate(train_loader):
#     # images = images.to(device)
#     images = images.to(device).float()

#     plt.figure(1)
#     for i in range(6):
#         plt.subplot(2, 3, i + 1)
#         plt.imshow(images[i][0])
#     outputs = model(images)
#     print(outputs.shape)

#     break
# print(outputs)
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.reshape(-1, 19 * 19).to(device)
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        # print('label', labels)
        images = images.float() / 255.0
        # forward
        outputs = model(images)

        # outputs = outputs.type(torch.LongTensor)
        loss = F.mse_loss(outputs, labels)
        # print(outputs)
        # loss = F.nll_loss(F.softmax(outputs), labels.long())
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
        # predicted = torch.ceil(predicted)
        predicted = predicted.view(-1, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        # for i in range(batch_size):
        #   label = labels[i]
        #   pred = predicted[i]

        #   if label == pred:
        #     n_class_correct[label] += 1

        #   n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc}%')
