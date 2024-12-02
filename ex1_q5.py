from typing import List, Union

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# Neural Network Model
class LinearNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        return out

# Neural Network Model
class TwoLayerNet(nn.Module):
    def __init__(self, input_size: int, layer1_size: int, num_classes: int):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, layer1_size)
        self.fc2 = nn.Linear(layer1_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(nn.functional.relu(self.fc1(x)))
        return out

def train_model(train_loader, num_epochs: int, optimizer: torch.optim.Optimizer, net: nn.Module, criterion: nn.Module) -> List[float]:
    losses = []
    # Train the Model
    for _ in tqdm(range(num_epochs)):
        running_loss = 0.0
        for (data, labels) in train_loader:
            # Convert torch tensor to Variable - which contains data and gradient
            data = Variable(torch.flatten(data, 1))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"loss: {running_loss}")
        losses.append([running_loss])
    return losses


def plot_loss(losses: np.array, num_epochs: int):
    plt.figure()
    plt.plot(np.arange(num_epochs), losses)
    plt.xlabel('num epochs')
    plt.ylabel('total loss')

# Test the Model
def test_model(test_loader: torch.utils.data.DataLoader, net: nn.Module) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(torch.flatten(images, 1))
            # evaluation code - report accuracy
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            correct += (labels == predictions).sum()
            total += labels.size(0)
    accuracy = (100 * correct / total)
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f}%")
    return accuracy

def get_dataloaders(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, batch_size: int) -> Union[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    return train_loader, test_loader

def train_and_test(net: nn.Module, optimizer_class: torch.optim.Optimizer, train_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader, num_epochs: int, **kwargs) -> Union[List[float], float]:
    # Loss and Optimizer
    optimizer = optimizer_class(net.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()

    losses = train_model(train_loader, num_epochs, optimizer, net, criterion)
    accuracy = test_model(test_loader, net)
    return losses, accuracy

def save_model(net: nn.Module, file_name: str = 'model.pkl'):
    # Save the Model
    torch.save(net.state_dict(), file_name)
 

if __name__ == '__main__':
    # Hyper Parameters
    input_size = 784 # 28*28
    num_classes = 10
    num_epochs = 100
    batch_size = 100
    learning_rate = 1e-3
    layer1_size = 500

    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size)

    print("SGD training")
    linear_net1 = LinearNet(input_size, num_classes)
    losses_SGD, accuracy_SGD = train_and_test(linear_net1, torch.optim.SGD, train_loader, test_loader, num_epochs, lr=learning_rate)

    print("Adam training")
    linear_net2 = LinearNet(input_size, num_classes)
    losses_adam, accuracy_Adam = train_and_test(linear_net2, torch.optim.Adam, train_loader, test_loader, num_epochs)

    print("Two Layer Adam training")
    two_layer_net = TwoLayerNet(input_size, layer1_size, num_classes)
    losses_adam_two_layer, accuracy_two_layer = train_and_test(two_layer_net, torch.optim.Adam, train_loader, test_loader, num_epochs)

    plot_loss(np.column_stack((losses_SGD, losses_adam, losses_adam_two_layer)), num_epochs)
    plt.title("loss")
    plt.legend([f"SGD - single layer, accuracy: {accuracy_SGD:.2f}", f"Adam - single layer, accuracy: {accuracy_Adam:.2f}",
                 f"Adam - two layers, accuracy: {accuracy_two_layer:.2f}"])
    plt.savefig("loss_comparison.jpg")
