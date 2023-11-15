"""
Contains the MNISTModel class the other scripts are built to verify, as well as: the training and validation datasets;
    a Trainer class built to train the model on the datasets; and a helper function to load an instance of the MNISTModel
    from a state dictionary file.
"""

import random
from statistics import mean

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import tqdm, trange

to_tensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
train_dataset = mnist.MNIST(root="./data", train=True, transform=to_tensor, download=True)
validation_dataset = mnist.MNIST(root="./data", train=False, transform=to_tensor, download=True)


def load_MNIST_model(layer_sizes=(784, 100, 100, 10), activation=None, flatten=True, filename="mnist_model.pt"):
    """
    Construct an MNISTModel and load state dictionary from a file.
    :param layer_sizes: An iterable of the number of neurons in each layer. Defaults to (28x28, 100, 100, 10), ie,
                        an MLP for the MNIST problem using 2 hidden layers, 784 input neurons, and 10 output classes.
    :param activation: The activation module used in the original model. If None, uses default from MNISTModel class.
    :param flatten: Whether the original model had an nn.Flatten layer at the start. Default True.
    :param filename: Where to look for the saved state dictionary. Default mnist_model.pt.
    :return: The reinstantiated model, loaded with saved weights.
    """
    model = MNISTModel(layer_sizes=layer_sizes, activation=activation, flatten=flatten)
    model.load_state_dict(torch.load(filename))
    return model


class MNISTModel(nn.Module):
    """
    A generic MLP with defaults designed for the MNIST classification problem: 784 input pixels and 10 classes.
    """

    def __init__(self, layer_sizes=(784, 100, 100, 10), activation=None, flatten=True):
        """
        Creates an MLP for the MNIST classification problem.
        :param layer_sizes: An iterable of the number of neurons in each layer. Defaults to (28x28, 100, 100, 10), ie,
                        2 hidden layers of 100 neurons each, 784 input neurons, and 10 output classes.
        :param activation: An activation module to use between Linear layers. If None, uses nn.ReLU()
        :param flatten: Whether to include an nn.Flatten() layer at the start of the net. Default True
        """
        super().__init__()
        activation = activation or nn.ReLU()

        modules = nn.ModuleList()
        if flatten:
            modules.append(nn.Flatten())
        for num_in, num_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            modules.append(nn.Linear(num_in, num_out))
            modules.append(activation)
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.net = nn.Sequential(modules)

    def forward(self, x):
        return self.net(x)


class MNISTTrainer:
    """
    A helper class designed to train an MNISTModel on the MNIST dataset.
    """

    def __init__(self, model, train_dataloader, validation_dataloader=None, criterion=None, optimizer=None,
                 device=None):
        """
        :param model: The model to train. Written for MNISTModel
        :param train_dataloader: The training dataloader. Written to use the MNIST dataset. Images should be tensors.
        :param validation_dataloader: If None, validation will not be performed; otherwise, it will be done every epoch.
        :param criterion: Loss function. If None, uses nn.CrossEntropyLoss()
        :param optimizer: Optimizer. If None, uses Adam with lr=3e-4
        :param device: The device to train on. If None, uses the cpu.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def validation_step(self, x, y):
        with torch.inference_mode():
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            return loss.item(), (y_hat.argmax(1) == y).float().mean().item()

    def train_one_epoch(self, epoch_number):
        loop = tqdm(self.train_dataloader, leave=False)
        loop.set_description(f"In Trainer, epoch {epoch_number + 1}")
        losses = []
        for i, (x, y) in enumerate(loop):
            loss = self.train_step(x, y)
            losses.append(loss)
            if not i % (len(self.train_dataloader) / 10):
                loop.set_postfix({"loss": loss})
        return losses, mean(losses)

    def validation_epoch(self):
        loop = tqdm(self.validation_dataloader, leave=False)
        loop.set_description(f"In Trainer, validation")
        losses = []
        accuracies = []
        for x, y in loop:
            loss, accuracy = self.validation_step(x, y)
            losses.append(loss)
            accuracies.append(accuracy)
            loop.set_postfix({"loss": loss})
        return mean(losses), mean(accuracies)

    def train_epochs(self, num_epochs):
        loop = trange(num_epochs)
        loop.set_description(f"In Trainer, training progress")
        all_losses = []
        mean_validation_losses = []
        mean_validation_accuracies = []
        for epoch_number in loop:
            losses, mean_loss = self.train_one_epoch(epoch_number)
            all_losses += losses
            if self.validation_dataloader is not None:
                mean_validation_loss, mean_validation_accuracy = self.validation_epoch()
                mean_validation_losses.append(mean_validation_loss)
                mean_validation_accuracies.append(mean_validation_accuracy)
                loop.set_postfix(
                    {"validation_accuracy": mean_validation_accuracy, "validation_loss": mean_validation_loss,
                     "train_loss": mean_loss})
            else:
                loop.set_postfix({"train_loss": mean_loss})
        return all_losses if self.validation_dataloader is None else all_losses, mean_validation_losses, mean_validation_accuracies


if __name__ == "__main__":
    # Example usage:

    # Train a default MNISTModel on the dataset for 15 epochs
    train_dataloader = DataLoader(train_dataset, batch_size=1000)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1000)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    trainer = MNISTTrainer(model=MNISTModel(), train_dataloader=train_dataloader,
                           validation_dataloader=validation_dataloader, device=device)

    epochs = 15
    all_losses, mean_validation_losses, mean_validation_accuracies = trainer.train_epochs(epochs)

    # Plot losses and accuracies
    plt.figure(1)
    plt.title("Training losses")
    plt.xlabel("Step")
    plt.ylabel("Cross entropy loss")
    plt.plot(range(epochs * len(train_dataloader)), all_losses)
    plt.figure(2)
    plt.title("Mean validation losses")
    plt.xlabel("After epoch")
    plt.ylabel("Cross entropy loss")
    plt.plot(range(1, epochs + 1), mean_validation_losses)
    plt.figure(3)
    plt.title("Mean validation accuracies")
    plt.xlabel("After epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(1, epochs + 1), mean_validation_accuracies)
    plt.show()

    # Verify qualitatively using a random image from the validation dataset
    model = trainer.model
    model.eval()
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")
    x, y = random.choice(validation_dataset)
    to_pil_image(x.reshape(1, 28, 28)).resize((280, 280)).show()
    with torch.inference_mode():
        print(f"For the shown image, true label is {y}; model predicted {model(x.to(device)).argmax(1).item()}")

    # Save
    torch.save(model.state_dict(), "mnist_model.pt")
