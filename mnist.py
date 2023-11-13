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


def load_MNIST_model(hidden_size=100, filename="mnist_model.pt"):
    model = MNISTModel(hidden_size=hidden_size)
    model.load_state_dict(torch.load(filename))
    return model


class MNISTModel(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(28 * 28),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.net(x)


class MNISTTrainer:
    def __init__(self, model, train_dataloader, validation_dataloader=None, criterion=None, optimizer=None,
                 device=None):
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
    # Train
    train_dataloader = DataLoader(train_dataset, batch_size=1000)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1000)
    device = torch.device("mps")  # Could be cuda, mps, ...
    trainer = MNISTTrainer(model=MNISTModel(), train_dataloader=train_dataloader,
                           validation_dataloader=validation_dataloader, device=device)

    epochs = 15
    all_losses, mean_validation_losses, mean_validation_accuracies = trainer.train_epochs(epochs)

    # Plot
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

    # Verify qualitatively
    model = trainer.model
    model.eval()
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")
    x, y = random.choice(validation_dataset)
    to_pil_image(x.reshape(1, 28, 28)).resize((280, 280)).show()
    with torch.inference_mode():
        print(f"For the shown image, true label is {y}; model predicted {model(x.to(device)).argmax(1).item()}")

    # Save
    torch.save(model.state_dict(), "mnist_model.pt")
