import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from tqdm.auto import tqdm
import random
from typing import Optional
import pickle
from model import efficienet_pool, ensemble


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-5
epochs = 15


def train(model, images, optimizer, criterion):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for data in tqdm(images):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / images.shape[0])
    return epoch_loss, epoch_acc


def train_efficient(images, num_classes):
    model = efficienet_pool(num_classes=num_classes)
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss = []
    train_accuracy = []
    # Training loop.
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = train(
            model, images, optimizer, criterion
        )
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        print(
            f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}"
        )
    return model


class QuadraticKappaLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = "cohen_kappa_loss",
        epsilon: Optional[float] = 1e-10,
    ):

        super(QuadraticKappaLoss, self).__init__()
        self.num_classes = num_classes
        self.y_pow = 2

        self.epsilon = epsilon

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).to(device)
        y_true = y[y_true]

        y_true = y_true.float()

        repeat_op = (
            torch.Tensor(list(range(num_classes)))
            .unsqueeze(1)
            .repeat((1, num_classes))
            .to(device)
        )
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred**self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(
            torch.reshape(hist_rater_a, [num_classes, 1]),
            torch.reshape(hist_rater_b, [1, num_classes]),
        )
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)


def train_ensemble(images):
    model1 = torch.load("model1.pth")
    model1.load(model1["state_dict"])

    model2 = torch.load("model2.pth")
    model2.load(model2["state_dict"])

    for params in model1.parameters():
        params.requires_grad = False

    for params in model2.parameters():
        params.requires_grad = False

    model = ensemble(6)
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function.
    # criterion = nn.CrossEntropyLoss() #TODO
    # Use quadratic Kappa loss.
    criterion = QuadraticKappaLoss(num_classes=6)

    # Lists to keep track of losses and accuracies.
    train_loss = []
    train_accuracy = []
    # Training loop.
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = train(
            model, images, optimizer, criterion
        )
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        print(
            f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}"
        )
    return model


if __name__ == "__main__":
    images = pickle.load(open("images.obj", "rb"))
    num_classes = 6
    model = train_efficient(images, num_classes)
    torch.save(model, "model.pth")
    model = train_ensemble(images)
    torch.save(model, "ensemble.pth")
    print("Training completed.")
