import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
import random
import pickle
from model import efficienet_pool, ensemble

device = ('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-5
epochs = 15

def train(model, images, optimizer, criterion):
    model.train()
    print('Training')
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
    epoch_acc = 100. * (train_running_correct / images.shape[0])
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
        train_epoch_loss, train_epoch_accuracy = train(model, images, optimizer, criterion)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    return model


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
    criterion = nn.CrossEntropyLoss() #TODO
    
    # Lists to keep track of losses and accuracies.
    train_loss = []
    train_accuracy = []
    # Training loop.
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = train(model, images, optimizer, criterion)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    return model


if __name__ == "__main__":
    images = pickle.load(open('images.obj','rb'))
    num_classes = 6
    model = train_efficient(images, num_classes)
    torch.save(model, "model.pth")
    model = train_ensemble(images)
    torch.save(model, "ensemble.pth")
    print("Training completed.")
    