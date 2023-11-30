import time
import math
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import numpy as np


def find_device():
    """
    This always returns 'cpu' for now, unless we need to utilize GPU processor.
    """
    # is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    # if is_cuda:
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = 'cpu'
    return device


def get_predictions(model, device, data_loader, sig_func):
    y_true = []
    y_prob = []

    for inputs, labels in data_loader.loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        output = sig_func(output)
        y_prob.extend(output.cpu().data.numpy())
        y_true.extend(labels.cpu().data.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    return y_true, y_prob


def calculate_auc(model, device, data_loader, sig_func):
    y_true, y_prob = get_predictions(model, device, data_loader, sig_func)
    auc = roc_auc_score(y_true, y_prob)
    return auc


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %fs' % (m, s)


def train_batch(model, inputs, labels, device, criterion, optimizer):
    # Train model
    model.train()
    # 1. Zero accumulated gradients
    model.zero_grad()
    # 2. Move data to device
    inputs, labels = inputs.to(device), labels.to(device)
    # 3. Run forward pass through model
    output = model(inputs)
    # 4.  Calculate the loss and update gradients.
    loss = criterion(output, labels.double())
    loss.backward()
    # 5. Update parameters
    optimizer.step()


def validate_nn(model, device, train_loader, val_loader, sigmoid):
    model.eval()
    train_auc = calculate_auc(model=model, device=device, data_loader=train_loader, sig_func=sigmoid)
    val_auc = calculate_auc(model=model, device=device, data_loader=val_loader, sig_func=sigmoid)
    return train_auc, val_auc
