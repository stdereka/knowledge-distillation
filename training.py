import torch
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as f


def fit_epoch(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    processed_data = 0

    ground = []
    predicted = []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        for i in range(len(labels)):
            labels[i] = labels[i].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, *labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)

        ground.append(labels[0].cpu())
        predicted.append(preds.cpu())

        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)

    ground = np.hstack(ground)
    predicted = np.hstack(predicted)

    train_loss = running_loss / processed_data
    train_acc = accuracy_score(ground, predicted)

    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    processed_size = 0

    ground = []
    predicted = []
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        for i in range(len(labels)):
            labels[i] = labels[i].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, *labels)
            preds = torch.argmax(outputs, 1)

        ground.append(labels[0].cpu())
        predicted.append(preds.cpu())

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)

    ground = np.hstack(ground)
    predicted = np.hstack(predicted)

    val_loss = running_loss / processed_size
    val_acc = accuracy_score(ground, predicted)

    return val_loss, val_acc


def train(train_dataset, val_dataset, model, epochs, batch_size, device, opt, criterion):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    history = []
    log_template = "Epoch: {ep}, train_loss: {t_loss:0.4f}, val_loss: {v_loss:0.4f}, " \
                   "train_acc: {t_acc:0.4f}, val_acc: {v_acc:0.4f}"

    for epoch in range(epochs):
        train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt, device)

        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history.append([train_loss, train_acc, val_loss, val_acc])

        print(log_template.format(ep=epoch + 1, t_loss=train_loss, v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

    return history


def predict(model, test_loader, device, logit=False):
    with torch.no_grad():
        logits = []

        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    if logit:
        return torch.cat(logits).numpy()
    else:
        return f.softmax(torch.cat(logits), dim=-1).numpy()


class DistillationLoss:
    def __init__(self, temperature=10, alpha=0.1):
        self.temperature = temperature
        self.alpha = alpha

    def __call__(self, y, labels, teacher_labels):
        cross_entropy_hard = f.cross_entropy(y, labels)
        cross_entropy_soft = nn.KLDivLoss()(f.log_softmax(y/self.temperature, dim=-1),
                                            f.softmax(teacher_labels/self.temperature, dim=-1))
        loss = self.alpha * cross_entropy_hard + (1 - self.alpha) * (self.temperature ** 2) * cross_entropy_soft
        return loss
