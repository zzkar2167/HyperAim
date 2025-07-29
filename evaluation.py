import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import random


class End2EndNN(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(End2EndNN, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hyperedge_index, d_list, edge_weight=None):
        z = self.encoder.get_embedding(hyperedge_index, x, edge_weight, d_list)
        return self.classifier(z)

class MLP_HENN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MLP_HENN, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def train_node_classification_finetune(model, data, device, d_list, train_mask, val_mask, test_mask, lr=0.005,
                                       weight_decay=5e-4, epochs=200, seed=42, dataset_type=None, dataset_name=None):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    train_idx = torch.where(train_mask)[0].to(device)
    val_idx = torch.where(val_mask)[0].to(device)
    test_idx = torch.where(test_mask)[0].to(device)
    labels = data.labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc = 0
    best_model_state = None
    early_stop_counter = 0
    max_patience = 100

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        out = model(
            x=data.features,
            hyperedge_index=data.hyperedge_index,
            d_list=d_list,
            edge_weight=None
        )
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            out = model(
                x=data.features,
                hyperedge_index=data.hyperedge_index,
                d_list=d_list,
                edge_weight=None
            )
            val_pred = out[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

            test_pred = out[test_idx].argmax(dim=1)
            test_acc = (test_pred == labels[test_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best Val Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}")
        return best_val_acc, best_test_acc
    else:
        print("Skipped due to no valid model state")
        return 0.0, 0.0

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_node_classification_linear(embeddings, labels, train_mask, val_mask, test_mask, device, epochs=200, lr=1e-3,
                                     w_decay=1e-6):

    classifier = MLP(in_dim=embeddings.shape[1], hidden_dim=128, n_class=labels.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=w_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_param = None

    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(embeddings)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            classifier.eval()
            with torch.no_grad():
                logits = classifier(embeddings)
                val_acc = compute_accuracy(logits[val_mask], labels[val_mask])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_param = copy.deepcopy(classifier.state_dict())

    classifier.load_state_dict(best_param)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings)
        test_acc = compute_accuracy(logits[test_mask], labels[test_mask])

    return best_val_acc, test_acc

