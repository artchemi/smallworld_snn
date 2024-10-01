import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

from tqdm import tqdm

# Network Architecture
num_inputs = 100
num_hidden = 50
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

dtype = torch.float
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load and preprocess data
        data_sample = self.data[idx]
        label_sample = self.labels[idx]

        if self.transform:
            data_sample = self.transform(data_sample)

        return data_sample, label_sample


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    

def evaluate(model, loader) -> float:
    """_summary_

    Args:
        model (_type_): _description_
        loader (_type_): _description_

    Returns:
        float: _description_
    """

    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
    
            # forward pass
            test_spk, _ = model(data.view(data.size(0), -1))

            # calculate total accuracy
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total

    

def main():
    # a = torch.Tensor([0.5, 0.3, 0.4, 1, 0.5])
    # test_series = spikegen.rate(a, num_steps=25)

    net = Net().to(device)
    
    # print(test_series)
    # print(model(test_series))

    path = 'data_29_09_2024_005040'
    # path = 'data_25_09_2024_143505'

    # --- custom train dataset --- 
    X_train = []
    labels_train = []

    for i in range(0, 10):
        with open(path + f'/epoch_1/rates_{i}_label.npy', 'rb') as file:
            data = np.load(file)
        
        for pattern in data:
            X_train.append(np.reshape(pattern, 100))
            labels_train.append(i)

    X_train = torch.tensor(np.array(X_train))
    labels_train = torch.tensor(np.array(labels_train), dtype=torch.long)


    train_dataset = CustomDataset(data=X_train, labels=labels_train)

    # --- custom test dataset ---

    X_test = []
    labels_test = []

    for i in range(0, 10):
        with open(path + f'/epoch_1/rates_test/rates_{i}_label.npy', 'rb') as file:
            data = np.load(file)
        
        for pattern in data:
            X_test.append(np.reshape(pattern, 100))
            labels_test.append(i)

    X_test = torch.tensor(np.array(X_test))
    labels_test = torch.tensor(np.array(labels_test), dtype=torch.long)

    test_dataset = CustomDataset(data=X_test, labels=labels_test)

    # --- normalization ---
    train_dataset.data = train_dataset.data / 42
    test_dataset.data = test_dataset.data / 42

    # --- dataloaders --- 
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def print_batch_accuracy(data, targets, train=False):
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
            print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

    def train_printer(
        data, targets, epoch,
        counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        print_batch_accuracy(data, targets, train=True)
        print_batch_accuracy(test_data, test_targets, train=False)
        print("\n")

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))

    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)

    spk_rec, mem_rec = net(data.view(batch_size, -1))
    print(mem_rec.size())

    loss_val = torch.zeros((1), dtype=dtype, device=device)

    # sum loss at every step
    for step in range(num_steps):
        loss_val += loss(mem_rec[step], targets)

    print(f"Training loss: {loss_val.item():.3f}")
    print_batch_accuracy(data, targets, train=True)

    # clear previously stored gradients
    optimizer.zero_grad()

    # calculate the gradients
    loss_val.backward()

    # weight update
    optimizer.step()

    # calculate new network outputs using the same data
    spk_rec, mem_rec = net(data.view(batch_size, -1))

    # initialize the total loss value
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    # sum loss at every step
    for step in range(num_steps):
        loss_val += loss(mem_rec[step], targets)

    print(f"Training loss: {loss_val.item():.3f}")
    print_batch_accuracy(data, targets, train=True)


    # --- TRAINING CYCLE ---

    num_epochs = 50
    loss_hist = []
    test_loss_hist = []
    counter = 0

    # Outer training loop
    for epoch in tqdm(range(num_epochs)):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(
                        data, targets, epoch,
                        counter, iter_counter,
                        loss_hist, test_loss_hist,
                        test_data, test_targets)
                counter += 1
                iter_counter +=1
        
        print(f'Train accuracy after epoch #{epoch} --- {evaluate(net, train_loader) * 100}%')
        print(f'Train accuracy after epoch #{epoch} --- {evaluate(net, test_loader) * 100}%')



    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
    

