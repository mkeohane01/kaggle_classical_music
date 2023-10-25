import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

# define the network
class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(24, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = torch.sigmoid(self.out(x))
        return x

nnet = NNet()

def train_network(train_data, val_data, model, lr=0.001, epochs=200, is_print=False):
    """
    Train the network
    """
    # define the loss
    criterion = nn.BCELoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # define lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for e in range(epochs):
        running_loss = 0
        running_accuracy = 0
        for features, labels in train_data:
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            output = model(features)
            # compute loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            # update running loss
            running_loss += loss.item()
            # update running accuracy
            running_accuracy += (output.round() == labels).float().mean().item()
        else:
            # compute validation loss and accuracy
            # compute TPR aka recall as well
            num_real_positives = 0
            val_recall = 0
            val_recall_01 = 0
            val_recall_05 = 0
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for features, labels in val_data:
                    output = model(features)
                    val_loss += criterion(output, labels).item()
                    val_accuracy += (output.round() == labels).float().mean().item()
                    val_recall += ((output.round() == labels) & (labels == 1)).float().sum().item()
                    # calcuale recall where you threshold with 0.1
                    val_recall_01 += (((output > 0.1).float().round() == labels) & (labels==1)).float().sum().item()
                    val_recall_05 += (((output > 0.05).float().round() == labels) & (labels==1)).float().sum().item()
                    num_real_positives += (labels == 1).float().sum().item()
            # store metrics
            train_losses.append(running_loss/len(train_data))
            val_losses.append(val_loss/len(val_data))
            train_accuracies.append(running_accuracy/len(train_data))
            val_accuracies.append(val_accuracy/len(val_data))
        
            # print metrics
            if is_print:
                print(f"Epoch: {e+1}/{epochs}.. ",
                    f"T Loss: {running_loss/len(train_data):.3f}.. ",
                    f"T Acc: {running_accuracy/len(train_data):.3f}.. ",
                    f"V Loss: {val_loss/len(val_data):.3f}.. ",
                    f"V Acc: {val_accuracy/len(val_data):.3f}",
                    f"V Recall 0.05: {val_recall_05/num_real_positives:.3f}",
                    f"V Recall 0.1: {val_recall_01/num_real_positives:.3f}")
                
    return model, train_losses, val_losses, train_accuracies, val_accuracies

# generate predictions using test data 
def predict_outputs(model,testloader,device):
    """
    generate predictions using test data
    """

    model.eval() # Set the model to evaluation mode
    predictions = []
    for i, data in enumerate(testloader):
        inputs = data[0].to(device)
        outputs = model.forward(inputs)
        predictions.append(outputs.detach().numpy())
    return np.concatenate(predictions)