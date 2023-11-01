"""
This file contains the Nueral Net class as well as 
functions to train and test the neural network
To use import the NNet class, train_network, and predict_output functions
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn.metrics import roc_curve, auc
import numpy as np

class NNet(nn.Module):
    """
    Class containing the neural network model
    3 hidden layers with 50, 50, and 25 nodes respectively
    Inputs inlcude the number of features and dropout rate
    """
    def __init__(self, n_feats=25, dropout=0.0):
        super().__init__()
        self.hidden1 = nn.Linear(n_feats, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.hidden3 = nn.Linear(50, 25)
        self.out = nn.Linear(25, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = F.relu(self.hidden3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.out(x))
        return x

def train_network(train_data, val_data, model, lr=0.001, epochs=200, is_print=False):
    """
    Train the nueral network while printing metrics and saving the model
    -- saves the model when validation loss is minimized
    Inputs the training and validation data along with the model
    Returns: The trained model and lists of training and validation losses
    """
    # define the loss
    criterion = nn.BCELoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # define lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_loss = np.Inf

    # train the network over the given number of epochs
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
            val_loss = 0
            val_accuracy = 0
            num_truepos = 0
            num_realpos = 0
            with torch.no_grad():
                for features, labels in val_data:
                    val_output = model(features)
                    val_loss += criterion(val_output, labels).item()
                    val_accuracy += (val_output.round() == labels).float().mean().item()
                    # calculate true positives and real positives
                    num_truepos += ((val_output.round() == labels) & (labels == 1)).float().sum().item()
                    num_realpos += (labels == 1).float().sum().item()
                    
            # store metrics
            train_losses.append(running_loss/len(train_data))
            val_losses.append(val_loss/len(val_data))
            train_accuracies.append(running_accuracy/len(train_data))
            val_accuracies.append(val_accuracy/len(val_data))
            val_recall = num_truepos/num_realpos
        
            # print metrics
            if is_print:
                print(f"Epoch: {e+1}/{epochs}.. ",
                    f"T Loss: {running_loss/len(train_data):.3f}.. ",
                    f"V Loss: {val_loss/len(val_data):.3f}.. ",
                    f"V Recall: {val_recall:.3f}.. ")
            
            # save when validation loss decreases below minimum
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"Saving model at epoch {e+1}")
                torch.save(model.state_dict(), 'models/' + 'opt_checkpoint.pth')
                    
                
    return model, train_losses, val_losses

def predict_outputs(model,testloader,device):
    """
    Generate predictions using test data and the given model
    Returns: numpy array of predictions
    """

    model.eval() # Set the model to evaluation mode
    predictions = []
    for i, data in enumerate(testloader):
        inputs = data[0].to(device)
        outputs = model.forward(inputs)
        predictions.append(outputs.detach().numpy())
    return np.concatenate(predictions)