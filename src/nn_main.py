from process_data import load_and_process_data
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def setup_dataloaders(test_data, train_data, val_percent, batch_size=32):
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).float(), 
                            torch.from_numpy(np.array(y_train)).float().view(-1,1))
    testset = TensorDataset(torch.from_numpy(np.array(X_test)).float())

    valset = TensorDataset(torch.from_numpy(np.array(X_val)).float(), 
                            torch.from_numpy(np.array(y_val)).long())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, valloader


def main():
    """
    Main function
    """
    
    # load and process data
    train_data, test_data = load_and_process_data()


if __name__ == "__main__":
    main()

