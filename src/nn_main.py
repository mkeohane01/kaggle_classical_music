from process_data import load_and_process_data
from nueral_network import train_network, NNet, predict_outputs
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

data_folder = 'data/'
model_folder = 'models/'

def setup_dataloaders(test_df, train_df, val_percent=0.15, batch_size=32, is_print=False):
    """
    Input test and train data then split into train and validation sets and return dataloaders for each
    return: trainloader, testloader, valloader
    """
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(train_df, test_size=val_percent, stratify=train_df['label'])

    ## split in to features and labels
    train_x  = train_data.drop(['account.id', 'label'], axis=1)
    train_y = train_data['label']

    val_x = val_data.drop(['account.id', 'label'], axis=1)
    val_y = val_data['label']

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_x)
    X_test = scaler.transform(test_df.drop(['account.id'], axis=1))
    X_val = scaler.transform(val_x)

    if is_print:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {train_y.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {val_y.shape}")
        print(f"X_test shape: {X_test.shape}")
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).float(), 
                            torch.from_numpy(np.array(train_y)).float().view(-1,1))
    testset = TensorDataset(torch.from_numpy(np.array(X_test)).float())

    valset = TensorDataset(torch.from_numpy(np.array(X_val)).float(), 
                            torch.from_numpy(np.array(val_y)).float().view(-1,1))

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, valloader, val_y

def generate_roc(preds, labels):
    """
    Generate ROC curve
    """
    # calculate true positive rate and false positive rate along with roc
    fpr, tpr, thresholds = roc_curve(labels, preds)

    # calculate area under the curve
    roc_auc = auc(fpr, tpr)

    # plot the ROC curve
    # plot ROC curve
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    return roc_auc

def plot_loss(train_losses, val_losses):
    """
    Plot training and validation loss
    """
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()



def main():
    """
    Main function
    """
    
    # load and process data
    train_data, test_data = load_and_process_data()

    # setup dataloaders
    trainloader, testloader, valloader, val_y = setup_dataloaders(test_data, train_data)

    # initialize model
    nnet = NNet()

    # Train model
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_network(trainloader, valloader, nnet, lr=0.000125, epochs=60, is_print=True)

    # Test model
    preds = predict_outputs(model, testloader, device='cpu')

    # save predictions to csv
    preds_df = pd.DataFrame({'ID': test_data['account.id'], 'Predicted': preds.flatten()})
    preds_df.to_csv(data_folder + 'nn_preds.csv', index=False)

    # Save model
    torch.save(model.state_dict(), model_folder + 'model_3.pth')

    # test model using validation set
    preds = predict_outputs(model, valloader, device='cpu')
    print(generate_roc(preds.flatten(), val_y.values.flatten()))

    # plot loss
    plot_loss(train_losses, val_losses)

    # load opt checkpoint
    model.load_state_dict(torch.load(model_folder + 'opt_checkpoint.pth'))

    # test model using validation set
    val_preds = predict_outputs(model, valloader, device='cpu')
    print(generate_roc(val_preds.flatten(), val_y.values.flatten()))

    # save predictions of opt model to csv
    test_preds = predict_outputs(model, testloader, device='cpu')
    preds_df = pd.DataFrame({'ID': test_data['account.id'], 'Predicted': test_preds.flatten()})
    preds_df.to_csv(data_folder + 'nn_opt_preds.csv', index=False)

    # plot loss
    plot_loss(train_losses, val_losses)

if __name__ == "__main__":
    main()

