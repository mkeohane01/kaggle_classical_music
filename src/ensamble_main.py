"""
Ensamble the predictions from the nueral network and xgboost models by loading saved models
To use this file, just run it and call ensamble_main()
"""
import pandas as pd
from nueral_network import NNet, predict_outputs 
import xgboost as xgb
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from process_data import load_and_process_data
from nn_main import setup_dataloaders, generate_roc
import numpy as np

data_dir = 'data/'
model_folder = 'models/'
def ensamble_main():
    """
    Load and pipeline the various models to create an ensamble prediction
    Prints the ROC AUC score for each model and the ensamble on validation data (subset of training)
    Saves the ensamble predictions to a csv
    """

    # load and process data
    train, test = load_and_process_data(is_print=False)

    # split the data into training and validation sets
    train_data, val_data = train_test_split(train, test_size=0.60, stratify=train['label'])

    ## split in to features and labels
    train_x  = train_data.drop(['account.id', 'label'], axis=1)
    train_y = train_data['label']

    val_x = val_data.drop(['account.id', 'label'], axis=1)
    val_y = val_data['label']

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_x)
    X_test = scaler.transform(test.drop(['account.id'], axis=1))
    X_val = scaler.transform(val_x)

    # grab feature names from train_x
    feature_names = train_x.columns.values.tolist()

    # setup dataloaders and xgboost DMatrix
    trainloader, testloader, valloader = setup_dataloaders(X_train, X_test, X_val, train_y, val_y)
    
    dval = xgb.DMatrix(X_val, label=val_y, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    # load in the pre-trained models
    # nueral network
    nn_model = NNet()
    nn_model.load_state_dict(torch.load(model_folder + 'opt_checkpoint.pth'))
    # xgboost
    xg_model = xgb.Booster()
    xg_model.load_model(model_folder + 'xgboost.json')

    # make predictions on test data
    nn_model.eval()
    nn_preds = predict_outputs(nn_model, testloader, device='cpu')
    xg_preds = xg_model.predict(dtest)

    # make predictions on validation data
    nn_model.eval()
    nn_val_preds = predict_outputs(nn_model, valloader, device='cpu')
    xg_val_preds = xg_model.predict(dval)

    # average the predicions
    ensamble_preds = (nn_preds.flatten() + xg_preds)/2
    ensamble_val_preds = (nn_val_preds.flatten() + xg_val_preds)/2

    # print the ROC AUC score for the ensamble and each model
    print(f"Ensamble ROC AUC: {generate_roc(ensamble_val_preds.flatten(), val_y.values.flatten(), label='Ensamble')}")
    print(f"NN ROC AUC: {generate_roc(nn_val_preds.flatten(), val_y.values.flatten(), label='NN')}")
    print(f"Boost ROC AUC: {generate_roc(xg_val_preds.flatten(), val_y.values.flatten(), label='Boost')}")

    # save predictions to csv
    ensamble_preds_df = pd.DataFrame({'ID': test['account.id'], 'Predicted': ensamble_preds})
    ensamble_preds_df.to_csv(data_dir + 'ensamble_preds.csv', index=False)


if __name__ == '__main__':
    ensamble_main()
