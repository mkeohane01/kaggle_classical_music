"""
Main file to train and test the xgboost model
Loads data, fits model, saves model, and saves predictions
To use this file, just run it and call xgboost_main()
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from process_data import load_and_process_data
import xgboost as xgb
import json

data_folder = 'data/'
model_folder = 'models/'

def xgboost_main():
    """
    Main function to train and test the xgboost model
    Loads data, fits model, saves model, and saves predictions
    """
    # load and process data
    train, test = load_and_process_data(is_print=False)
    
    # sum up the number of positive and negative labels
    pos = train['label'].sum()
    neg = len(train) - pos
    # calculate the scale_pos_weight parameter
    scale_pos_weight = neg/pos / 2

    # split into training and validation  sets
    train, val = train_test_split(train, test_size=0.15, stratify=train['label'])
    train_X = train.drop(['account.id', 'label'], axis=1)
    train_y = train['label']
    val_X = val.drop(['account.id', 'label'], axis=1)
    val_y = val['label']

    # scale the data using StandardScaler
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test.drop('account.id', axis=1))
    # print(f"train shape: {train_X_scaled.shape}")
    
    # get names of features from train_X
    feature_names = train_X.columns.values.tolist()

    # make in to xgb type
    dtrain = xgb.DMatrix(train_X_scaled, label=train_y, feature_names=feature_names)
    dval = xgb.DMatrix(val_X_scaled, label=val_y, feature_names=feature_names)
    dtest = xgb.DMatrix(test_scaled, feature_names=feature_names)

    # initialize parameters
    # lambda is L2 regularization, alpha is L1 regularization
    params = {'max_depth': 15, 'lambda': 100, 'objective': 'binary:logistic', 'eta': 0.35, 'scale_pos_weight': scale_pos_weight, 'num_parallel_tree': 8, 'max_delta_step': 5}
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    # train model
    bst = xgb.train(params, dtrain, 15, evallist, early_stopping_rounds=3)

    xgb.plot_importance(bst)
    # save feature importance to a file
    with open(model_folder+'feature_importance.json', 'w') as f:
        json.dump(bst.get_score(importance_type='weight'), f)
    plt.show()

    # save model and contents
    bst.save_model(model_folder + 'xgboost.json')

    # make predictions
    preds_val = bst.predict(dval)
    preds_test = bst.predict(dtest)
    
    # generate ROC curve
    fpr, tpr, thresholds = roc_curve(val_y, preds_val)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    # save test predictions in xgpredictions.csv
    preds_df = pd.DataFrame({'ID': test['account.id'], 'Predicted': preds_test})
    preds_df.to_csv(data_folder + 'xgpredictions.csv', index=False)
    

if __name__ == '__main__':
    xgboost_main()
