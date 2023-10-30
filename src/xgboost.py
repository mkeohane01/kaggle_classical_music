import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from process_data import load_and_process_data
import xgboost as xgb

data_folder = 'data/'
model_folder = 'models/'

def xgboost_main(is_print=False):
    """
    Main function
    """
    # load and process data
    train, test = load_and_process_data(is_print=is_print)
    
    # split into training and validation  sets
    train_X, val_X, train_y, val_y = train_test_split(train.drop('account.id', axis=1), train['label'], test_size=0.15, stratify=train['label'])

    # make in to xgb type
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dval = xgb.DMatrix(val_X, label=val_y)
    dtest = xgb.DMatrix(test.drop['account.id'])



    # initialize params
    params = {
        'max_depth': 2,
        'learning_rate': 1,
        'objective': 'binary:logistic',
        'n_estimators': 2,
        'eval_metric': 'auc'
    }
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    # create initialize xgboost model
    bst = xgb.XGBClassifier()

    # fit model
    bst.fit(train_X, train_y)


    # make predictions
    preds_val = bst.predict(val_X)
    preds_test = bst.predict(test)

    # generate ROC curve
    fpr, tpr, thresholds = roc_curve(val_y, preds_val)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    # save test predictions in xgpredictions.csv
    preds_df = pd.DataFrame({'ID': test['account.id'], 'Predicted': preds_test})
    preds_df.to_csv(data_folder + 'xgpredictions.csv', index=False)

if __name__ == '__main__':
    xgboost_main(is_print=True)
