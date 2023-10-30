import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from process_data import load_and_process_data
import xgboost as xgb

data_folder = 'data/'
model_folder = 'models/'

def xgboost_main():
    """
    Main function
    """
    # load and process data
    train, test = load_and_process_data(is_print=False)
    
    # split into training and validation  sets
    train, val = train_test_split(train, test_size=0.15, stratify=train['label'])
    train_X = train.drop(['account.id', 'label'], axis=1)
    train_y = train['label']
    val_X = val.drop(['account.id', 'label'], axis=1)
    val_y = val['label']

    # make in to xgb type
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dval = xgb.DMatrix(val_X, label=val_y)
    dtest = xgb.DMatrix(test.drop('account.id', axis=1))

    # initialize params
    params = {'max_depth': 22, 'eta': 1, 'objective': 'binary:logistic'}
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    # train model
    bst = xgb.train(params, dtrain, 20, evallist)

    xgb.plot_importance(bst)
    plt.show()

    # save model and contents
    bst.save_model(model_folder + 'xgboost.model')

    # make predictions
    preds_val = bst.predict(dval)
    preds_test = bst.predict(dtest)
    
    # generate ROC curve
    # fpr, tpr, thresholds = roc_curve(preds_val, dval.get_label())
    # roc_auc = auc(fpr, tpr)
    # print(f"ROC AUC: {roc_auc}")
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    # save test predictions in xgpredictions.csv
    preds_df = pd.DataFrame({'ID': test['account.id'], 'Predicted': preds_test})
    preds_df.to_csv(data_folder + 'xgpredictions.csv', index=False)
    

if __name__ == '__main__':
    xgboost_main()
