"""
Handles all of the loading and processing of the csv data files using pandas

"""


import pandas as pd

data_dir = 'data/'

def load_csvs(dir = data_dir, is_print=False):
    """
    Load the tickets_all, account, and subscriptions data from the given directory.
    return: tickets_all, account, subscriptions
    """
    tickets_all = pd.read_csv(dir + 'tickets_all.csv', encoding_errors='ignore')
    account = pd.read_csv(dir + 'account.csv', encoding_errors='ignore')
    subscriptions = pd.read_csv(dir + 'subscriptions.csv')
    if is_print:
        print(f"tickets_all shape: {tickets_all.shape}")
        print(tickets_all.columns)
        print(f"account shape: {account.shape}")
        print(account.columns)
        print(f"subscriptions shape: {subscriptions.shape}")
        print(subscriptions.columns)
    return tickets_all, account, subscriptions


def load_test_and_train(dir = data_dir, is_print=False):
    """
    Load the train and test data from the given directory.
    return: train, test
    """
    train = pd.read_csv(dir + 'train.csv')
    test = pd.read_csv(dir + 'test.csv')
    if is_print:
        print(f"train shape: {train.shape}")
        print(train.head(2))
        print(f"test shape: {test.shape}")
        print(test.head(2))
        
    return train, test


def count_subscriptions(subscriptions, test_data, train_data, is_print=False):
    """
    Count the number of subscriptions for each accountid in each data set.
    return: train_data, test_data: updated data with subs_count column
    """
    # for each account.id in train_df find how many times it appears in subscriptions_df account.id and store as a new column. make value 0 if it doesn't appear
    train_data['subs_count'] = train_data['account.id'].map(subscriptions['account.id'].value_counts())
    train_data['subs_count'].fillna(0, inplace=True)

    # do same thing for test_df
    test_data['subs_count'] = test_data['ID'].map(subscriptions['account.id'].value_counts())
    test_data['subs_count'].fillna(0, inplace=True)

    if is_print:
        print(train_data.head(2))
        print(f"train shape: {train_data.shape}")
        print(test_data.head(2))
        print(f"test shape: {test_data.shape}")
    
    return train_data, test_data

def pivot_seasons_subs(subs_df, is_print=False):
    """
    encode the seasons column in the subscriptions dataframe thorugh pivoting
    Return: subs_df: updated dataframe with one hot encoded seasons
    """
    subs_pivot = subs_df.pivot_table(index='account.id', columns='season', values='subscription_tier', aggfunc='max')
    subs_pivot = subs_pivot.fillna(0)

    if is_print:
        print(subs_pivot.head(2))
        print(f"df shape: {subs_pivot.shape}")
    return subs_pivot

def clean_data(accounts, tickets_all, is_print=False):
    """
    Processes the dataframes to only include the columns we can use in forms that we can use
    return: accounts_nums, tickets_all_nums
    """
    # start with tickets_all
    tickets_all_nums = tickets_all[['account.id', 'price.level', 'no.seats', 'season', 'set']]
    # convert price.level to float skipping invalid data
    tickets_all_nums['price.level'] = pd.to_numeric(tickets_all['price.level'], errors='coerce')
    # convert season to float (take fist number) eg 2016-2017 -> 2016
    tickets_all_nums['season'] = tickets_all['season'].str[:4].astype(float)
    # count how many times account.id appears in tickets_all and store as a new column then remove account.id duplicates
    tickets_all_nums['no.tickets'] = tickets_all_nums['account.id'].map(tickets_all_nums['account.id'].value_counts())
    tickets_all_nums.drop_duplicates(subset='account.id', inplace=True)
    tickets_all_nums.dropna(inplace=True)

    if is_print:
        print(tickets_all_nums.head())
        print(f"tickets_all shape: {tickets_all_nums.shape}")
        print(tickets_all_nums.dtypes)

    # now do accounts
    account_nums = accounts[['account.id', 'amount.donated.2013', 'amount.donated.lifetime','no.donations.lifetime']]
    account_nums.dropna(inplace=True)
    if is_print:
        print(account_nums.head())
        print(f"accounts shape: {account_nums.shape}")
        print(account_nums.dtypes)

    return account_nums, tickets_all_nums

def merge_data(accounts, tickets_all, subsriptions, is_print=False):
    """
    Merge the accounts and tickets_all dataframes on account.id
    return: merged
    """
    # temp_merged = pd.merge(accounts, tickets_all, on='account.id')
    # merge accounts and subscriptions where subscriptions has account.ids as index and accounts has account.ids as column
    merged = pd.merge(accounts, subsriptions, left_on='account.id', right_index=True)
    if is_print:
        print(merged.head(2))
        print(f"merged shape: {merged.shape}")
        print(merged.dtypes)
    return merged

def merge_data_with_test_and_train(test, train, merge_df, is_print=False):
    """
    Merge the test and train dataframes with the merged accounts and tickets_all dataframes on account.id preserving shape
    return: merged_train, merged_test
    """
    merged_train = pd.merge(train, merge_df, on='account.id', how='left')
    test.rename(columns={'ID': 'account.id'}, inplace=True)
    merged_test = pd.merge(test, merge_df, on='account.id', how='left')
    # replace all NaNs with 0
    merged_train.fillna(0, inplace=True)
    merged_test.fillna(0, inplace=True)
    if is_print:
        print(merged_train.head())
        print(f"train shape: {merged_train.shape}")
        print(merged_train.dtypes)
        print(merged_test.head())
        print(f"test shape: {merged_test.shape}")
        print(merged_test.dtypes)
    return merged_train, merged_test

def load_and_process_data(is_print=False):
    """
    Load and process the data from the csv and return the train and test dataframes
    """
    # read in data
    tickets_all_df, account_df, subscriptions_df = load_csvs(is_print=is_print)
    train_df, test_df = load_test_and_train(is_print=is_print)

    # process data
    accounts_nums, tickets_all_nums = clean_data(account_df, tickets_all_df, is_print=is_print)
    subs_piv = pivot_seasons_subs(subscriptions_df, is_print=is_print)
    # count subscriptions
    # train, test = count_subscriptions(subscriptions_df, test_df, train_df, is_print=is_print)
    # merge data
    merged = merge_data(accounts_nums, tickets_all_nums, subs_piv, is_print=is_print)
    train, test = merge_data_with_test_and_train(test_df, train_df, merged, is_print=is_print)
    return train, test

if __name__ == "__main__":
    load_and_process_data(is_print=True)