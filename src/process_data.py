"""
Handles all of the loading and processing of the csv data files using pandas
To use this file, import it and call load_and_process_data()

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

# depricated - no longer counting subscriptions like this
def count_subscriptions(subscriptions, test_data, train_data, is_print=False):
    """
    DEPERICATED - Count the number of subscriptions for each accountid in each data set.
    return: train_data, test_data: updated data with subs_count column
    """
    # for each account.id in train_df find how many times it appears in subscriptions_df account.id and store as a new column. make value 0 if it doesn't appear
    train_data['subs_count'] = train_data['account.id'].map(subscriptions['account.id'].value_counts())
    train_data['subs_count'].fillna(-99, inplace=True)

    # do same thing for test_df
    test_data['subs_count'] = test_data['ID'].map(subscriptions['account.id'].value_counts())
    test_data['subs_count'].fillna(-99, inplace=True)

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
    # create one hot encoded columns for each season
    subs_pivot = subs_df.pivot_table(index='account.id', columns='season', values='subscription_tier', aggfunc='max')
    subs_pivot = subs_pivot.fillna(-99)
    # rename each season columns to be subs_{season}.
    subs_pivot.columns = ['subs_' + str(col) for col in subs_pivot.columns]
    # remove all of the columns before 2009
    # subs_pivot = subs_pivot.drop(subs_pivot.columns[0:8], axis=1)
    if is_print:
        print(subs_pivot.head(2))
        print(f"df shape: {subs_pivot.shape}")
    return subs_pivot

# Depricated - Not using tickets_all data anymore
def encode_tickets(tickets_df, is_print=False):
    """
    DEPRICATED - Encode the seasons, set, and locations column in the tickets dataframe thorugh pivoting
    Return: tickets_df: updated dataframe with one hot encoded seasons, sets, and locations
    """
    # create one hot encoded columns for each location
    location_pivot = tickets_df.pivot_table(index='account.id', columns='location', values='no.seats', aggfunc='count')

    # create one hot encoded columns for each set
    set_pivot = tickets_df.pivot_table(index='account.id', columns='set', values='price.level', aggfunc='count')
    # rename each set columns to be set_1, set_2, etc.
    set_pivot.columns = ['set_' + str(col) for col in set_pivot.columns]

    # create one hot encoded columns for each season
    season_pivot = tickets_df.pivot_table(index='account.id', columns='season', values='no.seats', aggfunc='count')
    # rename each season columns to be ticket_{season}.
    season_pivot.columns = ['ticket_' + str(col) for col in season_pivot.columns]

    # merge the three dataframes
    tickets_pivot_temp = pd.merge(location_pivot, set_pivot, on='account.id')
    tickets_pivot = pd.merge(tickets_pivot_temp, season_pivot, on='account.id')

    # fill NaNs 
    tickets_pivot.fillna(-99, inplace=True)
    if is_print:
        print(tickets_pivot.head(2))
        print(f"df shape: {tickets_pivot.shape}")
        print(tickets_pivot.dtypes)
    return tickets_pivot

def clean_account_data(accounts, is_print=False):
    """
    Processes the account dataframe to only include the columns we can use in forms that we can use
    return: accounts_nums
    """
    # keep the account columns we can work with
    account_nums = accounts[['account.id', 'billing.zip.code','amount.donated.2013', 'amount.donated.lifetime','no.donations.lifetime']]
    # convert zip code to float skipping invalid data
    account_nums['billing.zip.code'] = pd.to_numeric(account_nums['billing.zip.code'], errors='coerce')
    account_nums.dropna(inplace=True)
    
    if is_print:
        print(account_nums.head())
        print(f"accounts shape: {account_nums.shape}")
        print(account_nums.dtypes)

    return account_nums

def merge_data(accounts, subsriptions, is_print=False):
    """
    Merge the accounts and subscriptions dataframes on account.id
    -- No longer merging tickets_all, but can add back in if needed
    return: merged dataframe
    """
    # merge accounts and subscriptions where subscriptions has account.ids as index and accounts has account.ids as column
    merged = pd.merge(accounts, subsriptions, left_on='account.id', right_index=True)
    # merge temp_merged and tickets_all where tickets_all has account.ids as index and merged has account.ids as column
    # merged = pd.merge(temp_merged, tickets_all, left_on='account.id', right_index=True)

    if is_print:
        print(merged.head(2))
        print(f"merged shape: {merged.shape}")
        print(merged.dtypes)
    return merged

def merge_data_with_test_and_train(test, train, merge_df, is_print=False):
    """
    Merge the test and train dataframes with the total merged dataframes on account.id preserving shape
    return: merged_train, merged_test
    """
    merged_train = pd.merge(train, merge_df, on='account.id', how='left')
    test.rename(columns={'ID': 'account.id'}, inplace=True)
    merged_test = pd.merge(test, merge_df, on='account.id', how='left')
    # replace all NaNs with 0
    merged_train.fillna(-99, inplace=True)
    merged_test.fillna(-99, inplace=True)
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
    Load and process the data from the csv and return the train and test dataframes with all the wanted features
    Returns: train, test
    """
    # read in data
    tickets_all_df, account_df, subscriptions_df = load_csvs(is_print=is_print)
    train_df, test_df = load_test_and_train(is_print=is_print)

    # process data
    accounts_nums = clean_account_data(account_df, is_print=is_print)
    subs_piv = pivot_seasons_subs(subscriptions_df, is_print=is_print)
    # tickets_all = encode_tickets(tickets_all_df, is_print=is_print)

    # merge data
    merged = merge_data(accounts_nums, subs_piv, is_print=is_print)
    train, test = merge_data_with_test_and_train(test_df, train_df, merged, is_print=is_print)

    return train, test

if __name__ == "__main__":
    load_and_process_data(is_print=True)