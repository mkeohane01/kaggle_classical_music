# Kaggle Classical Music Competition

This repository contains the code for the [Kaggle Classical Music Meets Classical ML Fall 2023 competition](https://www.kaggle.com/competitions/classical-music-meets-classical-ml-fall-2023) hosted by Duke AIPI 520.

The goal of the project is to predict which users will subscribe to season tickets for the 2013-14 season.

## Repo Structure

Data processing functions
- src/process_data.py

Nueral Network scripts
- src/nueral_network.py
- src/nn_main.py - executable

XGBoost scripts
- src/boost_main.py - executable

Ensambling scripts
- src/ensamble_main.py - executable

Input and output data
- data/

Saved Models and Model Info
- models/

Notebooks for scratch work
- notebooks/

Any saved figures
- figs/

## Data

The data used in this project comes from the Kaggle competition. It consists of various user data including account, subscription, ticket, and concert information.

### Choosing Datasets

I combed through the data to choose features that would be useable by an ML classifier. I need values encoded numerically in order to use them in the predictor. 

I started with just the numerical account information but wanted to encompass previous ticket and account purchases. For both the ticket_all and subscriptions data, I grouped and pivoted the columns to count how many times each user (account.id) either bought tickets or subscribed each season respectively.

I found through validation that while the previous subscriptions helped the model performance, no matter how I encoded the ticket data, it would alway decrease model performance.

The datasets I ended up using for the best performing model were
 - account.csv
 - subscriptions.csv

### Feature Extraction

In order to extract the wanted features I used the following steps. I found these features by using a variety of options and validating performance:

1. **Clean Accounts**: Get 'billing.zip.code','amount.donated.2013', 'amount.donated.lifetime', and 'no.donations.lifetime' from the accounts df and then convert billing.zip to numerical.

2. **Pivot Subscriptions**: In order to count which season each account previously bought tickets, I used a pivot table to get the subscription tier for each year the account.id subscribed.
- ```subs_pivot = subs_df.pivot_table(index='account.id', columns='season', values='subscription_tier', aggfunc='max')```
3. **Merge / Fill NANs**: Finally I merged the train/test sets with these two dataframes. It is essential that the features match exactly for each. I also fill NANs of the dfs with values to contrast and therefore decided on -99.

### Data Prepping
In order to be trained well, the data also needs to be processed accordingly. The main processing tool I implemented was to scale the data using sklearn StandardScaler in order to help training by normalizing the input data.

For the Nueral Network I also used dataloaders to be able to train on minibatches of size 32.

All in all I ended up with 25 features to use.
## Models

I ensambled two seperate models for classifcation in this project:

1. **Nueral Network**: The first model that I worked on was to build a simple Nueral Network using pytorch. 
- Design: I used a linear model with 3 fully connected layers of size 50, 50, and 25 respectively. I also added dropout (15%) at a variable rate to help prevent overfitting.
- Training: I used an Adam optimizer with lr=0.0001 to give time for covergenge and implemented adaptive optimization. I started using a large epoch number but settled at around 40 but implemented early stopping. I saved the model whenever the validation loss improved and used that checkpoint as the optimal model to help prevent against overfitting.

2. **XGBoost**: The second model used for my predicion was an gradient boosting random forest.
- Design: I used XGBoost, an implementation of gradient boosting to create a decision forest model. The problem with decision trees is often overfitting, so I took careful measures to combat this. My final model had 8 trees in the forest each meant to fill gaps in each others losses. Each tree had a max depth of 15.
- Training: I trained using 20 iterations but early stopping if the validaiton decreased 3 times. I used binary:logistic as the objective for classification and an eta (learning rate) of 0.35. I also made sure to implement a heavy L2 regularization to help prevent overfitting. I also tried an L1 pentaly but found it variable because it would remove different features each time changing the results significantly. 

3. **Ensamble**: I combined these two models by averaging the predictions of each as an ensamble model. I belive that each of these models performs similarly well, but has their differences and therefore working together have a more consistant prediction and less overfitting due to this limiting variance. 


## Metrics and Results

The performance of the models was evaluated using auroc of the validation sets. For each training of either model, I used about 15-20% of the data for a validation set. With this validation set, I was able to measure the AUROC for the models after training. Using this (and the public test set) I was able to make decisions about my models and choices of parameters, structure, and feature selection.

I have a 96.2% AUROC currently on the public test set but noticed the validation scores of each of my models ranging from 93-97%. After combining with ensabling I found more consistancy between 94-96% but this validation set has some bleed over from the testing and therefore is biased. 

All and all I learned a lot about creating a full ML solution from feature selection, to data pipeline, and training various models. A key is to validate properly and find reasons to make key decisions.


