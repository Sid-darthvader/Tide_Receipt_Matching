# Tide Receipt Matching

The problem falls into the category of imblanced binary classification.

This repository contains 4 notebooks and should be explored in the following order:
1) [Data Cleaning.ipynb](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Data%20Cleaning.ipynb)
2) [Data Analysis.ipynb](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Data%20Analysis.ipynb)
3) [Model Exploration.ipynb](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Model%20Exploration.ipynb)
4) [Finalizing Model & Generating Predictions.ipynb](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Finalizing%20Model%20%26%20Generating%20Predictions.ipynb)

## [Data Cleaning](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Data%20Cleaning.ipynb)
The original dataset had a large imbalance amongst the 2 classes (Match v/s Non-Match transactions). This would have led to sub-par performance of ML models in identifying the correct matching transaction for a given receipt id.

Class imbalance was improved by ~5 percentage points by dropping the entries corresponding to receipt_ids which had no match amongst all the available transactions. 

Head over to the [Data Cleaning](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Data%20Cleaning.ipynb) notebook for more details.

![Imbalance_Comparison](./Plots/Imbalance_Comparison.png)

## [Data Analysis](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Data%20Analysis.ipynb)
The plot below depicts the distribution of the total number of transaction matches possible for a given receipt.

It tells us that the most common possibility is 5 potential transaction matches for a scanned receipt.

![Transactions per recepit id](./Plots/PossibleTransactionsHist.png)

Further, we can see how other features of the matching vector are distributed

![Feature Distribution Full Data](./Plots/Feature_distribution_full_data.png)

Due to the heavyily imbalanced classes almost all features--except *DateMappingMatch* & *AmountMappingMatch*-- are predominantly populated with 0s.

Things get slightly better and we have marginally better distributed featuress once we restrict the data to only the matched entries. However, the distribution is still very skewed which makes it extremely challenging to build good classification models and accurately determine the correct transaction match.

![Feature Distribution Full Data](./Plots/Feature_distribution_only_matches.png)

For more details, explore the [Data Analysis notebook](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Data%20Analysis.ipynb).


## [Model Exploration](https://github.com/Sid-darthvader/Tide_Receipt_Matching/blob/main/Model%20Exploration.ipynb)
We test the performance of ~15 different ML classifiers using 10-fold cross-validation on:

### 1) Dataset balanced using synthetic data- [SMOTE](https://arxiv.org/pdf/1106.1813)
![CV_results_SMOTE](./Plots/CV_results_SMOTE.png)


### 2) Original Dataset
![CV_results_original](./Plots/CV_results_original.png)

It might be tempting to see the high accuracy and quickly declare the model as a good one, alas things are not so simple for Imbalanced Classification problems. From the bunch of classification metrics available, we use **precision-recall** over **AUC score** and standard accuracy score (since its more advisable to do so for imbalanced classes). The **F1 score** is calculated as the geometric mean of precision and recall and we choose it as our primary metric to shortlist models!

#### Note: 
- **Precision:**  Out of all the predicted transaction matches, how many were predicted correctly by the model?
- **Recall:** Out of all the transactions, how many matches were correctly identified by the model?

Furthermore, we would try to have a higher recall(sensitivity) since we are interested in identifying more matches.

So as we can see, balancing the dataset using synthetic data does not lead to good results. We proceed with the imbalanced dataset for all further tasks.

