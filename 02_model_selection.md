# Model accuracy & model selection

Recall the last practical session. We extracted several sets of features from
protein sequences. We then estimated the "train error" using the metric
"accuracy" on our models.

This is a poor way to estimate whether our models will generalize well. We
here going to set up a proper validation scheme.

## Estimating the generalization error

- Split the data into train and test, keeping 2/3 for train and 1/3 for test.
  Re-fit the three models on the training data, and estimate the accuracy on
  the test data.

  Compare the accuracy on the test data with the one on the training data.
  Comment.

- Now, compute the accuracy of a "random model," between the labels and the
  labels permutated. What is that value ?
  Comment.

  Look at scikit-learn's documentation on metrics to do model evaluation. Find
  a better measure. Which one do you think should be appropriate?

  Recompute the training error and the testing error with that metric (or
  several metric if you choose).

- Now, randomize the data, and split it again the data into train and test.
  Compute the accuracy of the model with the two metrics. Are the results
  identical?

  To randomize the data, you can use the `np.random` module of `numpy`:
  
  ```python
  import numpy as np

  # Create a ndarray from 0 to the number of proteins -1
  indices = np.arange(len(X))
  print(indices[:10])
  np.random.shuffle(indices)
  print(indices[:10])
  ```
  
  Once the `indices` are shuffled, you can split your dataset in train and
  test using numpy's indexing strategy.

  (Note that in practice, it's recommended to randomize the order of the
  samples before splitting into train test *in case* there is an order to the
  data samples.)

## Moving to a cross validation scheme

To better estimate the generalization error, specially when the dataset is
small, we typically use cross validation. We split the dataset into k subsets,
and use k-1 subset for training, and 1 subset for testing. This allows to
obtain k independant estimation of the error, and thus better estimate the
generalization error.

**Question:** In a K-fold cross-validation, how many times does each sample
appear in a test set? In a training set?  

Create a function that takes in input the number of samples and the number of
folds, and returns a list of tuples (`train_indices`, `test_indices`) with the
k-folds.

```python
def make_kfolds(n_instance, n_folds):
    """
    set up a K-fold croos-validation.
    
    Parameters:
    -----------
    n_instances: int
        the number of instances in the dataset.
    n_folds: int
        the number of folds of the cross-validation scheme
        
    Outputs:
    --------
    fold_list: list
        list of folds, a fold is a tuple of 2 lists, 
            the first one containing the indices of instances of the training set,
            the second one containing the indices of instances of the test set
    """
    your code
```

Check, for each fold, that no indices overlap between training and testing.
You can use `np.intersect1d` to do this.

Use this function to estimate the generalization error of the model using the
accuracy and the metric of your choice. Are the results stable? Comment.

Now update your code to use `sklearn`'s KFold
(`sklearn.model_selection.KFold`).

Compare the results between using `sklearn`'s `KFold and sklearn's
StratifiedKFold.

Comment.

## Training and testing error as a function of the number of samples
- Now, we are going to look at how the training and testing error vary based
  on the number of samples in the data.

  Load the data in `data/protein_dataset_2`. The structure of this folder is
  strictly identical to `data/protein_dataset`. Load the data.

  How many proteins are there in this dataset?

- Now, compute the (mean) training error and (mean) testing error by using the first 1000
  proteins, then 2000, 3000,  etc. Plot the training and testing error as a
  function of the number of datapoints. What do you see?
  comment.

## Estimating the ROC curve

- Another means of estimating how well data performs is using ROC curves. Look
  at scikit-learn's documentation, and plot the ROC curves:
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

## Other means of splitting into K-fold

Look at scikit-learn's documentation on GroupKFold
(https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold)
Can you think of an example of use cases of group k-fold?
