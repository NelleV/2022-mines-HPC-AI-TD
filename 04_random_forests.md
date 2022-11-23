# Trees & forests

The goal of this lab is to explore and understand tree-based models on
classification problems.

We will focus successively on decision tre`es, bagging trees, and random forests. 

## Understanding decision trees

A decision tree predicts the value of a target variable by learning simple
decision rules inferred from the data features.

In scikit-learn, they are implemented in
[tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
for classification and
[tree.DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
for regression.

### Toy dataset

In order to better understand how a decision tree processes the feature space,
we will first work on a simulated dataset.

- Generate some classification data using the following code:


```python
x1 = np.random.multivariate_normal([2,2], [[0.1,0],[0,0.1]], 50)
x2 = np.random.multivariate_normal([-2,-2], [[0.1,0],[0,0.1]], 50)
x3 = np.random.multivariate_normal([-3,3], [[0.1,0.1],[0,0.1]], 50)
X1 = np.concatenate((x1,x2,x3), axis=0)

y1 = np.random.multivariate_normal([-2,2], [[0.1,0],[0,0.1]], 50)
y2 = np.random.multivariate_normal([2,-2], [[0.1,0],[0,0.1]], 50)
y3 = np.random.multivariate_normal([-3,-3], [[0.01,0],[0,0.01]], 50)
X2 = np.concatenate((y1,y2,y3), axis=0)

X_demo = np.concatenate((X1, X2), axis=0)
y_demo = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))
```

X1 and X2 correspond to two different class.

- Visualize the two first dimensions using a scatter plots. What do you expect
  the decision boundaries to look like?

- Train a DecisionTreeClassifier from sklearn. 

  Plot the decision boundaries. Two options are available to do this:

    1. **Manually:** create a mesh, i.e. a fine grid of values between the
       minimum and maximum values of the data. Using the fitted
       DecisionTreeClassifier to label each point of the mesh in order to
       estimate the decision boundaries. Then use matplotlib's `plt.contourf`
       function to plot the results.
    2. **Using scikit-learn** Look at the `inspection.DecisionBoundaryDisplay`
       function.

- Now change the splitter of the decision tree to random. This means that the
  algorithm will consider the feature along which to split *randomly* (rather
  than picking the optimal one), and then select the best among several
  *random* splitting point. Run the algorithm several times. What do you
  observer?


### Classification data

Now let's move to real data. The samples are tumors, each described by the
expression (= the abundance) of 3,000 genes. The goal is to separate the
endometrium tumors from the uterine ones.

The data can be found in `data/small_Endometrium_Uterus.csv`.

- Load the data, and look at the size, shape, and values.
  Two columns contain metadata information `ID_REF` and `Tissue`. Drop them
  and create the feature matrix.
- Split the data into 5-fold for cross validation.
- Train a decision tree classifier, and compute the accuracy.
- Retrain a decision tree 4 times, and compute the accuracy. Do you get
  different values? Why? Read the documentation for help.
- Compute the mean and standard deviation of the area under the ROC curve for
  those 5 decision trees (check out sklearn's metric module for easy functions
  to compute the AUC). And plot the 5 ROC curves.
- What parameters of DecisionTreeClassifier can you play with to define trees
  differently than with the default parameters? Cross-validate these using a
  grid search with
  [model_selection.GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
  Plot the optimal decision tree on the previous plot. Did you manage to
  improve performance?


## Bagging trees.

We will resort to ensemble methods to try to improve the performance of single
decision trees. Let us start with _bagging trees_: The different trees are to
be built using a _bootstrap sample_ of the data, that is to say, a sample
built by randomly drawing n points _with replacement_ from the original data,
where n is the number of points in the training set.

Bagging is efficient when used with low bias and high variance weak learners.
Indeed, by averaging such estimators, we lower the variance by obtaining a
smoother estimator, which is still centered around the true density (low
bias).

Bagging decision trees hence makes sense, as decision trees have:

* low bias: intuitively, the conditions that are checked become multiplicative
  so the tree is continuously narrowing down on the data (the tree becomes
  highly tuned to the data present in the training set).
* high variance: decision trees are very sensitive to where it splits and how
  it splits. Therefore, even small changes in input variable values might
  result in very different tree structure.


**Note**: Bagging trees and random forests start being really powerful when
using large number of trees (several hundreds). This is computationally more
intensive, especially when the number of features is large, as in this lab.
For the sake of computational time, we suggeste using small numbers of trees,
but you might want to repeat this lab for larger number of trees at home.

- Cross-validate a bagging ensemble of 5 decision trees on the data. Plot the
  resulting ROC curve, compared to the 5 decision trees you trained earlier.
  Use
  [ensemble.BaggingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html).

  How do the bagging trees perform compared to individual trees?
- Use cross_validate_optimize to optimize the number of decision trees to use
  in the bagging method. How many trees did you find to be an optimal choice?


## Random forest

In practice, simply bagging is typically not enough. In order to get a good
reduction in variance, we require that the models being aggregated be
uncorrelated, so that they make “different errors”. Bagging will usually get
you highly correlated models that will make the same errors, and will
therefore not reduce the variance of the combined predictor.

- What is the difference between bagging trees and random forests? How does it
  intuitively fix the problem of correlations between trees ?

- Cross-validate a random forest of 5 decision trees on the data. Plot the
  resulting ROC curve, compared to the bagging tree made of 5 decision trees.
  Use
  [ensemble.RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- What parameters of the random forest can be optimized? Use cross validation
  to optimize the following:
    - number of decision trees
    - number of features to consider at each split.

    How many trees do you find to be an optimal choice? How does the optimal
    random forest compare to the optimal bagging trees? How do the training
    times of the random forest and the bagging trees compare?
- Compare the random forest and bagged trees with l1 and l2 regularized linear
  models.
