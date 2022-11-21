Here are outlines for practical sessions


# Discovering pandas, numpy, and sklearn

- See jupyternotebook 01-PCA.ipynb


# Feature extraction

Most feature extraction requires domain specific knowledge. Extracting useful
features from images (which can be represented as matrices of numbers) is very
different than extracting useful features from text (e.g., wikipedia
articles).

In many cases, feature extraction will require specific implementation. In
this practical session, we are going to extract features from protein
sequences and fit a "simple" linear model to try to predict it's function.

Proteins are chains of amino acids. They can thus be represented as a string
of characters. Each character encodes for a protein: "L" -> lysine.
The sequence of the protein is linked to its function (as well as its 3D
structure). When discovering a new organisms, a common challenge is to attempt
to infer the function of each of its proteins.

In this practical session, we are going to attempt such a task (in a
simplified setting). The goal here is to predict whether a protein is part of
protein families involved in secretion systems from its sequence.

In order to do this, we are going to extract simple features from its sequence
and fit a linear model.

- Start by loading protein sequence data
  (`data/protein_dataset/protein_sequences.csv`) using `pd.read_csv`. You can
  use `head` to look at the structure of the data.

  The first column corresponds a unique protein identifier (e.g.,
  `GCA_000474035.1_ASM47403v1.AGX32198.1`), where the first element correspond
  to a unique organism identifier (`GCA_000474035.1`), and the second part a a
  protein identifier `ASM47403v1.1`. Then follows the protein sequence.

  Questions: how many proteins are there in this dataset?


- For each protein, extract the proportion of each amino acid in the sequence.
  The list of possible amino acids are:

  'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
  'S', 'T', 'V', 'W', 'X', 'Y'

  You should thus obtain a *feature matrix*, which we are going to call X.
  What is the shape of the feature matrix? Make sure it is of shape (n, p)
  where p is the number of amino acids (21) and n the number of proteins in
  the dataset.

- Load the labels (`data/protein_dataset/labels.csv`). Look at the
  distribution of positive and negative class in our dataset. You can use
  matplotlib's bar plot or compute statistics.

  To get an intuition on the data, we are going to explore the relationships
  between pairs of features, and between a feature and an output. 

  There are several means of doing this. The package `seaborn`, built upon
  matplotlib, makes some visualization fairly easy. Import seaboarn and use
  the jointplot function to visualize the relationship between the proportion
  of `A` and `M`.

  tip:

  ```python
  import seaboarn as sns
  sns.jointplot(X[:, 0], X[:, 1])
  ```

  Now, plot the relatioship between the proportion of `A` and whether the
  protein is part of one of the families of interest, changing the kind of
  jointplot into a regression (`kind='reg'`).

  Which of those features is the most informative?

- We are now going to fit a logistic regression onto this model using
  `sklearn.linear_model.LogisiticRegression`. By default, sklearn's logistic
  regression is penalized. Here, we are going to fit the unpenalized version
  of the logistic regression. Make sure you read the documentation carefully
  to fit an *unpenalized* logistic regression.

- When predicting on the data used for fitting the model (i.e., the feature
  martix X), compute the accuracy of the model: the proportion of correctly
  labeled samples. This is an estimation of the training error (the error of
  the model on the training data).

  Questions: do you think this is a good way to estimate how well the model is
  performing? Why?


  Look at the coefficient of the model, and in particular the coefficients
  associated to feature `A`and feature `N`. Can you conclude anything?

- Let's standardize the features. Use `sklearn.preprocessing.StandardScaler`
  to do this. Do the visualization again. Does it visually change? Why?

  Refit the model on the scaled data and estimate the training error. Does it
  change? Now look at the coefficients of the model, in particular assocated
  with `A` and `N`. Did they change? Why? Can you conclude anything?

- Let's now try to extract more complex features from the protein sequence.
  Extract the proportions of di-amino acids in the sequence : `AA`, `AC, `AD`,
  …, `YY`.

  Tip: look at `itertools.product` from the standard library.

- Fit a logistic regression on this new feature matrix of di-amino acids and
  similarly estimate the "training error." Is it better or worse than the
  first model?

- Now combine the amino acids feature matrix and the di-amino acid feature
  matrix, and fit a third model. Estimate once again the training error.

## More on feature engineering, data transformation, etc

### Feature engineering

In this practical sessions, we've worked on feature extraction and engineering
of protein sequences fairly basic features. Here's a number of elements you
may want to investigate when doing feature engineering:


* __Encoding categorical features:__ if a K-categorical feature is not ordered
  (categorie 1 is as far to categorie 2 as to categorie 3 etc), then it must
  not be encoded by a single integer specifying the categorie. We can encode
  such feature by creating K-1 binary features encoding the belonging to k-th
  category. (see
  [link](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features))

* __Feature binarization:__ some continuous features can gain predictive power
  when binarized. For exemple, in some prediction tasks, weekdays could be
  split into $working\ days$ and $not\ working\ days$. (see
  [link](http://scikit-learn.org/stable/modules/preprocessing.html#binarization))

* __Imputation of missing values:__ there are multiple strategies to input
  missing values when required (see
  [link](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)).

* __Dealing with time features or other periodic features:__ when considering
  the hour of the day as a feature, we can't encode it by the an integer
  between 1 and 24 as midnigth is as close to 11pm to 1am. An easy strategy to
  encode periodic features is to apply this transformation $x \mapsto
  \sin(\frac{2\pi x}{T})$ (T is the period). In the case of the hour of the
  day, it is   $x \mapsto \sin(\frac{2\pi x}{24})$.

* __Generating new features:__ you might want to combine the existing features into new ones that seem informative to you. It can be useful for exemple, notably when working with linear models, to generate polynomial features from the original ones. You can also use external data to transform your features; for instance, if one feature is a date, adding a feature that qualifies whether the day is a working day, a weekday or a holiday can be useful.
* ...

In many practical cases, feature engineering is the key to obtaining a huge improvement in performance.

### Pre-processing data: standardization and rescaling

You might want to consider standardizing your data, or applying some other
form of transformation. `scikit-learn` has a number of pre-processing steps
that can be applied to the data: `preprocessing.MaxAbsScaler`,
`preprocessing.QuantileTransformer`, `preprocessing.RobustScaler`,
`preprocessing.StandardScaler`.

### Unsupervised projection

If your number of features is high or correlated, it may be useful to reduce
it with an unsupervised step prior to supervised steps. We have already worked
on a widly used dimentionality reduction method in `Lab 1`, the Principal
Component Analysis. There are other means of projecting data. Look at
scikit-learn's documentation to see possible options.

We will discuss in `Lab 5` the combinaison of dimentionality reduction and a predictor.
