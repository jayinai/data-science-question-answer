# Data Science Question Answer

* [SQL](#sql)
* [Statistics and ML In General](#statistics-and-ml-in-general)
* [Supervised Learning](#supervised-learning)
* [Unsupervised Learning](#unsupervised-learning)
* [Reinforcement Learning](#reinforcement-learning)
* [System](#system)


## SQL

First off some good SQL resources:

* [W3schools SQL](https://www.w3schools.com/sql/)
* [SQLZOO](http://sqlzoo.net/)

Questions:

* [Difference between joins](#difference-between-joins)


### Difference between joins

* **(INNER) JOIN**: Returns records that have matching values in both tables
* **LEFT (OUTER) JOIN**: Return all records from the left table, and the matched records from the right table
* **LEFT (OUTER) JOIN**: Return all records from the left table, and the matched records from the right table
* **FULL (OUTER) JOIN**: Return all records when there is a match in either left or right table

![](assets/sql-join.png)

[back to top](#data-science-question-answer)


## Statistics and ML In General

* [Cross Validation](#cross-validation)
* [Feature Importance](#feature-importance)
* [Mean Squared Error vs. Mean Absolute Error](#mean-squared-error-vs.-mean-absolute-error)
* [L1 vs L2 regularization](#l1-vs-l2-regularization)
* [Correlation vs Covariance](#correlation-vs-covariance)
* [Would adding more data address underfitting?](#would-adding-more-data-address-underfitting?)
* [Activation Function](#activation-function)


### Cross Validation

Cross-validation is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a validation set to evaluate it. For example, a k-fold cross validation divides the data into k folds (or partitions), trains on each k-1 fold, and evaluate on the remaining 1 fold. This results to k models/evaluations, which can be averaged to get a overall model performance.

![](assets/cv.png)

[back to top](#data-science-question-answer)


### Feature Importance

* In linear models, feature importance can be calculated by the scale of the coefficients
* In tree-based methods (such as random forest), important features are likely to appear closer to the root of the tree.  We can get a feature's importance for random forest by computing the averaging depth at which it appears across all trees in the forest.

[back to top](#data-science-question-answer)


### Mean Squared Error vs. Mean Absolute Error

* **Similarity**: both measure the average model prediction error; range from 0 to infinity; the lower the better
* Mean Squared Error (MSE) gives higher weights to large error (e.g., being off by 10 just MORE THAN TWICE as bad as being off by 5), whereas Mean Absolute Error (MAE) assign equal weights (being off by 10 is just twice as bad as being off by 5)
* MSE is continuously differentiable, MAE is not (where y_pred == y_true)

[back to top](#data-science-question-answer)


### L1 vs L2 regularization

* **Similarity**: both L1 and L2 regularization **prevent overfitting** by shrinking (imposing a penalty) on the coefficients
* **Difference**: L2 (Ridge) shrinks all the coefficient by the same proportions but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection.
* **Which to choose**: If all the features are correlated with the label, ridge outperforms lasso, as the coefficients are never zero in ridge. If only a subset of features are correlated with the label, lasso outperforms ridge as in lasso model some coefficient can be shrunken to zero.
* In Graph (a), the black square represents the feasible region of the L1 regularization while graph (b) represents the feasible region for L2 regularization. The contours in the plots represent different loss values (for the unconstrained regression model ). The feasible point that minimizes the loss is more likely to happen on the coordinates on graph (a) than on graph (b) since graph (a) is more **angular**.  This effect amplifies when your number of coefficients increases, i.e. from 2 to 200. The implication of this is that the L1 regularization gives you sparse estimates. Namely, in a high dimensional space, you got mostly zeros and a small number of non-zero coefficients.

![](assets/l1l2.png)

[back to top](#data-science-question-answer)


### Correlation vs Covariance

* Both determine the relationship and measure the dependency between two random variables
* Correlation is when the change in one item may result in the change in the another item, while covariance is when two items vary together (joint variability)
* Covariance is nothing but a measure of correlation. On the contrary, correlation refers to the scaled form of covariance
* Range: correlation is between -1 and +1, while covariance lies between negative infinity and infinity.


[back to top](#data-science-question-answer)


### Would adding more data address underfitting?

Underfitting happens when a model is not complex enough to learn well from the data. It is the problem of model rather than data size. So a potential way to address underfitting is to increase the model compleixty (e.g., to add higher order coefficients for linear model, increase depth for tree-based methods, add more layers / number of neurons for neural networks etc.)

[back to top](#data-science-question-answer)


### Activation Function

For neural networks

* Non-linearity: ReLU is often used. Use Leaky ReLU (a small positive gradient for negative input, say, `y = 0.01x` when x < 0) to address dead ReLU issue
* Multi-class: softmax
* Binary: sigmoid
* Regression: linear

[back to top](#data-science-question-answer)


## Supervised Learning


## Unsupervised Learning

* [Autoencoder](#autoencoder)


### Autoencoder

* The aim of an autoencoder is to learn a representation (encoding) for a set of data
* An autoencoder always consists of two parts, the encoder and the decoder. The encoder would find a lower dimension representation (latent variable) of the original input, while the decoder is used to reconstruct from the lower-dimension vector such that the distance between the original and reconstruction is minimized
* Can be used for data denoising and dimensionality reduction 


![](assets/autoencoder.png)


## Reinforcement Learning


## System

* [Cron job](#cron-job)
* [Linux](#Linux)

### Cron job

The software utility **cron** is a **time-based job scheduler** in Unix-like computer operating systems. People who set up and maintain software environments use cron to schedule jobs (commands or shell scripts) to run periodically at fixed times, dates, or intervals. It typically automates system maintenance or administration -- though its general-purpose nature makes it useful for things like downloading files from the Internet and downloading email at regular intervals.

![](assets/cron-job.png)

Tools:
* [Apache Airflow](https://airflow.apache.org/)

[back to top](#data-science-question-answer)


### Linux

Using **Ubuntu** as an example.

* Become root: `sudo su`
* Install package: `sudo apt-get install <package>`

[back to top](#data-science-question-answer)