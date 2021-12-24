# Linear-Regression
## Linear-Regression with one variable.
### steps:
#### 1. Select features which lead us to our `Price`.
* For predicting `Price` in our dataset, we need to consider 1 feature from our dataset to deal with.
* Our feature is `sqft_living`.
#### 2. Prepare the Data.
* Select feature and target from the dataset file.
* Normalize the data.
* Separate the data to `training data` and `testing data`.
* Add Ones column for Theta zero.
* Convert the training and testing data to matrices.
* Initialize theta matrix.

#### 3. Implement the hypothesis function
* Linear Regression hypothesis function.
* ![](https://i.imgur.com/XtED8wx.png)
#### 4. Implement the cost function
* ![](https://i.imgur.com/5gFqVyz.png)
#### 5. Implement the gradient descent
* ![](https://i.imgur.com/zr1ja1Y.png)
#### 6. Predict `Price` values . 
* with X_test data and theta using the hypothesis function we can predict our `Price` and compare it with our Y_test.

## Linear-Regression with multiple variables.
* same steps with adding more features and modify hypothesis, cost function and gradient descent to fit the multiple features.
