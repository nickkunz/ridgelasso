## Preface

Included in this repo is the RMarkdown document with the code used to create this README.md (excludes this Preface section and plot image links). Also included here is the R file containing only the requirements for analytical reproduction (excludes the narrative). In addition, the images folder provides all the plots generated. Enjoy!
<br><br>

## Introduction
This exercise briefly explores the topic of linear regularization or 'shrinkage'. It broadly serves as an exploratory exercise with commonly applied benchmarks for more complex machine learning methods. It exclusively focuses on two models, Ridge Regression and LASSO Regression. This exercise assumes theoretical knowledge and the notation of Ridge Regression's L2 penalty and LASSO's L1 penalty. This work was completed in partial fulfillment for the course Machine Learning for Finance & Marketing at Columbia Business School in 2018 with Dr. Lentzas, and was taken in part from 'Lab 2: Ridge Regression and the Lasso' from Chapter 6 in the text "An Introduction to Statistical Learning with Applications in R" by James, Witten, Hastie, and Tibshirani (2016).
<br><br>

## Requirements

First, we load the 'ISLR' library and the 'glmnet' library. The 'ISLR' library is required to load data we are interested in analyzing. The 'glmnet' library contains the 'glmnet( )' function, which is required to conduct Ridge Regression and LASSO Regression.

```{r}
## load libraries
library(ISLR)  # data
library(glmnet)  # regression
```
<br>

## Data

Here we load the 'Hitters' data from the 'ISLR' library. The data contains information regarding the performance of a given set of baseball players. The objective will be to predict the baseball player's salary by the other remaining features contained in the data frame. The first step after loading the data will be to remove any observations containing missing information utilizing the function 'na.omit( )'.

```{r}
## load data
Hitters = ISLR::Hitters 

## remove all observations missing data
Hitters = na.omit(Hitters)  
```
<br>

## Data Inspection

<h4>Feature Names</h4>

After the data has been loaded, we briefly inspect the data frame by viewing the feature names by calling the 'names( )' function.

```{r}
## view feature names
names(Hitters)  
```
<br>

<h4>Data Frame Dimension</h4>

Next, we view the number of observations and features contained within the data frame by calling the 'dim( )' function. Here we see that there are 263 observations and 20 features, which we know the names of from the previous section.
<br>

```{r}
## view data frame dimension
dim(Hitters)  
```
<br>

<h4>Preview Observations</h4>

After, we preview the first few observations to get a better idea of what is contained in the data frame by calling the 'head( )' function. Here we can see that the 263 observations contain the names of each of the baseball players.

```{r}
## view first data frame observations
head(Hitters)  
```
<br>

## Data Pre-Processing

After we have briefly inspected the data, we operationalize the data frame by both creating independent 'dummy variable' features and matrices for each feature contained the 'Hitters' data by utilizing the 'model.matrix( )' function. The 'model.matrix( )' function is useful here, as it automatically pre-processes the data for the 'glmnet( )' function to handle. Note, that the 'glmnet( )' function can only handle numerical data.

```{r}
## operationalize data
x = model.matrix(Salary~., Hitters)[,-1]  # create dummy independent variable features
y = Hitters$Salary  # create dependent variable feature
```
<br>

## Linear Regularization

<h4>Lambda</h4>

Before we can conduct linear regularization or 'shrinkage' with Ridge Regression or LASSO Regression, we need to specify the tuning parameter λ (lambda). The 'glmnet( )' function defaults to an automatically specified range of λ. However, we overide this default by specifying a wide range of values for λ. Here, λ = 10^10 to λ = 10^-2. This effectively accounts for nearly all of the useful values for λ. 

```{r}
## specify lambda
lambda = 10 ^ seq(from = 10, 
                  to = -2, 
                  length = 100)
```
<br>

<h4>Ridge Regression</h4>

Here we conduct Ridge Regression utilizing the 'glmnet' library. Note that the 'glmnet( )' function's third argument 'alpha'. The 'alpha' argument controls the type of regression. To fit Ridge Regression, 'alpha = 0' is specified. To fit LASSO Regression 'alpha = 1' is specified. Here we utilize Ridge Regression, as indicated by 'alpha = 0'. Note that the 'glmnet( )' function standardizes the numerical scale of the features by default. This default can be turned off by utilizing the argument 'standardize = FALSE'. However, in this case we apply automatic standarization.

```{r}
## ridge regression
ridge = glmnet(x = x, 
               y = y, 
               alpha = 0,  # 0 = ridge, 1 = lasso
               lambda = lambda) 
```
<br>

<h4>Lasso Regression</h4>

Here we conduct LASSO Regression utilizing the 'glmnet' library. Notice that the 'glmnet( )' function's third argument 'alpha' similar to the section above. Here we would like to utilize LASSO Regression, as indicated by 'alpha = 1'. Note that the 'glmnet( )' function still standardizes the numerical scale of the features by default. Again, this default can be turned off by utilizing the argument 'standardize = FALSE'. However, in this case we apply automatic standarization.

```{r}
## lasso regression
lasso = glmnet(x = x, 
               y = y, 
               alpha = 1, # 0 = ridge, 1 = lasso
               lambda = lambda) 
```
<br>

<h4>Lambda Grid</h4>

Next, we view the 'n x m' dimension of the matrix that stores the results of the regression by calling the 'dim( )' function on the coefficients. We see that the result is a matrix of 20 × 100, containing 20 observations or rows 'n' by 100 features or columns 'm'. Meaning, that there are 20 independent variables (one for each predictor, plus an intercept), which predict 100 regression models with a different tuning parameter (one for each λ). 

```{r}
## ridge regression dimension
dim(coef(ridge))

## lasso regression dimension
dim(coef(lasso))
```
<br>

<h4>Coefficients for Arbitrary Lambdas</h4>

After, we are interested in examining the coefficients for a specific λ (out of 100 different possibile λ previously specified). In this particular case, we examine the middle of the lambda grid, where λ = 11,498. Ridge Regression will only be examined here for the sake of brevity. However, the same analysis can be conducted for LASSO Regression.

```{r}
## specific lambda
round(ridge$lambda[50])
```

Next, we take the previous case where λ = 11,498, to calculate and view the model's coefficients.

```{r}
## ridge regression coefficients
coef(ridge)[,50]
```

Furthermore, we can explore additional λ values. In the following case, where λ = 705.

```{r}
## specific lambda
round(ridge$lambda[60])
```

Much like the previous case, we calculate and view the model's coefficients, where λ = 705.

```{r}
## ridge regression coefficients
coef(ridge)[,60]
```
<br>

<h4>Prediction</h4>

Here we introduce the 'predict( )' function. In the following example, we utilize it to call the Ridge Regression coefficients, where λ = 50.

```{r}
## ridge regression prediction function
predict(ridge, 
        s = 50,  # lambda
        type = "coefficients")[1:20,]
```
<br>

## Data Splitting

Now that we have explored linear regularization or 'shrinkage', we transition into testing its predictive accuracy by splitting the data into training and testing data. Splitting is important, as it allows us to optimally refit the model's estimate by reducing the the test mean squared error (Test MSE). The Test MSE is the primary metric we use for assessing the predictive accuracy of our model. A prior step before we move forward with splitting our data into a training and test set, is specifying a random number with the function 'set.seed( )'. This allows our results to be reproducible for further analysis. We will continue to utilize the 'set.seed( )' function for the remainder of this study. However, its explaination will be truncated here. After we begin at a random number, we split our data into training and test sets by calling the function 'sample( )'.

```{r}
## random number
set.seed(1)  

## training and test split
train = sample(1:nrow(x), nrow(x)/2)  # create training data
test = (-train)  # create test data
y.test = y[test]  # create test
```
<br>

## Prediction with Ridge Regression

<h4>Training</h4>

Here we begin to train the Ridge Regression utilizing the 'glmnet( )' function. Recall that to fit a Ridge Regression, 'alpha = 0'. To fit a LASSO Regression 'alpha = 1'. Here we would like to utilize Ridge Regression, as indicated by 'alpha = 0'. Also recall that we had overidden the 'glmnet ( )' default λ by explictly specifing a wider range of values for λ.

```{r}
## ridge regression training
ridge_model = glmnet(x[train,], 
                     y[train], 
                     alpha = 0,  # 0 = ridge, 1 = lasso
                     lambda = lambda)
```
<br>

<h4>Visualize Coefficients</h4>

To help better interpret the Ridge Regression results, we plot the coefficients. In this example, we exhibit three different ways of viewing them; by log-lambda, l1 norm, and by fractional deviance. 

```{r}
## visualize ridge regression coefficients by log-lambda
plot(ridge_model, xvar = "lambda")
```
![ridge_lambda.png](https://github.com/nickkunz/ridgelasso/blob/master/images/ridge_lambda.png)

```{r}
## visualize ridge regression coefficients by l1 norm
plot(ridge_model, xvar = "norm")
```
![ridge_norm.png](https://github.com/nickkunz/ridgelasso/blob/master/images/ridge_norm.png)

```{r}
## visualize ridge regression coefficients by fraction of deviance
plot(ridge_model, xvar = "dev")
```
![ridge_dev.png](https://github.com/nickkunz/ridgelasso/blob/master/images/ridge_dev.png)

<h4>Prediction</h4>

After we trained the model, we then utilize it for making predictions on the test set of data. Note that the 's = λ' argument specifies the tuning parameter value for λ, which is chosen as a prior step. In this case, we have chosen λ = 75.

```{r}
## ridge regression prediction, λ = 75
ridge_predict_75 = predict(ridge_model,
                           s = 75,  # lambda
                           newx = x[test,])  
```
<br>

<h4>Performance Evaluation</h4>

Next, we evaulate the model's performance, where λ = 75, by calculating the Test MSE.

```{r}
## test MSE, λ = 75
round(mean((ridge_predict_75 - y.test)^2))
```
<br>

<h4>Tuning Lambda</h4>

We can experiment with different values for λ to see if we can reduce the Test MSE and improve our model's predictive performance. Here we reduce the tuning parameter value, where λ = 4. We see that by reducing λ, reduced our Test MSE, therefore increasing our model's predictive performance. We could potentially create a model by arbitrarily choosing every value for λ. However, that would be largely impractical. Rather, we utilize Cross-Validation to achieve this, which is conducted in the Resampling section.

```{r}
## ridge regression prediction, λ = 4
ridge_predict_4 = predict(ridge_model,
                          s = 4,  # lambda
                          newx = x[test,]) 
```
<br>

<h4>Performance Evaluation</h4>

Next, we evaulate the model's performance where λ = 4 by calculating the Test MSE.

```{r}
## test MSE, λ = 4
round(mean((ridge_predict_4 - y.test)^2))
```
<br>

## Resampling

<h4>Cross-Validation</h4>

Rather than manually testing different arbitrary values to find the optimal solution for λ in reducing the Test MSE, we resample our training and test sets utilizing Cross-Validation. The 'glmnet( )' function allows us to do this with little effort by utilizing 'cv.glmnet( )'. Note that the default k number of folds for 'cv.glmnet( )' is 10. Although this tuning parameter for Cross-Validation can be changed with the argument 'nfolds = k', we apply the default setting here.

```{r}
## random number
set.seed(1)

## ridge regression training & cross-validation
cv.ridge_out = cv.glmnet(x[train ,],
                         y[train], 
                         alpha = 0)  # 0 = ridge, 1 = lasso

## ridge regression, λ = optimal
optimal_lambda = cv.ridge_out$lambda.min
round(optimal_lambda)
```
<br>

<h4>Visualize Optimal Lambda</h4>

To help better interpret our Ridge Regression results, we plot the Test MSE's of all the possible λ's.

```{r}
## plot ridge results, λ = optimal
plot(cv.ridge_out)
```
![testmse_ridge_lambda.png](https://github.com/nickkunz/ridgelasso/blob/master/images/testmse_ridge_lambda.png)


<h4>Prediction</h4>

After we trained the model, we utilize it for making predictions on the test set of data with the optimal λ.

```{r}
## ridge regression prediction, λ = optimal
ridge_predict_optimal = predict(ridge_model,
                                s = optimal_lambda,  # lambda
                                newx = x[test,])
```
<br>

<h4>Performance Evaluation</h4>

Next, we evaulate the model's performance with the optimal λ by calculating the Test MSE.

```{r}
## test MSE, λ = optimal
round(mean((ridge_predict_optimal - y.test)^2))
```
<br>

<h4>Refitting</h4>

As a final step, we refit the Ridge Regression model on the entire data set utilizing the optimal λ and view the coefficients.

```{r}
## refit ridge regression
ridge_out = glmnet(x = x,
                   y = y,
                   alpha = 0)  # 0 = ridge, 1 = lasso

## refit ridge regression coefficients
predict(ridge_out, 
        s = optimal_lambda,  # lambda
        type = "coefficients")[1:20,]
```
<br>

## Prediction with Lasso Regression

<h4>Training</h4>

After we conduct Ridge Regression, we compare the results by conducting LASSO Regression utilizing the same method and 'glmnet( )' function above. However, in the case of LASSO Regression, we change the regularization penalty with the argument 'alpha = 1' utilizing the 'glmnet( )' function. Other than the argument 'alpha = 1', we proceed just as we did with Ridge Regression.

```{r}
## lasso regression training
lasso_model = glmnet(x[train ,],
                     y[train],
                     alpha = 1,  # 0 = ridge, 1 = lasso
                     lambda = lambda)
```
<br>

<h4>Visualize Coefficients</h4>

To help better interpret the LASSO Regression results, we plot the coefficients. In this example, we exhibit three different ways of viewing them; by log-lambda, l1 norm, and by fractional deviance. 

```{r}
## visualize lasso regression coefficients by log-lambda
plot(lasso_model, xvar = "lambda")
```
![lasso_lambda.png](https://github.com/nickkunz/ridgelasso/blob/master/images/lasso_lambda.png)

```{r}
## visualize lasso regression coefficients by l1 norm
plot(lasso_model, xvar = "norm")
```
![lasso_norm.png](https://github.com/nickkunz/ridgelasso/blob/master/images/lasso_norm.png)

```{r}
## visualize lasso regression coefficients by fraction of deviance
plot(lasso_model, xvar = "dev")
```
![lasso_dev.png](https://github.com/nickkunz/ridgelasso/blob/master/images/lasso_dev.png)

<h4>Cross-Validation</h4>

We now conduct Cross-Validation with the LASSO Regression, similarly to Ridge Regression.

```{r}
## random number
set.seed (1)

## lasso regression training & cross-validation
cv.lasso_out = cv.glmnet(x[train ,],
                         y[train],
                         alpha = 1)  # 0 = ridge, 1 = lasso

## lasso regression, λ = optimal
optimal_lambda = cv.lasso_out$lambda.min
round(optimal_lambda)
```
<br>

<h4>Visualize Optimal Lambda</h4>

To help better interpret our LASSO Regression results, we plot the Test MSE's of all the possible λ's.

```{r}
## plot ridge results, λ = optimal
plot(cv.lasso_out)
```
![testmse_lasso_lambda.png](https://github.com/nickkunz/ridgelasso/blob/master/images/testmse_lasso_lambda.png)


<h4>Prediction</h4>

After we trained the model, we utilize it for making predictions on the test set of data with the optimal λ.

```{r}
## lasso regression prediction, λ = optimal
lasso_predict_optimal = predict(lasso_model,
                                s = optimal_lambda,
                                newx = x[test,])
```
<br>

<h4>Performance Evaluation</h4>

Next, we evaulate the model's performance with the optimal λ by calculating the Test MSE.

```{r}
## lasso test MSE, λ = optimal
round(mean((lasso_predict_optimal - y.test)^2))
```
<br>

<h4>Refitting</h4>

As a final step, we refit the LASSO Regression model on the entire data set utilizing the optimal λ and view the coefficients. Here we see that 12 of the 19 coefficients are exactly zero. Meaning, the LASSO Regression with the optimal λ selected with Cross-Validation contains only 7 features. This is distintly different than Ridge Regression, where the L2 penalty 'shrinks' the coefficients to near 0, LASSO Regression's L1 penalty can reduce the coefficients to 0, effectively eliminating those features from the model, which is exhibited here.

```{r}
## refit lasso regression 
lasso_out = glmnet(x = x,
                   y = y,
                   alpha = 1)  # 0 = ridge, 1 = lasso

## refit lasso coefficients
predict(lasso_out,
        s = optimal_lambda,
        type = "coefficients")[1:20,]
```
<br>

## Conclusion

The natural next step in this exercise would be to utilize both Ridge Regression and LASSO Regression simultaneously by conducting an Elastic Net Regularization. However, that is beyond the scope of this study and will not be adddressed here. This exercised focused exclusively on the application of linear regularization or 'shrinkage' in Ridge Regression and LASSO Regression. This study serves as an exploratory exercise with commonly applied benchmarks for more complex machine learning methods. More information in this regard can be found in the text "An Introduction to Statistical Learning with Applications in R" by James, Witten, Hastie, and Tibshirani (2016).

<br><br>
