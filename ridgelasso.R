## title: linear regularization
## by: nick kunz
## date: feb 13, 2019

## load libraries
library(ISLR)  # data
library(glmnet)  # regression

## load data
Hitters = ISLR::Hitters 

## remove all observations missing data
Hitters = na.omit(Hitters) 

## data inspection
names(Hitters)  ## view feature names
dim(Hitters)  ## view data frame dimension
head(Hitters)  ## preview observations

## operationalize data
x = model.matrix(Salary~., Hitters)[,-1]  # create dummy independent variable features
y = Hitters$Salary  # create dependent variable feature

## specify lambda
lambda = 10 ^ seq(from = 10, 
                  to = -2, 
                  length = 100)

## random number
set.seed(1)  

## training and test split
train = sample(1:nrow(x), nrow(x)/2)  # create training data
test = (-train)  # create test data
y.test = y[test]  # create test

## random number
set.seed(1)

## ridge regression training & cross-validation
cv.ridge_out = cv.glmnet(x[train ,],
                         y[train], 
                         alpha = 0)  # 0 = ridge, 1 = lasso

## ridge regression, λ = optimal
optimal_lambda = cv.ridge_out$lambda.min
round(optimal_lambda)

## plot ridge results, λ = optimal
plot(cv.ridge_out)

## ridge regression prediction, λ = optimal
ridge_predict_optimal = predict(ridge_model,
                                s = optimal_lambda,  # lambda
                                newx = x[test,])

## test MSE, λ = optimal
round(mean((ridge_predict_optimal - y.test)^2))

## refit ridge regression
ridge_out = glmnet(x = x,
                   y = y,
                   alpha = 0)  # 0 = ridge, 1 = lasso

## refit ridge regression coefficients
predict(ridge_out, 
        s = optimal_lambda,  # lambda
        type = "coefficients")[1:20,]

## visualize refit ridge regression coefficients
plot(ridge_out)

## lasso regression training
lasso_model = glmnet(x[train ,],
                     y[train],
                     alpha = 1,  # 0 = ridge, 1 = lasso
                     lambda = lambda)

## random number
set.seed (1)

## lasso regression training & cross-validation
cv.lasso_out = cv.glmnet(x[train ,],
                         y[train],
                         alpha = 1)  # 0 = ridge, 1 = lasso

## lasso regression, λ = optimal
optimal_lambda = cv.lasso_out$lambda.min
round(optimal_lambda)

## plot ridge results, λ = optimal
plot(cv.lasso_out)

## lasso regression prediction, λ = optimal
lasso_predict_optimal = predict(lasso_model,
                                s = optimal_lambda,
                                newx = x[test,])

## lasso test MSE, λ = optimal
round(mean((lasso_predict_optimal - y.test)^2))

## refit lasso regression 
lasso_out = glmnet(x = x,
                   y = y,
                   alpha = 1,  # 0 = ridge, 1 = lasso
                   lambda = lambda)

## refit lasso coefficients
predict(lasso_out,
        s = optimal_lambda,
        type = "coefficients")[1:20,]

## visualize refit lasso coefficients
plot(lasso_out)
