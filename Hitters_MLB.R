############### Hitters (Major League Baseball data from 1986 and 1987 seasons) ######################

# Data Set: This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
# Number of records, n = 322 (Major league players information)
# Number of variables, p = 20 
# AtBat: Number of times at bat in 1986
# Hits: Number of hits in 1986
# HmRun: Number of home runs in 1986
# Runs: Number of runs in 1986
# RBI: Number of runs batted in in 1986
# Walks: Number of walks in 1986
# Years: Number of years in the major leagues
# CAtBat: Number of times at bat during his career
# CHits: Number of hits during his career
# CHmRun: Number of home runs during his career
# CRuns: Number of runs during his career
# CRBI: Number of runs batted in during his career
# CWalks: Number of walks during his career
# League:  A factor with levels A and N indicating player’s league at the end of 1986
# Division: A factor with levels E and W indicating player’s division at the end of 1986
# PutOuts: Number of put outs in 1986
# Assists: Number of assists in 1986
# Errors: Number of errors in 1986
# Salary: 1987 annual salary on opening day in thousands of dollars  (Response Variable)
# NewLeague: A factor with levels A and N indicating player’s league at the beginning of 1987

install.packages("ISLR") # to access the "Hitters" data set.
install.packages("leaps") # to access the regsubsets()
install.packages("gbm")  # to access boosting
install.packages("glmnet")   # to access Ridge Regession and Lasso procedure
install.packages("pls")  # to access PCR and PLS
install.packages("randomForest")  # for bagging


library(ISLR)
library(leaps)
library(gbm)
library(glmnet)
library(pls)
library(randomForest)

fix(Hitters)
names(Hitters)
dim(Hitters)

sum(is.na(Hitters))  # there are 59 'NA' records, let's remove it

Hitters = na.omit(Hitters)
sum(is.na(Hitters))

### Linear Regression using best-subset selection(predictors) and choose the model based on BIC to reduce the test MSE ### 

regfit.full = regsubsets(Salary ~ . , Hitters)
summary(regfit.full)

# It does a best subset selection by identifying the best model based on the RSS.
# The number of astrick indicates the given variable is included in the model.
# Hits and CRBI are the most favored predictors.
# By default regsubsets reports up to the best eight-variable model. Lets use the entire predictor set using nvmax

regfit.full = regsubsets(Salary ~ . , Hitters, nvmax = 19)
reg.summary = summary(regfit.full)

names(reg.summary)

# Now's let's plot RSS, Adjusted R-sq, Cp and BIC for all of the models at once to decide which model to select.

par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")

plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)

plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)

plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)

# BIC plot shows the best model is the 6 variable models, where BIC is the lowest (estimate of Test MSE)

# Alternatively, we can use the the built-in plot function of regsubsets()

plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")

# The BIC plot shows 6 black squares at the top, showing the optimal model associated with that statistics.
# These 6 variables are AtBat, Hits, Walks, CRBI, DivisionW and PutOuts.
# Let's find the regression coefficients of this model-

coef(regfit.full, 6)

### Forward and Backward stepwise selection ###

regfit.fwd = regsubsets(Salary ~ . , data = Hitters, nvmax = 19, method = "forward")
plot(regfit.fwd, scale = "bic")

regfit.bwd = regsubsets(Salary ~ . , data = Hitters, nvmax = 19, method = "backward")
plot(regfit.bwd, scale = "bic")

# Let's predict

lm.fit = lm(Salary ~ AtBat + Hits + Walks + CRBI + Division + PutOuts, data = Hitters)
summary(lm.fit)  # all coefs are statistically significant

lm.pred = predict(lm.fit, Hitters)
mean((lm.pred - Hitters$Salary)^2)  # the MSE is 99600
sqrt(mean((lm.pred - Hitters$Salary)^2))  # the RSE = 315

# On an average the Predicted Salary deviated from the True Salary by 315 thousands US dollar
# Is this good or bad? well, the answer is let's check with respect to the mean salary

sqrt(mean((lm.pred - Hitters$Salary)^2)) / mean(Hitters$Salary) * 100

# The percentage error looks quite high, i.e. 59 %.

# It is quite evident as we did not do any resampling (i.e. test and train and doinr it several times in different ways)
# Now lets employ these changes and see how it works.

##### Choosing among models using Validation set approach in conjuction with the regsubset and regression #######

# Validation set is a direct method to estimate the test error rate unlike AIC, BIC etc. (indirect methods)
# Let's divide the data set into tain and test.

set.seed(1)
train = sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
test = (!train)

# Fit the regsubsets() to the training set in order to perform best subset selection.

regfit.best = regsubsets(Salary ~ . , data = Hitters[train, ], nvmax = 19)

# We now compute validation set error (test MSE) for the best model of each size.
# Let's create a model matrix from the test data set.

test.mat = model.matrix(Salary ~ . , data = Hitters[test, ])  # the response variable is omitted by default.

# Now we run a loop as many times as the number of the predictors i.e. 19
# For each size i, we extract the coefficients from the regfit.best for the best model of that size and
# multiply them into the appropriate columns of the test model matrix to form the predictions and
# calculate the test MSE (validation set MSE)

val.errors = rep(NA, 19)
for (i in 1:19)
{
	coefi = coef(regfit.best, id = i)
	pred = test.mat[, names(coefi)]%*%coefi   # matrix multiplication for all rows and specific columns
	val.errors[i] = mean((Hitters$Salary[test] - pred)^2)
}
val.errors
which.min(val.errors) # the validation error is minimum for the 10 variable model.
coef(regfit.best, 10)

# So, the best 10 variable model is-
#(Intercept)       AtBat        Hits       Walks      CAtBat       CHits 
#-80.2751499  -1.4683816   7.1625314   3.6430345  -0.1855698   1.1053238 
#     CHmRun      CWalks     LeagueN   DivisionW     PutOuts 
#  1.3844863  -0.7483170  84.5576103 -53.0289658   0.2381662 


# Finally we perform the best subset selection on the full data set to obtain the more accurate coefficient estimates.

regfit.best = regsubsets(Salary ~ . , data = Hitters, nvmax = 19)
coef(regfit.best, 10)

# (Intercept)        AtBat         Hits        Walks       CAtBat        CRuns 
# 162.5354420   -2.1686501    6.9180175    5.7732246   -0.1300798    1.4082490 
#        CRBI       CWalks    DivisionW      PutOuts      Assists 
#   0.7743122   -0.8308264 -112.3800575    0.2973726    0.2831680 

## As regsubsets() doesn't have a predict method, now take this best model and fit a linear regression.

lm.fit = lm(Salary ~ AtBat + Hits + Walks + CAtBat + CRuns + CRBI + CWalks + Division + PutOuts + Assists, data = Hitters)
summary(lm.fit)

# Note , the parameter coefficents using both regsubsets and linear regression are same and they are all statistically significant.

## Now, let's fit an lm() on train and predict the test and check for the MSE and RSE.

lm.fit = lm(Salary ~ AtBat + Hits + Walks + CAtBat + CRuns + CRBI + CWalks + Division + PutOuts + Assists,
		data = Hitters, subset = test)
summary(lm.fit)  # the r-squared have increased.

Hitters.test = Hitters[test, ]
lm.pred = predict(lm.fit, Hitters.test)
mean((Hitters.test$Salary - lm.pred)^2)  # the test MSE 95785  (lower than indirect method)
sqrt(mean((Hitters.test$Salary - lm.pred)^2))  # the test RSE 309

# On an average the Predicted Salary deviated from the True Salary by 309 thousands US dollar

sqrt(mean((Hitters.test$Salary - lm.pred)^2)) / mean(Hitters.test$Salary) * 100

# The percentage error is 54 % but better than indirect method of computing test MSE like BIC etc.


########## Choosing different model sizes using K fold Cross-validation approach ############

k = 10
set.seed(1)
folds = sample(1:k, nrow(Hitters), replace = TRUE)
cv.errors = matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))  # 10 rows and 19 columns with values 'NA'

# Now we write a loop for k-fold cross validation.
# In the jth fold, the elements fo the folds that equal j are in the test set and
# the remainder are in the training set. 

# There are no predict() for regsubsets(). So, instead we can write our own function.

predict.regsubsets = function(object, newdata, id,...)
{
	form = as.formula(object$call[[2]])
	mat = model.matrix(form, newdata)
	coefi = coef(object, id=id)
	xvars = names(coefi)
	mat[, xvars]%*%coefi
}

for(j in 1:k)
{
	best.fit = regsubsets(Salary ~ . , data = Hitters[folds != j,], nvmax = 19)
	for(i in 1:19)
	{
		pred = predict(best.fit, Hitters[folds == j,], id = i)
		cv.errors[j,i] = mean((Hitters$Salary[folds == j] - pred)^2)
	}
}

# The first row indicates the test MSE(k = 1) across all the 19 predictors best model.
# Similarly second row is the test MSE(k = 2) across all the 19 predictors best model. ...
# So, the column sum will be the average of all the test sets (1 to 10) across all the 19 predictors best model.

mean.cv.errors = apply(cv.errors, 2, mean)  # for a matrix 1 indicates rows, 2 indicates columns, c(1, 2) indicates rows and columns. 
mean.cv.errors

par(mfrow = c(1, 1))
plot(mean.cv.errors, type = 'b')

# The cross-validation error selects an 11-variable best model. We now do the best subset selection on the full data set.

reg.best = regsubsets(Salary ~ . , data = Hitters, nvmax = 19)
coef(reg.best, 11)

reg.pred = predict(reg.best, Hitters.test, id = 11)
mean((Hitters.test$Salary - reg.pred)^2)  # test MSE 107990

lm.fit = lm(Salary ~ AtBat + Hits + Walks + CAtBat + CRuns + CRBI + CWalks + League + Division + PutOuts + Assists,
		data = Hitters, subset = test)
summary(lm.fit)

lm.pred = predict(lm.fit, Hitters.test)
mean((Hitters.test$Salary - lm.pred)^2)  # test MSE is 95717

sqrt(mean((Hitters.test$Salary - lm.pred)^2))  # the test RSE 309

# On an average the Predicted Salary deviated from the True Salary by 309 thousands US dollar

sqrt(mean((Hitters.test$Salary - lm.pred)^2)) / mean(Hitters.test$Salary) * 100

# The percentage error is 54 % but better than indirect method of computing test MSE like BIC etc.


#################################### Ridge Regression ########################################

x = model.matrix(Salary ~ . , Hitters)
y = Hitters$Salary

set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = -train
y.test = y[test]

# Model matrix contains all the predictors and also transforms any qualitative variables to dummy variables.
# It is needed as glmnet can only take numerical and qualitative inputs.

grid = 10^seq(10, -2, length = 100)
ridge.mod = glmnet(x[train, ], y[train], alpha = 0, lambda = grid, thresh = 1e-12)  # alpha = 0 denotes Ridge Regression

# Here, we are choosing the value of lambda (tuning parameter or the shrinkage parameter) across a grid of values.
# Higher the value of lambda , more is the coefficient shrinkage tending towards zero but never zero.
# As lambda increases from zero to a certain higher value, the variance decreases more sharply at the cost of a little bias increases
# We need to select the point where the trade off suits us (in terms of the test MSE)

# The value of lambda is from 10^10 to 10^-2 essentially covering full range of scanarios from the null model
# containing the intercept, to the least squares fit.
# By default glmnet() standardizes the variables so that they are on the same scale.
# To turn off this default setting, we use an argument called standardize=FLASE

# Associated with each value of lambda is a vector of ridge regression coefficients, stored in a matrix that can be accessed by coef().
# In this case, it is a 20 x 100 matrix with 20 rows (one for each predictor, plus an intercept) and 100 columns for each value of lambda (length = 100)

dim(coef(ridge.mod))

# Now, after the model fit, let's do the prediction, using the optimal / best value of lambda so that we get the best model.
# We can go back to our old friend, k-fold cross validation (which is also built in glmnet() )

cv.out = cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)  # plotting lambda and mean squared error

names(cv.out)  # of many nodes, our choice would be to drill down to 'lambda.min'

bestlam = cv.out$lambda.min  # value of lambda where the MSE is minimum (least cross validation error)

# Value of lambda is 646, let's predict with this value of lambda

ridge.pred = predict(ridge.mod, s = bestlam, newx = x[test,])
mean((ridge.pred - y.test)^2)

sqrt(mean((ridge.pred - y.test)^2)) # the RSE is 314

sqrt(mean((y.test - ridge.pred)^2)) / mean(y.test) * 100

# The test MSE is 98971

# Now let's refit our ridge regression model on the full data set, using the value of lambda chosen by the coss-validation.

out = glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlam)

############################# The Lasso #############################

lasso.mod = glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

# Here the coefficients get to zero based on the strength of shrinking, alpha = 1 represents it is lasso.

# Let's perform the k-fold cross validation to select the best tuning parameter (lambda)

set.seed(2)
cv.out = cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

bestlam = cv.out$lambda.min
lasso.pred = predict(lasso.mod, s = bestlam, newx = x[test, ])
mean((lasso.pred - y.test)^2)

# The test MSE is 101560

sqrt(mean((lasso.pred - y.test)^2))  # the RSE 318

# Now, let's see the coefficients (they will be sparse)

out = glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef = predict(out, type = "coefficients", s = bestlam)
lasso.coef  # you only have 6 statistically significant coefficients.

# Hits, Walks , CRuns , CRBI, DivisionW , PutOuts

# The above list is very similar to the one which we had got earlier using the least BIC value (indirect method)
# These 6 variables are AtBat, Hits, Walks, CRBI, DivisionW and PutOuts.


##################### Dimension Reduction Methods #######################

############# Principal Component Regresions ##############


set.seed(3)

pcr.fit = pcr(Salary ~ . , data = Hitters, scale = TRUE, validation = "CV")

# We use PCR to reduce the dimension of the problem. Our goal is to choose the principal components which
# explains most of the variability of the data as well as shows a relationship with the response.
# These are called the first principal component, second principal component etc.
# Then number of principal components are chosen using the 10-fold cross-validation method and then using these components
# the OLS(model) is fitted instead of doing it using the basket of predictors. Again we have to strike a balance b/w bias and variance.
# The data is standardized before doing it.

summary(pcr.fit)

# It reports the cross validation score for each possible number of components starting from M = 0
# Lets plot the cross-validation scores, MSEP plots the cross-validation MSE.

validationplot(pcr.fit, val.type = "MSEP")

# From the plot, the smallest cross-validation error occurs when M = 16 components are used.
# It is barely any dimension reduction from M = 19.
# We can also see that at M = 6, the cross validation error is about the same as M = 16.
# This suggests that a model that uses just a small number of componenets might suffice.

# Also, summary captures the amount of variance explained, we see for M = 6, about 89% of the variability is explained.

# Let's fit PCR on the training data and evaluate the test set performance

set.seed(5)
pcr.fit = pcr(Salary ~ . , data = Hitters, subset = train, scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")

# Now we can see that the lowest cross-validation error occurs when M = 6 components are used.
# Let's predict

pcr.pred = predict(pcr.fit, Hitters[test, ], ncomp = 6)
mean((pcr.pred - y.test)^2)

sqrt(mean((pcr.pred - y.test)^2))

# The test MSE is 96587, which is competitive with the results obtained from ridge-regression and lasso.

sqrt(mean((pcr.pred - y.test)^2))

# Let's fit the PCR on the entire data set with ncomp = 6.

pcr.fit = pcr(y ~ x, scale = TRUE, ncomp = 6)
summary(pcr.fit)

################## Partial Least Squares ######################

# PCR identifies the linear combinations or directions, that best represent the predictors .
# PLS is a supervised alternative to PCR.

set.seed(5)
pls.fit = plsr(Salary ~ . , data = Hitters, subset = train, scale = TRUE, validation = "CV")
summary(pls.fit)

validationplot(pls.fit, val.type = "MSEP")

# The lowest cross-validation error occurs when M = 3 partial least squars directions are used.
# Let's evaluate the test set MSE.

pls.pred = predict(pls.fit, Hitters[test, ], ncomp = 3)
mean((pls.pred - y.test)^2)

# The test MSE is 100862 which is slightly higher.

sqrt(mean((pls.pred - y.test)^2))

# Finally we perform the pls using the entire data set.

pls.fit = plsr(Salary ~ . , data = Hitters, scale = TRUE, ncomp = 3)
summary(pls.fit)

# While PLS searches for directions that explain variance in both the predictors and the response.

################# Boosting to predict the salary in the Hitters data set. ###################

# Log transforming the salary

# Hitters$Salary = log(Hitters$Salary)

fix(Hitters)

train = 1:200  # training set with first 200 obs.
Hitters.train = Hitters[train, ]
Hitters.test = Hitters[-train, ]

### Now perform boosting on the training data set.

set.seed(102)
pow = seq(-10, -0.1, by = 0.2)
lambda = 10^pow
length.lambda = length(lambda)
train.errors = rep(NA, length.lambda)
test.errors = rep(NA, length.lambda)

for(i in 1:length.lambda)
{
	boost.Hitters = gbm(Salary ~ . , data = Hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambda[i])
	train.pred = predict(boost.Hitters, Hitters.train, n.trees = 1000)
	test.pred = predict(boost.Hitters, Hitters.test, n.trees = 1000)
	train.errors[i] = mean((Hitters.train$Salary - train.pred)^2)
	test.errors[i] = mean((Hitters.test$Salary - test.pred)^2)
}

# Here a shrinkage parameter applied to each tree in expansion. It is also known as the learning rate or step-size reduction.
# Boosting is known for slow and steady learning.

# Plotting training set and test set MSE with respect to the shrinkage parameters.

par(mfrow = c(1, 2))
plot(lambda, train.errors, type = "b", xlab = "Shrinkage", ylab = "Train MSE", col = "green", pch = 20)
plot(lambda, test.errors, type = "b", xlab = "Shrinkage", ylab = "Test MSE", col = "blue", pch = 20)

min(test.errors)
lambda[which.min(test.errors)]

sqrt(min(test.errors))

# the RSE is 240

# Minimum test MSE is 57709.19 when the value of lambda is 0.01


### Now, let's go back to the boosted model and set the optimal shrinkage parameter which has the lowest test set MSE ###

boost.best = gbm(Salary ~ . , data = Hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambda[which.min(test.errors)])
summary(boost.best)

# The result summary is given below (they are the most significant parameters)-

#CAtBat       CAtBat 20.5648409
#CRuns         CRuns 10.1417339
#CHits         CHits  8.8890308
#CWalks       CWalks  7.6914006
#Years         Years  6.8173231

# Now let's apply bagging to the data set.

set.seed(105)
rf.Hitters = randomForest(Salary ~ . , data = Hitters.train, ntree = 500, mtry = 19)
rf.pred = predict(rf.Hitters, Hitters.test)
mean((Hitters.test$Salary - rf.pred)^2)

sqrt(mean((Hitters.test$Salary - rf.pred)^2))

# The test RSE using Bagging is 232 which is slightly lower than test set MSE for boosting.
