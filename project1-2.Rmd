---
title: "Practical Machine Learning-Prediction Assignment Submission"
author: "Tim Tran"
date: "January 11, 2016"
output: html_document
---

# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Goals of project: One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
The goal of the project is to predict the manner of exercises by using the "classe" variable in the training set. Any other variables can be used to predict with. Prepare a report on how the model is built, and how to validate it. What is the expected out of sample error is, and why you made the choices you did.  The best prediction model will be use for the prediction of 20 different test cases. You will apply the machine learning algorithm to the 20 cases available in the test data above and submit your predictions in format to the Course Project Prediction Quiz for automated grading.  




# I. Data Description

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## Training data set can be found on the following URL:

```{r, echo=TRUE}


trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"




```

## Testing data set can be found on the followring URL:

```{r, echo=TRUE}


testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"



```

# II.Loading the data

```{r, echo=TRUE}



training <- read.csv(url(trainURL))
testing <- read.csv(url(testURL))




```

##Take a look at the dataset

```{r, echo=TRUE}
str(training, list.len= 15)


```


```{r, echo=TRUE}
table(training$classe)
prop.table(table(training$user_name, training$classe), 1)
prop.table(table(training$classe))



```



# III. Splitting training sets into test and validation sets
## We will split the training sets into two factors for cross validation purposes. We randomly subsample 70% of set for training purposes(actual model building), while 30% remainder will be used only for testing, evaluation, and accuracy measurement. 

```{r, echo=TRUE}

library(caret)

set.seed(12345)
inTrain <- createDataPartition(y=training$classe, p= 0.70, list=FALSE)
train1 <- training[inTrain,]
test1 <- training[-inTrain,]
dim(train1)
dim(test1)



```

## The training data set (train1) contains 13737 observations or about 70% of entire training data set. The training data set (train2) contains 5885 or about 30% of the entire training data set. The dataset train2 will never be looked at and will be used for accuracy measurements. 


## Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with cleaning procedures below. The Near Zero variance(NZV) variables are removed and the ID variables as well. 

# IV. Cleaning the data
## Remove variables with Nearly Zero Variance

```{r, echo=TRUE}

NZV <- nearZeroVar(train1)
train1 <- train1[, -NZV]
test1  <- test1[, -NZV]
dim(train1)
dim(test1)


```


## Remove variables that are mostly NA
```{r, echo=TRUE}
AllNA    <- sapply(train1, function(x) mean(is.na(x))) > 0.95
train1 <- train1[, AllNA==FALSE]
test1  <- test1[, AllNA==FALSE]
dim(train1)
dim(test1)


```

# Remove identification only variables (column 1 to 5)

```{r, echo=TRUE}

train1 <- train1[, -(1:5)]
test1  <- test1[, -(1:5)]
dim(train1)
dim(test1)


```

## By cleaning the data above, the number of variables has been reduced to 54


```{r, echo= TRUE}

library(corrplot)
corMatrix <- cor(train1[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))

```

# Data Analysis

## V. Prediction with Decision Tree
 
```{r,}


library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(knitr)

set.seed(12345)


#train_control <- trainControl(method = "cv", number=10)

modFit1 <- rpart(classe ~ ., data= train1, method = "class")

prediction1 <- predict(modFit1, test1, type= "class")
cmtree <- confusionMatrix(prediction1, test1$classe)
cmtree

## Fit the model after preprocessing
modFit2 <- train(classe ~ ., data= train1, method = "rpart", preProcess = c("center", "scale"))

fancyRpartPlot(modFit1)

## Overall Statistics

##  Accuracy : 0.7368          
##          95% CI : (0.7253, 0.748)
##    No Information Rate : 0.2845         
##    P-Value [Acc > NIR] : < 2.2e-16      
                                         
##                  Kappa : 0.6656         
##  Mcnemar's Test P-Value : < 2.2e-16      


#fancyRpartPlot(modFit1)

```

```{r, echo=TRUE}

plot(cmtree$table, col= cmtree$byClass, main= paste("Decision Tree Confusion Matrix:Accuracy=", round(cmtree$overall['Accuracy'],4)))

```


## Decision tree has a prediction accuracy of 0.7368.
## After preprocessing, the decision tree model has a prediction accuracy of 0.514. The accuracy of the decision tree models is not good at all. The accuracy is 79%. Preprocessing the data did not help the performance of the regression tree based on predictions. We will chose random forest next. 

# VI. Prediction with Random Forest
## We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general.


## We use a randomForest function to fit the predictor to the training set. In the computer used for this analysis, the default number of trees( 100) gives a reasonable tradeoff training time and accuracy time. 

```{r, randomForest}
library(randomForest)
set.seed(12345)
fitModel <- randomForest(classe ~ ., data= train1,importance= TRUE, ntree= 100)
varImpPlot(fitModel)


```

## Since the Random Forests yielded better Results with 0.9962, we will apply the random forest model to the test subsample


# VII. Applying the Model to the Testing subsample.

## We use the predictor on the testing subsample to get an estimate of its out of sample error. The error estimate can be obtained with the confusionMatrix function of caret package.

## Prediction on Test dataset

```{r, echo=TRUE}

predictions <- predict(fitModel, test1, type= "class")

confusions <- confusionMatrix(predictions, test1$classe )
confusions

## Overall Statistics
##  Accuracy : 0.9947          
##             95% CI : (0.9945, 0.9974)
##   No Information Rate : 0.2845          
##    P-Value [Acc > NIR] : < 2.2e-16       
                                          
##                 Kappa : 0.9952      



```
    
  
# VIII. Using the Random Forest model to apply to predict the 20 quiz results (testing datasets) as shown below.

```{r, echo=TRUE}

predictRandForest <- predict(fitModel, newdata=testing)
predictRandForest



```



## Plot of random Forest
```{r, echo=TRUE}
plot(fitModel)


```



```{r, echo=TRUE}

plot(confusions$table, col = confusions$byClass, main= paste("Random Forest Confusion Matrix:Accuracy=", round(confusions$overall['Accuracy'], 4)))

```


# IX. Prediction with Generalized Boosted Regression

```{r, echo=TRUE}

set.seed(12345)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

gbmFit <- train(classe~ ., data= train1, method = "gbm", trControl = fitControl, verbose= FALSE)

gbmFinMod1 <- gbmFit$finalModel

gbmPredTest <- predict(gbmFit, newdata= test1)

gbmAccuracyTest <- confusionMatrix(gbmPredTest, test1$classe)
gbmAccuracyTest

# Overall Statistics
                                          
#              Accuracy : 0.9839          
#                95% CI : (0.9803, 0.9869)
#   No Information Rate : 0.2845          
#   P-Value [Acc > NIR] : < 2.2e-16       
                                          
#                  Kappa : 0.9796          
# Mcnemar's Test P-Value : NA                  





```

## Generalized Boosted Regression has prediction of 0.987


```{r,}
plot(gbmFit, ylim= c(0.9, 1))


```

# X. Prediction with Linear Discriminant Analysis
## We will test the Linear Discriminant Analysis in accurately predicting the testing data. This assumes that the data follow a probablistic model. 

```{r, echo=TRUE}

set.seed(2127)

modFitLDA <- train(classe ~., data= train1, method = "lda")

```




```{r, echo=TRUE}

predictLDA <- predict(modFitLDA, newdata= test1)

confusionMatrix(predictLDA, test1$classe)

#  Overall Statistics
                                          
#               Accuracy : 0.7071          
#                 95% CI : (0.6952, 0.7187)
#    No Information Rate : 0.2845          
#    P-Value [Acc > NIR] : < 2.2e-16       
                                          
#                  Kappa : 0.6291          
# Mcnemar's Test P-Value : < 2.2e-16       


```



# XI. Predicting Results on the Test Data

## The accuracy of the 4 regression modelling methods above are:
##  a. Random Forest: 0.9947
##  b. Decision Tree: 0.7368
##  c. GBM:  0.9839
##  d. LDA:  0.7071

## Random Forests gave the best accuracy in my Testing dataset of 99.62% out of the four prediction models. THis is more accurate than Decision Tree of 79.58%, Generalized Boosted Regression of 98.7%, and the Linear Discriminant Analysis LDA of 70.71%.  




