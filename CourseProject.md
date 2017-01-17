# Activity Quality Controller
by Dr P. Physics  
15 January 2017  







## 1. Introduction 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. Although people largely monitor how much they do in terms of acitvity, they more rarely monitor how well they do it. Using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants that were asked to perform barbell lifts correctly and incorrectly in 5 different ways, we build a predictive model for exercise quality. After trying several algorithms, we choose as final model a Random Forest with an estimated out-of-sample accuracy of 99.32%.

More information on the DLA data set used can be found here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). Please also consult the reference in the bibliography.


## 2. Data Manipualtion

### 2.1 Loading and Cleaning Data

Upon a first dowload of the files, a first content inspection allowed to spot two instance types of non-available variables, NA and #DIV/0!. We therefore match both strings to NAs when importing the files in R (see code below).

```r
rm(list=ls())
trnfileurl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
tstfileurl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
if (!(file.exists("pml-training.csv"))){ download.file(trnfileurl,destfile="./pml-training.csv",method="curl")}
if (!(file.exists("pml-testing.csv"))){ download.file(tstfileurl,destfile="./pml-testing.csv",method="curl")}
traindata <- read.csv("./pml-training.csv",sep=",", na.strings=c("NA", "", "#DIV/0!"), header = TRUE)
testdata <- read.csv("./pml-testing.csv",sep=",", na.strings=c("NA", "", "#DIV/0!"), header = TRUE)
dims<-c(dim(traindata)[1],dim(traindata)[2],dim(testdata)[1],dim(testdata)[2])
dims
```

```
## [1] 19622   160    20   160
```

The training data contains 19622 observations of 160 different variables associated to sensor measurements from up to 4 sensor belts positionned at different body areas. 

We remove entries with only NA and variables with near zero variance. Using the command "as.vector(apply(traindata, 2, function(x) length(which(is.na(x)))))", we can easily see that a large number of variables have almost no recorded values at all (19216 NAs versus 20 observations) and hence likely no have predictive power. We remove all variables that have more than 97% of observations not available.


```r
varzerovarianc <- nearZeroVar(traindata) 
traindataclean <- traindata[,-varzerovarianc]
toomanynasvars <- sapply(traindataclean, function(x) mean(is.na(x))) > 0.97
traindataclean <- traindataclean[,toomanynasvars==FALSE]
```

The resulting number of variables after this cleaning is now 59.

### 2.2 Cross-validation and data sub-sampling

The testing set contains only 20 observations. The testing set is therefore really a validation set that we thus rename as such for clarity. To put in place cross-validation for our model(s), we randomly apply sub-sampling without replacement to our training set to create to new data sets with a 70% partition, 70% of the observations going into the new training set and 30% of observations in the testing set. This new testing will be used to cross-validate the different models we test in this assignment. Only the most accurate model will be applied on the validation set.


```r
validationdata <- testdata
inTrain  <- createDataPartition(traindataclean$classe, p=0.70, list=FALSE)
trainset <- traindataclean[inTrain, ]
testset  <- traindataclean[-inTrain, ]
```

The resulting training set and testing sets have 13737 and 5885 respectively. Given that the data in the first six columns of the data sets are not sensor readings but simple descriptive variables, we can remove them as well. 


```r
trainset <- trainset[, -(1:6)]
testset  <- testset[, -(1:6)]
```

## 3. Predictive Model Building

Our approach is to test a number of different machine learning approaches and test the accuracy of each different model using the testing set. Based on the resulting prediction accuracies, we will choose the best model to submit for validation on the validation set. We will use the main models described during the lecture: a generalised boosted model (GBM), a decision tree (DT) and a Random Forest (RF). For all models, we select a number of key metrics to display to assess performance, like overall accuracy but also accuracy by movement class. For reproducibility, we set a specific seed value for each model we test.

### 3.1 Generalised Boosted Model (GBM)

The gradient boosted model, or generalised boost model is one choice of boosting classification provided in the R caret package. 


```r
set.seed(11111)
# training the model
model_GB <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
model_GB_Fit  <- train(classe ~ ., data=trainset, method = "gbm",trControl = model_GB, verbose = FALSE)
# final model information
model_GB_Fit$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 42 had non-zero influence.
```

```r
# predicting on testing set
model_GB_Pred <- predict(model_GB_Fit, newdata=testset)
model_GB_Conf <- confusionMatrix(model_GB_Pred, testset$classe)
perf_GB <- c(model_GB_Conf$overall[1],model_GB_Conf$byClass[,11])
perf_GB
```

```
##  Accuracy  Class: A  Class: B  Class: C  Class: D  Class: E 
## 0.9631266 0.9873660 0.9630731 0.9706884 0.9760124 0.9819651
```

```r
plot(model_GB_Conf$table,col=model_GB_Conf$byClass,main=paste("Overall GB Model Accuracy = ",round(model_GB_Conf$overall[1]*100,2)))
```

![](CourseProject_files/figure-html/modelgeneral-1.png)

### 3.2 Decision Tree (DT)

Decision trees map observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). When the target variable can take a finite set of values, like in this assignment, they are called classification trees. The advantage of decision trees is that they are rather simple to understand and interpret and require little data preparation.


```r
set.seed(33333)
# training the model
model_DT_Fit <- rpart(classe ~ ., data=trainset, method="class")
# predicting on testing set
model_DT_Pred <- predict(model_DT_Fit, newdata=testset, type="class")
model_DT_Conf <- confusionMatrix(model_DT_Pred, testset$classe)
perf_DT <- c(model_DT_Conf$overall[1],model_DT_Conf$byClass[,11])
perf_DT
```

```
##  Accuracy  Class: A  Class: B  Class: C  Class: D  Class: E 
## 0.7034834 0.8687900 0.7346629 0.8229636 0.8010584 0.8195623
```

```r
plot(model_DT_Conf$table,col=model_DT_Conf$byClass,main=paste("Overall DT Model Accuracy = ",round(model_DT_Conf$overall[1]*100,2)))
```

![](CourseProject_files/figure-html/modeldectree-1.png)

The figure belows gives an overview of the model classification on the training sample.


```r
fancyRpartPlot(model_DT_Fit,main="Classification Tree",sub="")
```

![](CourseProject_files/figure-html/plotmodelDT-1.png)

### 3.3 Random Forest (RF)

Random forests or random decision forests are another learning method for classification and regression. In consists in constructing a multitude of decision trees at training time. The Random Forest technique tends to be less probed to the problem of overfitting than classical decision trees tend to have. We therefore use this model to see if we can improve the accuracy of the decision tree's predictions.


```r
set.seed(22222)
# training the model
model_RF <- trainControl(method="cv", number=3, verboseIter=FALSE)
model_RF_Fit <- train(classe ~ ., data=trainset, method="rf",trControl=model_RF)
# final model information
model_RF_Fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.72%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B   17 2635    6    0    0 0.0086531226
## C    0   15 2376    5    0 0.0083472454
## D    0    0   45 2206    1 0.0204262877
## E    0    0    3    5 2517 0.0031683168
```

```r
# predicting on testing set
model_RF_Pred <- predict(model_RF_Fit, newdata=testset)
model_RF_Conf <- confusionMatrix(model_RF_Pred, testset$classe)
perf_RF <- c(model_RF_Conf$overall[1],model_RF_Conf$byClass[,11])
perf_RF
```

```
##  Accuracy  Class: A  Class: B  Class: C  Class: D  Class: E 
## 0.9932031 0.9992876 0.9963126 0.9920940 0.9888031 0.9994338
```

```r
plot(model_RF_Conf$table,col=model_RF_Conf$byClass,main=paste("Overall RF Model Accuracy = ",round(model_RF_Conf$overall[1]*100,2)))
```

![](CourseProject_files/figure-html/modelrdforest-1.png)

## 4. Results

### 4.1 Best model selection

The table below summarizes the prediction accuracies from the three models. As already observed in the previous chapter, the Decision Tree model performs the worst (30% less accuracy) while the other two models have a very high accuracy, the Random Forest performing however slightly better than the Generalised Boost one. We therefore choose the Random Forest as our final predicition model.

| Model Name | Accuracy [%] |
| :--------- | :----------- |
| Generalised Boosted | 96.31 |
| Decision Tree | 70.35 |
| Random Forest Model | 99.32|

### 4.2 Out of sample error

The accuracy of a model tends to be over-estimated when calculated based on the predictions on the training data itself, i.e. *in-sample*. This is why we constructed our cross-validation data sample (the testing set) to perform a more reliable estimation of the expected out-of-sample accuracy, that is the accuracy of the model on data it has never seen before (in this case on the validation data set, the original *testing* set for this exercise). The expected *out-of-sample error* is simply the expected rate of missclassifications which can be estimated based on the estimated out-of-sample accuracy (1 minus the accuracy).

For our chosen model, the Random Forest, the expected out-of-sample accuracy and out-of-sample error is therefore 99.32% and 0.68% respectively.

## 5. Conclusion

The best prediction model is the Random Forest with an accuracy of 99.32%. This model is the one applied on the validation set. The code use to produce the predictions for the validation and assignment validation data set is below.


```r
#generate data for submitting
prediction <- predict(model_RF_Fit, validationdata, type="raw")
fileName <- "AssignmentPredictions.txt"
write.table(prediction,file=fileName,quote=FALSE,row.names=FALSE,col.names=FALSE)
```

## Bibliography       

* Ugulino, W., Cardador, D., Vega, K., Velloso, E., Milidiu, R., Fuks, H., **Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements.**, Proceedings of 21st Brazilian Symposium on Artificial Intelligence. 

