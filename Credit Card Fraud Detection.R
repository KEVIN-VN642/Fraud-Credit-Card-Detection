library(dplyr)
library(stringr)
library(caret)
library(ggplot2)
library(corrplot)
library(rpart)
library(Rborist)
library(ROSE)

#You can download dataset creditcard.csv from below link
#https://www.kaggle.com/mlg-ulb/creditcardfraud
#then unzip the file then read creditcard to R by command
##import data
#dat <- read.csv('creditcard.csv')


#alternative method is use following code:
temp <- tempfile()
url <- 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/creditcard.csv.zip'

download.file(url,temp)

dat <- read.csv(unz(temp,"creditcard.csv"),header = FALSE)
col_names <- colnames(dat)
new_colnames <- c(c("Time"),col_names[1:28],c("Amount", "Class"))
colnames(dat) <- new_colnames
head(dat)
unlink(temp)
rm(col_names,new_colnames)




#############################DATA EXPLORATION###################################
################################################################################
#dimension of data
dim(dat)
#284807     31

#Viewing first 6 rows of dataset
head(dat[,1:14])

head(dat[,15:31])

#Checking whether have missing values
anyNA(dat)
# there are no missing values in the dataset then our data is clean

#Cheking how many fraud and non-fraud records
table(dat$Class) #0 is non-fraud, 1 is fraud
#     0       1 
#  284315    492 

prop.table(table(dat$Class))
#         0           1 
#     0.998272514 0.001727486 


#there is less than 0.2% of cases are fraud so there is very high imbalance between fraud and 
#non-fraud classes.
#the traditional measure like accuracy is not suitable here as we will have over 99% accuracy
#even we labels all transactions as non-fraudulent. So, we will use AUC (Area Under Curve) as
#the metric to measure performance between different models.


#Now we can check correlation between features of dataset
corrs <- cor(dat[,-1],method = "pearson")
corrplot(corrs,number.cex = .9,method="circle",type="full",tl.cex=0.8,tl.col = "black")
#We can see that most of data features are not correlated because these are components are 
#extracted from principle component analysis.


#############################DATA PREPARATION ##################################
################################################################################
#Time feature do not indicate the actual time of transaction, so we can remove it from
#dataset
dat <- dat[,-1]
#change Class variable to factor
dat$Class <- as.factor(dat$Class)
levels(dat$Class) <- c("Not_Fraud","Fraud")
#scale numeric variables
dat[,-30] <- scale(dat[,-30])
head(dat)

#Now, we can separate the dataset into two sets: train and test
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = dat$Class, times = 1, p = 0.25, list = FALSE)
train <- dat[-test_index,]
test <- dat[test_index,]
dim(train)
dim(test)


#Due to high imbalance of Class variable, we will use different technique to
#overcome this issue. There are some methods to use here which are "down sampling"
#"up sampling", and Randomly Over Sampling Examples (ROSE). We will use all of
#them and compare the performance on each method.

#down sampling
set.seed(100,sample.kind = "Rounding")
down_train <- downSample(x=train[,-ncol(train)],y=train$Class)
table(down_train$Class)
#Not_Fraud     Fraud 
#   369         369 

#up sampling
set.seed(100,sample.kind = "Rounding")
up_train <- upSample(x=train[,-ncol(train)],y=train$Class)
table(up_train$Class)
#Not_Fraud     Fraud 
#  213236     213236 


#Random Over-Sampling Examples(ROSE)

set.seed(100,sample.kind = "Rounding")
rose_train <- ROSE(Class ~ .,data=train)$data
table(rose_train$Class)
#Not_Fraud     Fraud 
# 107024       106581


########################TRAINING DIFFERENT MODELS###############################
################################################################################

#1. Decision Trees Method

#use original training dataset
set.seed(1000,sample.kind = "Rounding") #in order to compare between sampling method
origin_fit <- rpart(Class ~ ., data=train)
#make prediction on test set
pred_fit <- predict(origin_fit,newdata=test,method="class")

roc.curve(test$Class,pred_fit[,2])
#Area under the curve (AUC): 0.927

#Use down sampling
set.seed(1000,sample.kind = "Rounding") #in order to compare between sampling method
down_fit <- rpart(Class ~ ., data=down_train)
#make prediction on test set
down_pred <- predict(down_fit,newdata=test,method="class")

roc.curve(test$Class,down_pred[,2])
#Area under the curve (AUC): 0.962


#use up sampling
set.seed(1000,sample.kind = "Rounding") #in order to compare between sampling method
up_fit <- rpart(Class ~ ., data=up_train)
#make prediction on test set
up_pred <- predict(up_fit,newdata=test,method="class")

roc.curve(test$Class,up_pred[,2])
#Area under the curve (AUC): 0.957

#use rose sampling
set.seed(1000,sample.kind = "Rounding") #in order to compare between sampling method
rose_fit <- rpart(Class ~ ., data=rose_train)
#make prediction on test set
rose_pred <- predict(rose_fit,newdata=test,method="class")

roc.curve(test$Class,rose_pred[,2])

#Area under the curve (AUC): 0.941



#2. Logistic Regression Method
#use origin sample 
set.seed(1000,sample.kind = "Rounding")
glm_fit <- glm(Class ~ ., data=train,family='binomial')
pred_glm <- predict(glm_fit,newdata=test,type="response")
roc.curve(test$Class,pred_glm,plotit = TRUE)
#Area under the curve (AUC): 0.976

#use up sample dataset--not converge
#model do not converge for this case
glm_fit_up <- glm(Class ~ ., data=up_train,family='binomial')

#use down sample dataset -- not converge
#model do not converge for this case
glm_fit_down <- glm(Class ~ ., data=down_train,family='binomial')


#use rose sample dataset
set.seed(1000,sample.kind = "Rounding")
glm_fit_rose <- glm(Class ~ ., data=rose_train,family='binomial')
pred_glm_rose <- predict(glm_fit_rose,newdata=test,type="response")
roc.curve(test$Class,pred_glm_rose)
#Area under the curve (AUC): 0.982

#3. Random Forest Method
#use original dataset
set.seed(1000,sample.kind = "Rounding")
rf_fit_origin <- Rborist(train[,-30],train[,30],nTree = 1000,minNode = 20,maxLeaf = 13)
rf_pred_origin <- predict(rf_fit_origin,test[,-30],ctgCensus="prob")
prob_origin <- rf_pred_origin$prob

roc.curve(test$Class,prob_origin[,2])
#Area under the curve (AUC): 0.971

#use up sampling dataset
set.seed(1000,sample.kind = "Rounding")
rf_fit_up <- Rborist(up_train[,-30],up_train[,30],nTree = 1000,minNode = 20,maxLeaf = 13)
rf_pred_up <- predict(rf_fit_up,test[,-30],ctgCensus="prob")
prob_up <- rf_pred_up$prob

roc.curve(test$Class,prob_up[,2])
#Area under the curve (AUC): 0.978

#use down sampling dataset
set.seed(1000,sample.kind = "Rounding")
rf_fit_down <- Rborist(down_train[,-30],down_train[,30],nTree = 1000,minNode = 20,maxLeaf = 13)
rf_pred_down <- predict(rf_fit_down,test[,-30],ctgCensus="prob")
prob_down <- rf_pred_down$prob

roc.curve(test$Class,prob_down[,2])
#Area under the curve (AUC): 0.977


#use rose sampling dataset
set.seed(1000,sample.kind = "Rounding")
rf_fit_rose <- Rborist(rose_train[,-30],rose_train[,30],nTree = 1000,minNode = 20,maxLeaf = 13)
rf_pred_rose <- predict(rf_fit_rose,test[,-30],ctgCensus="prob")
prob_rose <- rf_pred_rose$prob

roc.curve(test$Class,prob_rose[,2])
#Area under the curve (AUC): 0.962
