#importing dataset

dataset=read.csv('Social_Network_Ads.csv')

dataset$Gender=factor(dataset$Gender,
                      levels = c('Male','Female'),
                      labels = c(1,0))

#splitting dataset
install.packages('caTools')
library(caTools)

split=sample.split(dataset,SplitRatio = 0.25)
traindataset=subset(dataset, split==FALSE)
testdataset=subset(dataset, split==TRUE)

# feature scaling

#classifier

install.packages('e1071')
library(e1071)

classifier=svm(formula=Purchased~Gender+Age+EstimatedSalary,
               data=traindataset,
               type='C-classification',
               kernel='radial')

#predicition
pred=predict(classifier, newdata=testdataset)

#confusion matrix
CM=table(testdataset[,5], pred)
print(CM)
