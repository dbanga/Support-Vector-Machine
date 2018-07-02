#Importing data set
dataset=read.csv('Social_Network_Ads.csv')

dataset$Gender=factor(dataset$Gender, 
                      levels = c('Male', 'Female'),
                      labels=c(0,1))

# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)

#X=dataset[3:4]
#Y=dataset[5]

split = sample.split(dataset, SplitRatio = 0.25)
training_set = subset(dataset, split == FALSE)
test_set = subset(dataset, split == TRUE)


# Fitting SVM to the Training set
install.packages('e1071')
library(e1071)

classifier=svm(formula=Purchased ~ EstimatedSalary+Age ,
                 data=training_set, 
                 type='C-classification',
                 kernal='linear')

# Predicting the Test set results
pred = predict(classifier, newdata = test_set)

# Making the Confusion Matrix
cm = table(test_set[, 5], pred)

print(cm)