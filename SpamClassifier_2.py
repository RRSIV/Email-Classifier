# -*- coding: utf-8 -*-
# coding: utf-8
#MultiNomial Naive Bayes Algirithm
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Function to read files (emails) from the local directory
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

#An empty dataframe with 'message' and 'class' headers
data = DataFrame({'message': [], 'class': []})

#Including the email details with the spam/ham classification in the dataframe
data = data.append(dataFrameFromDirectory('C:/Users/Siva/Desktop/Email Classifier/Email-Spam-Classifier-Using-Naive-Bayes-master/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:/Users/Siva/Desktop/Email Classifier/Email-Spam-Classifier-Using-Naive-Bayes-master/emails/ham', 'ham'))

#Head and the Tail of 'data'
data.head()
data.tail()

##Assigning the message to X and class to y variables
X = data.iloc[:, 1].values
y = data.iloc[:, 0].values


##Using cross validation library to split the training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Bag of words concept used via CountVectorizer()
#Vectorizer.fit_transformer: tokenises/ converts individual words into numbers(values). 
##and counts how many times each word occurs.
#How many times each word occurs in an email
#Represents the count of each word in a sparse matrix
vectorizer = CountVectorizer()
counts_train = vectorizer.fit_transform(X_train)
targets_train = y_train

##Fit the data into the classifier( Multinomial Naive Bayes algorithm used here which assumes the data to have multinomial distribution)
classifier = MultinomialNB()
classifier.fit(counts_train, targets_train)

##Predict class using predict function
X_test_token = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_token)

##Using confusion matrix to assesss performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
##171	1
##31	78

## we can see that we have got 249 out of 281 observation predicted correctly so out accuracy is 88.6%

##Also we see the false negatives are high compared to false postives, however the overall false 
##negative percentage is 31/281 i.e 11%. So the naive bayes model performs good in classifying the model

##Sample Custom made messages 
exampleInput = ["Hello Professor, I am writing to express my interest", "Free Viagra !!", "Please reply to get this offer"]
excount = vectorizer.transform(exampleInput)
print(excount)

prediction = classifier.predict(excount)
print(prediction)


