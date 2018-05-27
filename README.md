# Sentiment analysis on Rotten Tomatoes

To participate in Kaggle competition on sentiment analysis on Rotten Tomatores dataset, this project was create with SVM (Linar SVC). 
After preprocessing steps like tokenization and cleaning some unwanted characters. One of problem on sentiment analysis is polarity (negation sentence). Therefore, this classifier uses negation sign to mark the negation. Thus, the featrue is combination of unigram, bigram and trigram.

During the experiment, several machine learning methods have been employed such as Naive Bayes and Logistic Regression. However, the final model using LinearSVC which achieved 63.21%, while Naive Bayes and Logistic Regression perform 56.6% and 61.1% respectively.

