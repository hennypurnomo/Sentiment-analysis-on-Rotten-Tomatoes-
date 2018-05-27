"""
This project created for CE807 Text Analytics assignments - Sentiment analysis
Author:Rabia YASA KOSTAS and HENNY PURNOMO
Student numbers: 1700421 and 1700889
Date:22/02/2018
"""
#the main system adopted from http://ankoorb.blogspot.co.uk/2014/12/sentiment-analysis-on-rotten-tomatoes.html
#the parameters for countvectorizer from http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#the negation code from https://nlpforhackers.io/sentiment-analysis-intro/


import pandas as pd                                           #library for loading and storing data
from sklearn.feature_extraction.text import CountVectorizer   #library for converting the reviews into a matrix of token counts
from sklearn.feature_extraction.text import TfidfTransformer  #library for converting a count matrix to a tf representation
from sklearn import svm                                       #library for creating the classifier, SVM
from nltk import word_tokenize                                #library for tokenize the reviews
from nltk.sentiment.util import mark_negation                 #library for detecting and assigning special tag of negation on the review


#Opening train and test file from tsv with separator \t
train= pd.read_csv("train.tsv", sep="\t")
test= pd.read_csv("test.tsv", sep="\t")

#Converting the reviews into a count matrix
#The feature is Bag of Words with tokenization, removing tag </br>, n gram, mark_negation
count_vector = CountVectorizer(analyzer="word",                                         #prerequisite from preprocessor function
                            tokenizer=lambda text: mark_negation(word_tokenize(text)),  #to override  the tokenization with marking negation
                            preprocessor=lambda text: text.replace("<br />", " "),      #to override the preprocessing with replacing tag br with empty character
                            ngram_range=(1, 3),                                         #obtaining the combination of term as unigram, bigram and trigram
                            )

#Fit to data and transform it
train_counts = count_vector.fit_transform(train['Phrase'])

#Setting tf (without idf) and learn the idf vector
tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)

#Transforming a count matrix to a tf representation
train_tf = tf_transformer.transform(train_counts)

tfidf_transformer = TfidfTransformer()

#Fit to data and transform it
train_tfidf = tfidf_transformer.fit_transform(train_counts)

#Using Linear Support Vector Classification classifier for fit to data then transform it
classifier = svm.LinearSVC().fit(train_tfidf, train['Sentiment'])

#Fit to data and transform it
test_counts = count_vector.transform(test['Phrase'])

#Transforming a count matrix to a tf representation
test_tfidf = tfidf_transformer.transform(test_counts)

#Predicting with a classifier to test set
predicted = classifier.predict(test_tfidf)

#Creating a csv file with column name PhraseId and Sentiment
with open('ce807_assignment1_rabia_yasakostas_and_henny_purnomo.csv', 'w') as csvfile:
    csvfile.write('PhraseId,Sentiment\n')
    for i,j in zip(test['PhraseId'], predicted):
        csvfile.write('{},{}\n'.format(i,j))