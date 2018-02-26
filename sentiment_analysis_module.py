import nltk
import random
# from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
from nltk.classify import ClassifierI
from nltk import sent_tokenize , word_tokenize, PunktSentenceTokenizer

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

review_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/reviews.pickle","rb")
review = pickle.load(review_file)
review_file.close()

word_feature_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/word_feature_set.pickle","rb")
word_features = pickle.load(word_feature_file)
word_feature_file.close()

def feature_generation(review):
    words = set(review)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print(feature_generation(movie_reviews.words('pos/cv000_29590.txt')))

featureSets = [(feature_generation(x), category) for (x, category) in review]

# featureSets_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/feature_set.pickle","rb")
# featureSets = pickle.load(featureSets_file)
# featureSets_file.close()

random.shuffle(featureSets)

training_set = featureSets[:10000]
testing_set = featureSets[10000:]

open_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/NB_classifier.pickle","rb")
naive_bayes_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

vote_classification = VoteClassifier(naive_bayes_classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,LinearSVC_classifier)

def sentiment_analysed(text):
    features = feature_generation(text)
    return vote_classification.classify(features), vote_classification.confidence(features)

print(sentiment_analysed("this movie was bad"))

