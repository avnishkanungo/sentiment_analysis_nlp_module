import nltk
import random
from nltk.corpus import movie_reviews
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
    

review = [(list(movie_reviews.words(fileid)),category)
		  for category in movie_reviews.categories()
		  for fileid in movie_reviews.fileids(category)]

random.shuffle(review)
allowed_word_type = ["J"]
all_words = []



# req_data_tokenized = word_tokenize(req_data)

pos_req_data = nltk.pos_tag(review[1][0])

for w in pos_req_data:
	if w[1][0] in allowed_word_type:
		all_words.append(w[0].lower())

print(review[0][1])

# for w in movie_reviews.words():
# 	all_words.append(w.lower())
# pos_data = open('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/pos.txt',"r").read()
# neg_data = open('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/neg.txt',"r").read()

# review = []

# for r in pos_data.split('\n'):
# 	review.append((r,"pos"))

# for r in neg_data.split('\n'):
# 	review.append((r,"neg"))

# all_words = []
# pos_data_tokenize = word_tokenize(pos_data)
# neg_data_tokenize = word_tokenize(neg_data)

# allowed_word_type = ["J"]
# positive = nltk.pos_tag(pos_data_tokenize)
# negative = nltk.pos_tag(neg_data_tokenize)

# for w in positive:
# 	if w[1][0] in allowed_word_type:
# 		all_words.append(w[0].lower())

# for w in negative:
# 	if w[1][0] in allowed_word_type:
# 		all_words.append(w[0].lower())

save_documents = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/reviews.pickle","wb")
pickle.dump(review, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["inspired"])

word_features = list(all_words.keys())[:5000]

save_documents = open("/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/word_feature_set.pickle","wb")
pickle.dump(word_features, save_documents)
save_documents.close()

def feature_generation(review):
	words = set(review)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

# print(feature_generation(movie_reviews.words('pos/cv000_29590.txt')))

featureSets = [(feature_generation(x), category) for (x, category) in review]
random.shuffle(featureSets)

training_set = featureSets[:2900]
testing_set = featureSets[900:]

classifier  = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(classifier,testing_set))
classifier.show_most_informative_features(40)

save_classifier = open('/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/NB_classifier.pickle', 'wb')
pickle.dump(classifier,save_classifier)
save_classifier.close


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open('/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/MNB_classifier.pickle', 'wb')
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open('/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/BernoulliNB_classifier.pickle', 'wb')
pickle.dump(BernoulliNB_classifier,save_classifier)
save_classifier.close

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open('/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/LogisticRegression_classifier.pickle', 'wb')
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier = open('/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/SGDClassifier_classifier.pickle', 'wb')
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open('/Users/avnish/LearningNewstuff/Data_Analysis/sentiment_analysis_nlp/LinearSVC_classifier.pickle', 'wb')
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# vote_classification = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,LinearSVC_classifier)
# print("vote_classification accuracy percent:", (nltk.classify.accuracy(vote_classification, testing_set))*100)


# print("Classification:", vote_classification.classify(testing_set[0][0]), "Confidence %:",vote_classification.confidence(testing_set[0][0])*100)
# print("Classification:", vote_classification.classify(testing_set[1][0]), "Confidence %:",vote_classification.confidence(testing_set[1][0])*100)
# print("Classification:", vote_classification.classify(testing_set[2][0]), "Confidence %:",vote_classification.confidence(testing_set[2][0])*100)
# print("Classification:", vote_classification.classify(testing_set[3][0]), "Confidence %:",vote_classification.confidence(testing_set[3][0])*100)
# print("Classification:", vote_classification.classify(testing_set[4][0]), "Confidence %:",vote_classification.confidence(testing_set[4][0])*100)
# print("Classification:", vote_classification.classify(testing_set[5][0]), "Confidence %:",vote_classification.confidence(testing_set[5][0])*100)


