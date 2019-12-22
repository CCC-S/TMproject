# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from smart_open import smart_open

# numpy
import numpy as np
import random, os, string, csv, time, re

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# hyperopt-sklearn
from hpsklearn import HyperoptEstimator, svc, knn, sgd

import matplotlib.pyplot as plt
from sklearn import metrics
from nltk.corpus import stopwords


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

def generateDataFiles(dataDirectory, setType):
    dataFiles = [fi for fi in os.listdir(dataDirectory) if "twitter" in fi]
    duplicate = 0
    pos, neg, neu = 0, 0, 0
    hashTag, atSign = 0, 0
    globalDict = {}
    with open('./%s-pos.txt'%setType, 'w', encoding="utf-8") as fpos, open('./%s-neg.txt'%setType, 'w', encoding="utf-8") as fneg, open('./%s-neu.txt'%setType, 'w', encoding="utf-8") as fneu:

        for oneFile in dataFiles:
            with open(os.path.join(dataDirectory ,oneFile), encoding='utf-8') as fd:
                rd = csv.reader(fd, delimiter="\t")
                for row in rd:
                    ID, polarity, content = row[:]

                    if ID in globalDict:
                        duplicate += 1
                        continue

                    if '#' in content:
                        hashTag += 1
                    if '@' in content:
                        atSign += 1

                    # remove @# and following username and hashtag
                    #content = re.sub('@\w+', '', content)
                    #content = re.sub('#\S+', '', content)


                    # remove punctuation
                    content = content.translate(str.maketrans("","", string.punctuation))

                    # remoce punctuation except @#
                    #newPun = ''.join([i for i in string.punctuation if i is not '@' and i is not '#'])
                    #content = content.translate(str.maketrans("","", newPun))

                    # lowercasing
                    content = content.lower()

                    # remove non-ascii characters
                    #content = ''.join([i if ord(i) < 128 else ' ' for i in content])

                    if polarity == 'positive':
                        fpos.write(content)
                        fpos.write("\n")
                        pos += 1
                    elif polarity == 'negative':
                        fneg.write(content)
                        fneg.write("\n")
                        neg += 1
                    elif polarity == "neutral":
                        fneu.write(content)
                        fneu.write("\n")
                        neu += 1
                    else:
                        print("something wrong")
                        print(polarity)
                        exit()

                    # store every tweet in a dict to detect duplicate tweets
                    globalDict[ID] = '1'

    print("duplicate items in %s set: %s"%(setType, duplicate))
    print("@: %s, #: %s"%(atSign, hashTag))
    return pos, neg, neu

def buildVocab(model, sentences, modelName, epoch):

    model.build_vocab(sentences.to_array())

    #------train-----#
    #for epoch in range(10):

    a = time.time()
    model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=epoch)
    print("Training Time: %s"%(time.time()- a))

    #model.save('./model/%s.d2v'%modelName)
    model.save('/vol/share/groups/liacs/scratch/s2434792/model/%s.d2v'%modelName)

    return model


def prepareTrainData(model, pos, neg, neu, feature_size=100):

    numAll = pos + neg + neu
    train_arrays = np.zeros((numAll, feature_size))
    train_labels = np.zeros(numAll)

    for i in range(pos):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model[prefix_train_pos]
        train_labels[i] = 1

    for i in range(pos, pos+neg):
        prefix_train_neg = 'TRAIN_NEG_' + str(i-pos)
        train_arrays[i] = model[prefix_train_neg]
        train_labels[i] = 2

    for i in range(pos+neg, numAll):
        prefix_train_neu = 'TRAIN_NEU_' + str(i-pos-neg)
        train_arrays[i] = model[prefix_train_neu]
        train_labels[i] = 3

    return train_arrays, train_labels

def trainClf(classifier, train_arrays, train_labels):

    #clfs = [LogisticRegression(), SVC(), KNeighborsClassifier()]
    #clfs = [LogisticRegression()]

    #for classifier in clfs:
    classifier.fit(train_arrays, train_labels)
    return classifier

def prepareTestData(model, pos, neg, neu, feature_size=100):

    numAll = pos + neg + neu
    test_arrays = np.zeros((numAll, feature_size))
    test_labels = np.zeros(numAll)

    for i in range(pos):
        prefix_test_pos = 'TEST_POS_' + str(i)
        test_arrays[i] = model[prefix_test_pos]
        test_labels[i] = 1

    for i in range(pos, pos+neg):
        prefix_test_neg = 'TEST_NEG_' + str(i-pos)
        test_arrays[i] = model[prefix_test_neg]
        test_labels[i] = 2

    for i in range(pos+neg, numAll):
        prefix_test_neu = 'TEST_NEU_' + str(i-pos-neg)
        test_arrays[i] = model[prefix_test_neu]
        test_labels[i] = 3

    return test_arrays, test_labels

def testClf(clf, test_arrays, test_labels):

    targetName = ['Pos', 'Neg', 'Neu']
    predicted = clf.predict(test_arrays)
    test_report = metrics.classification_report(test_labels, predicted, target_names=targetName, output_dict=True, digits=3)
    print(test_report)

    # this evaluation method is recommanded by the summary paper
    macroAvgRecall = test_report['macro avg']['recall']
    F1PN = (test_report['Pos']['f1-score'] + test_report['Neg']['f1-score']) / 2
    accuracy = test_report['accuracy']
    print("MacroAvgRecall: %.3f"%(macroAvgRecall))
    print("F1PN: %.3f"%(F1PN))
    print("Accuracy: %.3f"%(accuracy))


def objective(clf, train_arrays, train_labels):
        #clf = SGDClassifier(**params)
        return cross_val_score(clf, train_arrays, train_labels, cv=5, scoring='recall_macro').mean()

def hyperoptForDoc2vec(sentences, trainPos, trainNeg, trainNeu, epoch):

    def f(params):
        print(params)
        model = Doc2Vec(**params, workers=10)
        model = buildVocab(model, sentences, "hpForDoc10New_%.2f"%time.time(), epoch)
        train_arrays, train_labels = prepareTrainData(model, trainPos, trainNeg, trainNeu, feature_size=params['vector_size'])
        acc = objective(SGDClassifier(), train_arrays, train_labels)
        return {'loss': -acc, 'status': STATUS_OK}

    space4sgd = {
        'min_count': hp.choice('min_count', range(1, 10)),
        'window': hp.choice('window', range(1, 10)),
        'vector_size': hp.choice('vector_size', range(50, 800)),
        'sample': hp.uniform('sample', 1e-5, 1e-1),
        'negative': hp.choice('negative', range(5, 10)),
    }

    print("hyperopt strat...")
    trials = Trials()
    best = fmin(f, space4sgd, algo=tpe.suggest, max_evals=100, trials=trials)
    print("hyperopt finish...")
    bestConverted = best
    print("best hyperparameter:\n%s"%bestConverted)
    scores = [-trial['result']['loss'] for trial in trials.trials]
    maxScore = max(scores)
    print("best score: %s"%maxScore)

    return bestConverted

def hpskForClf(train_arrays, train_labels):
    estim = HyperoptEstimator(classifier=sgd('mySGD'),
                              preprocessing=[],
                              algo=tpe.suggest,
                              max_evals=5,)
    #cross_val_score(estim, train_arrays, train_labels, cv=5, scoring='recall_macro').mean()
    estim.fit(train_arrays, train_labels)
    print(estim.best_model())

