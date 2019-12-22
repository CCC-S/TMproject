from utils import *

from gensim.models import Doc2Vec

# classifier
from hpForClf import *

if __name__ == '__main__':

    #-----Prepare Data----#
    trainDataDirectory = './data/downloaded'
    testDataDirectory = './data/gold'

    trainPos, trainNeg, trainNeu = generateDataFiles(trainDataDirectory, "train")
    testPos, testNeg, testNeu = generateDataFiles(testDataDirectory, "test")
    # this file list specifies the training set
    # remove 'test' here to avoid overfitting/cheating
    sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-neu.txt':'TRAIN_NEU', 'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'test-neu.txt':'TEST_NEU'}

    sentences = LabeledLineSentence(sources)

    #----Training & Testing-----#
    #for epoch in range(10,210,10):


    #hyperoptForDoc2vec(sentences, trainPos, trainNeg, trainNeu, epoch=10)

    #---Train Doc2vec---#
    #model = Doc2Vec(min_count=1, window=8, vector_size=100, sample=1e-4, negative=5, workers=7, dm=0)
    bestDoc2vecParams = {'min_count': 5, 'negative': 4, 'sample': 0.02393271768564099, 'vector_size': 677, 'window': 7}
    model = Doc2Vec(**bestDoc2vecParams, workers=10, dm=0)

    print("Building Vocabulary...")
    modelName = "BestDoc2vecdm=0"
    print(modelName)
    model = buildVocab(model, sentences, modelName, epoch=40)
    #model = Doc2Vec.load('/vol/share/groups/liacs/scratch/s2434792/model/%s.d2v'%modelName)
    print("Building Vocabulary Done")

    #---Train Classifiers---#

    #apply train data to cv and hyperopt to find the best hyperparameter settings
    train_arrays, train_labels = prepareTrainData(model, trainPos, trainNeg, trainNeu, bestDoc2vecParams['vector_size'])

    # print(objective(LogisticRegression(), train_arrays, train_labels))
    # print(objective(SGDClassifier(), train_arrays, train_labels))
    # print(objective(LinearSVC(), train_arrays, train_labels))

    # print("Tuning Hyperparameter...")
    # bestHyperParam = hyperoptForLSVC(train_arrays, train_labels)
    # #hpskForClf(train_arrays, train_labels)
    # print("Tuning Hyperparameter Done")

    #---Re-train the best classifier---#
    bestLRParams = {'C': 2.530730070514327, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1.7350620297910828, 'l1_ratio': 0.38330717450944984, 'max_iter': 751.4487235546252, 'multi_class': 'auto', 'penalty': 'none', 'solver': 'sag', 'tol': 0.012019187630311415, 'warm_start': True}
    clf = LogisticRegression(**bestLRParams)

    print("Re-training Classifier")
    clf = trainClf(clf, train_arrays, train_labels)
    print("Re-training Classifier Done")

    #-----final test-----#

    # train embedding model for test set
    # sourcesTest = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'test-neu.txt':'TEST_NEU'}
    # sentencesTest = LabeledLineSentence(sourcesTest)
    # modelTest = Doc2Vec(**bestDoc2vecParams, workers=10)
    # print("Building Vocabulary...")
    # modelName = "BestDoc2vecTest"
    # print(modelName)
    # modelTest = buildVocab(modelTest, sentencesTest, modelName, epoch=40)
    # print("Building Done")

    test_arrays, test_labels = prepareTestData(model, testPos, testNeg, testNeu, bestDoc2vecParams['vector_size'])

    testClf(clf, test_arrays, test_labels)
