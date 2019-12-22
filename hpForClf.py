from utils import *


'''
return the best hyperparameter settings
'''
def hyperoptForSGD(train_arrays, train_labels):

    def f(params):
        acc = objective(SGDClassifier(**params), train_arrays, train_labels)
        return {'loss': -acc, 'status': STATUS_OK}

    def convert_dic(sgdbest):
        if("learning_rate" in sgdbest):
            assert sgdbest["learning_rate"] < 4
            if(sgdbest["learning_rate"]==0):
                sgdbest["learning_rate"]="constant"
            elif(sgdbest["learning_rate"]==1):
                sgdbest["learning_rate"]="optimal"
            elif(sgdbest["learning_rate"]==2):
                sgdbest["learning_rate"]="invscaling"
            elif(sgdbest["learning_rate"]==3):
                sgdbest["learning_rate"]="adaptive"
        if("penalty" in sgdbest):
            assert sgdbest["penalty"] < 4
            if(sgdbest["penalty"]==0):
                sgdbest["penalty"]="l1"
            elif(sgdbest["penalty"]==1):
                sgdbest["penalty"]="l2"
            elif (sgdbest["penalty"] == 2):
                sgdbest["penalty"] = "elasticnet"
            elif (sgdbest["penalty"] == 3):
                sgdbest["penalty"] = "none"
        if("warm_start" in sgdbest):
            if(sgdbest["warm_start"]>0):
                sgdbest["warm_start"] = False
            else:
                sgdbest["warm_start"] = True
        if ("fit_intercept" in sgdbest):
            if (sgdbest["fit_intercept"] > 0):
                sgdbest["fit_intercept"] = False
            else:
                sgdbest["fit_intercept"] = True
        if ("early_stopping" in sgdbest):
            if (sgdbest["early_stopping"] > 0):
                sgdbest["early_stopping"] = False
            else:
                sgdbest["early_stopping"] = True
        return sgdbest

    space4sgd = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']),
    'alpha': hp.uniform('alpha', 0.00001, 0.1),
    'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
    'eta0': hp.uniform('eta0', 0.00001, 0.1),
    'power_t': hp.uniform('power_t', 0.3, 0.7),
    'warm_start': hp.choice('warm_start', [True, False]),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'max_iter': hp.uniform('max_iter', 10, 1000),
    'tol': hp.uniform('tol', 0.00001, 0.1),
    'early_stopping': hp.choice('early_stopping', [True, False]),
    'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.9)
    }

    print("hyperopt strat...")
    trials = Trials()
    best = fmin(f, space4sgd, algo=tpe.suggest, max_evals=100, trials=trials)
    print("hyperopt finish...")
    bestConverted = convert_dic(best)
    print("best hyperparameter:\n%s"%bestConverted)
    scores = [-trial['result']['loss'] for trial in trials.trials]
    maxScore = max(scores)
    print("best score: %s"%maxScore)

    return bestConverted

def hyperoptForLR(train_arrays, train_labels):

    '''
    all those if condition try to avoid parameter confliction
    '''
    def objectiveLR(params, clf, X_, y):
        if params['penalty'] == 'l2':
            if params['dual'] is True:
                if params['solver'] == 'liblinear':
                    if params['multi_class'] == 'multinomial':
                        return 0.001
                    else:
                        return cross_val_score(clf, X_, y, cv=5).mean()
                else:
                    return 0.001
            else:
                if params['solver'] == 'liblinear' and params['multi_class'] == 'multinomial':
                    return 0.001
                else:
                    return cross_val_score(clf, X_, y, cv=5).mean()
        elif params['penalty'] == 'l1':
            if params['dual'] is True:
                return 0.001
            else:
                if params['solver'] == 'liblinear':
                    if params['multi_class'] == 'multinomial':
                        return 0.001
                    else:
                        return cross_val_score(clf, X_, y, cv=5).mean()
                elif params['solver'] == 'saga':
                    return cross_val_score(clf, X_, y, cv=5).mean()
                else:
                    return 0.001
        elif params['penalty'] == 'elasticnet':
            if params['dual'] is True:
                return 0.001
            else:
                if params['solver'] == 'saga':
                    return cross_val_score(clf, X_, y, cv=5).mean()
                else:
                    return 0.001
        elif params['penalty'] == 'none':
            if params['dual'] is True:
                return 0.001
            else:
                if params['solver'] == 'liblinear':
                    return 0.001
                else:
                    return cross_val_score(clf, X_, y, cv=5).mean()
        else:
            return cross_val_score(clf, X_, y, cv=5).mean()



    def f(params):
        acc = objectiveLR(params, LogisticRegression(**params), train_arrays, train_labels)
        return {'loss': -acc, 'status': STATUS_OK}

    def convert_dic(lrbest):
        if("penalty" in lrbest):
            assert lrbest["penalty"] < 4
            if(lrbest["penalty"]==0):
                lrbest["penalty"]="l1"
            elif(lrbest["penalty"]==1):
                lrbest["penalty"]="l2"
            elif (lrbest["penalty"] == 2):
                lrbest["penalty"] = "elasticnet"
            elif (lrbest["penalty"] == 3):
                lrbest["penalty"] = "none"
        if("warm_start" in lrbest):
            if(lrbest["warm_start"]>0):
                lrbest["warm_start"] = False
            else:
                lrbest["warm_start"] = True
        if ("dual" in lrbest):
            if (lrbest["dual"] > 0):
                lrbest["dual"] = False
            else:
                lrbest["dual"] = True
        if ("fit_intercept" in lrbest):
            if (lrbest["fit_intercept"] > 0):
                lrbest["fit_intercept"] = False
            else:
                lrbest["fit_intercept"] = True
        if("solver" in lrbest):
            assert lrbest["solver"] < 5
            if(lrbest["solver"] == 0):
                lrbest["solver"] = "newton-cg"
            elif(lrbest["solver"] == 1):
                lrbest["solver"] = "lbfgs"
            elif (lrbest["solver"] == 2):
                lrbest["solver"] = "liblinear"
            elif (lrbest["solver"] == 3):
                lrbest["solver"] = "sag"
            elif (lrbest["solver"] == 4):
                lrbest["solver"] = "saga"
        if("multi_class" in lrbest):
            assert lrbest["multi_class"] < 4
            if(lrbest["multi_class"] == 0):
                lrbest["multi_class"] = "ovr"
            elif(lrbest["multi_class"] == 1):
                lrbest["multi_class"] = "multinomial"
            elif (lrbest["multi_class"] == 2):
                lrbest["multi_class"] = "auto"
        return lrbest

    space4lr = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']),
    'dual': hp.choice('dual', [True, False]),
    'tol': hp.uniform('tol', 0.00001, 0.1),
    'C': hp.uniform('C', 0, 5),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'intercept_scaling': hp.uniform('intercept_scaling', 0, 5),
    'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
    'max_iter': hp.uniform('max_iter', 10, 1000),
    'multi_class': hp.choice('multi_class', ['ovr', 'multinomial', 'auto']),
    'warm_start': hp.choice('warm_start', [True, False]),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1)
}

    print("hyperopt strat...")
    trials = Trials()
    best = fmin(f, space4lr, algo=tpe.suggest, max_evals=100, trials=trials)
    print("hyperopt finish...")
    bestConverted = convert_dic(best)
    print("best hyperparameter:\n%s"%bestConverted)
    scores = [-trial['result']['loss'] for trial in trials.trials]
    maxScore = max(scores)
    print("best score: %s"%maxScore)

    return bestConverted

def hyperoptForKNN(train_arrays, train_labels):

    def f(params):
        acc = objective(KNeighborsClassifier(**params), train_arrays, train_labels)
        return {'loss': -acc, 'status': STATUS_OK}

    def convert_dic(knnbest):
        if("weights" in knnbest):
            assert knnbest["weights"] < 2
            if(knnbest["weights"] == 0):
                knnbest["weights"] = "uniform"
            elif(knnbest["weights"] == 1):
                knnbest["weights"] = "distance"
        return knnbest


    space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 20)),
    'weights': hp.choice('weights', ['uniform', 'distance']),
    'p': hp.uniform('p', 1, 5)
}

    print("hyperopt strat...")
    trials = Trials()
    best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
    print("hyperopt finish...")
    bestConverted = convert_dic(best)
    print("best hyperparameter:\n%s"%bestConverted)
    scores = [-trial['result']['loss'] for trial in trials.trials]
    maxScore = max(scores)
    print("best score: %s"%maxScore)

    return bestConverted

def hyperoptForLSVC(train_arrays, train_labels):

    def f(params):
        acc = objective(LinearSVC(**params), train_arrays, train_labels)
        return {'loss': -acc, 'status': STATUS_OK}

    def convert_dic(lsvcbest):
        if ("dual" in lsvcbest):
            if (lsvcbest["dual"] > 0):
                lsvcbest["dual"] = False
            else:
                lsvcbest["dual"] = True
        if ("fit_intercept" in lsvcbest):
            if (lsvcbest["fit_intercept"] > 0):
                lsvcbest["fit_intercept"] = False
            else:
                lsvcbest["fit_intercept"] = True
        if("multi_class" in lsvcbest):
            assert lsvcbest["multi_class"] < 4
            if(lsvcbest["multi_class"] == 0):
                lsvcbest["multi_class"] = "ovr"
            elif(lsvcbest["multi_class"] == 1):
                lsvcbest["multi_class"] = "crammer_singer"
        return lsvcbest

    space4lsvc = {
    'dual': hp.choice('dual', [True, False]),
    'tol': hp.uniform('tol', 0.00001, 0.1),
    'C': hp.uniform('C', 0, 5),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'intercept_scaling': hp.uniform('intercept_scaling', 0, 5),
    'max_iter': hp.uniform('max_iter', 10, 1000),
    'multi_class': hp.choice('multi_class', ['ovr', 'crammer_singer']),
}

    print("hyperopt strat...")
    trials = Trials()
    best = fmin(f, space4lsvc, algo=tpe.suggest, max_evals=100, trials=trials)
    print("hyperopt finish...")
    bestConverted = convert_dic(best)
    print("best hyperparameter:\n%s"%bestConverted)
    scores = [-trial['result']['loss'] for trial in trials.trials]
    maxScore = max(scores)
    print("best score: %s"%maxScore)

    return bestConverted
