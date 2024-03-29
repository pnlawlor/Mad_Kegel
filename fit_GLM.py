#===============================================================================
# Import Stuff
#===============================================================================
from os import chdir
chdir("C:/Users/pnlawlor/Google Drive/Research/Projects/Mad_Kegel")
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LinearRegression as linreg
from sklearn.cross_validation import cross_val_score as cv_score
from sklearn.cross_validation import StratifiedKFold as SKFold
from sklearn.cross_validation import LeaveOneLabelOut as LOLO
from sklearn.cross_validation import KFold as KFold
from sklearn.grid_search import GridSearchCV as GSCV
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plot
from scipy.stats import logistic
import sklearn.preprocessing as pp
from sklearn.linear_model import LinearRegression as LinR
from sklearn.linear_model import ElasticNetCV as ENCV
from sklearn.linear_model import ElasticNet as EN
from scipy.stats import logistic

#===============================================================================
# Fit logistic GLM
#===============================================================================
def fit_logistic_GLM(X, y, 
                     C_value = np.array([-4,5]), 
                     num_cv = 5, 
                     verbose = False, 
                     intercept_scaling = 10,
                     penalty = 'l1',
                     reg_strength = None,
                     plot_results = False
                     ):
    scores_to_return = []
    loss_score = []
    X = pp.scale(X)
    # If regularization strength isn't specified, CV to find it
    if reg_strength == None:
        kf = SKFold(y = y, n_folds = num_cv)
        C_values = np.logspace(C_value[0], C_value[1], 10)
        C_dict = {"C": C_values}
        best_param = []
    #------------------------------------------------------------------------------ 
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            # Do grid search for regularization parameter
            clf = GSCV(
                        LR(C=1, penalty=penalty, dual=False,intercept_scaling=intercept_scaling),
                        C_dict,
                        cv=num_cv
                        )
            # Fit model
            clf.fit(X_train,y_train)
            best_param.append(clf.best_params_['C'])
            if verbose:
                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
        if verbose:
            print np.mean(np.asarray(scores))
        reg_strength = gmean(best_param)
#------------------------------------------------------------------------------ 
    kf2 = SKFold(y = y, n_folds = num_cv)
    clf = []
    clf_temp = LR(
             penalty=penalty, 
             dual=False,
             C = reg_strength,
             intercept_scaling = intercept_scaling
             )
    for train, test in kf2:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf_temp.fit(X_train, y_train)
        scores_to_return.append(clf_temp.score(X_test, y_test))
        clf.append(clf_temp)
        pred = clf_temp.predict_proba(X_test)[:,1]
        loss_score.append(lossFx(y_test,pred))
#------------------------------------------------------------------------------ 
    # Plot results
    if plot_results:
        plot_logistic_fit(clf,X,kf2)
    # Returns model, scores of each CV, best C parameter, CV fold indices
    return clf, scores_to_return, loss_score, reg_strength, kf2
        
#===============================================================================
# Plot fit and data with logistic GLM
#===============================================================================
def plot_logistic_fit(models, data, CV_info,num_columns = 2):
    num_cv = CV_info.n_folds
    num_rows = int(np.ceil(float(num_cv)/float(num_columns)))
    fig_temp = plot.subplots(nrows=num_rows, ncols=num_columns)
    fig = fig_temp[0]
    fig.tight_layout()
    axes = fig_temp[1]
    cv = 0
    #
    for train,test in CV_info:
        row_n = int(np.ceil(cv/num_columns))
        col_n = int(np.mod(float(cv),float(num_columns)))
        axes[row_n,col_n].set_title('CV fold %i' % (cv+1))
        intercept = models[cv].intercept_
        parameters = np.squeeze(np.asarray(models[cv].coef_))
#------------------------------------------------------------------------------ 
        # For plotting data along collapsed dimension
        collapsed_x_data = intercept + np.dot(parameters,data[test].transpose())
        y_data = models[cv].predict(data[test])
        y_data = np.asarray(y_data)
        axes[row_n,col_n].scatter(collapsed_x_data,y_data)
#------------------------------------------------------------------------------ 
        # For plotting function
        x_func = np.linspace(np.min(collapsed_x_data),np.max(collapsed_x_data),100)
        y_func = logistic.cdf(x_func)
        axes[row_n,col_n].plot(x_func,y_func)
#------------------------------------------------------------------------------ 
        cv += 1
#------------------------------------------------------------------------------ 
    plot.show()
    
#===============================================================================
# Random forests
#===============================================================================
def fit_RF(X,y,
           num_estimators = None, 
           verbose = False,
           plot_importance = False,
           num_cv = 5):
    X = pp.scale(X)
    # If num_estimators not provided, CV to estimate it
    if num_estimators == None:
        kf = SKFold(y = y, n_folds = num_cv)
        estimator_nums = np.logspace(0, np.size(X,axis=1), 10)
        estimator_nums = estimator_nums.astype(int)
        est_dict = {"n_estimators": estimator_nums}
        best_param = []
    #------------------------------------------------------------------------------ 
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            # Do grid search for num_estimators parameter
            clf = GSCV(
                        RF(n_estimators=1),
                        est_dict,
                        cv=num_cv
                        )
            # Fit model
            clf.fit(X_train,y_train)
            best_param.append(clf.best_params_['n_estimators'])
            if verbose:
                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
        if verbose:
            print np.mean(np.asarray(scores))
        num_estimators = gmean(best_param)
#------------------------------------------------------------------------------ 
    # Measure accuracy
    kf2 = SKFold(y = y, n_folds = num_cv)
    clf = []
    accuracy = []
    importances = []
    loss_score = []
    clf_temp = RF(n_estimators = num_estimators, 
             n_jobs = 1, 
             verbose = verbose)#,
#              compute_importances = True)
    for train, test in kf2:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf_temp.fit(X_train,y_train)
        clf.append(clf_temp)
        pred = clf_temp.predict_proba(X_test)[:,1]
        loss_score.append(lossFx(y_test,pred))
        accuracy.append(clf_temp.score(X_test, y_test))
        importances.append(clf_temp.feature_importances_)
        std = np.std([tree.feature_importances_ for tree in clf_temp.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
#------------------------------------------------------------------------------ 
        if verbose:
            print("Feature ranking:")
            for f in range(5):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#------------------------------------------------------------------------------ 
        if plot_importance:
            plot.figure()
            plot.title("Feature importances")
            plot.bar(range(10), importances[indices],
                   color="r", yerr=std[indices], align="center")
            plot.xticks(range(10), indices)
            plot.xlim([-1, 10])
            plot.show()
    return clf, accuracy, loss_score, num_estimators, kf2
    
def fit_linear_model(X, y, results, keys,
                     alpha = np.logspace(-5,2,50),
                     l1_ratio = np.array([.1, .5, .7, .9, .95, .99, 1]),
                     num_cv = 5, 
                     verbose = False, 
                     intercept_scaling = 10,
                     plot_results = False, 
                     labels = None
                     ):
    X = pp.scale(X)
    clf = []
    R2 = []
    coef = []
    prob = []
    score = []
    group_keys = []
    if num_cv > 1:
        num_cv2 = num_cv
    else:
        num_cv2 = 10
    # Find best alpha and lambda
    if (np.size(alpha)>1) or (np.size(l1_ratio)>1):
        print "Determining best values for L1 ratio and alpha..."
        clf_temp = ENCV(
                    l1_ratio = l1_ratio,
                    alphas = alpha,
                    cv = num_cv2,
                    fit_intercept = False,
                    verbose = verbose
                    )
        clf_temp.fit(X,y)
        best_alpha = clf_temp.alpha_
        best_l1_ratio = clf_temp.l1_ratio_
        print "Best L1 ratio: " + str(best_l1_ratio) + ", best alpha: " + str(best_alpha)
    else:
        best_alpha = alpha
        best_l1_ratio = l1_ratio
    # Now do cross-validation to estimate accuracy
    if num_cv > 1:
        if labels == None:
            kf = KFold(n = len(y), n_folds = num_cv)
        else:
            kf = LOLO(labels)
        #
        for train, test in kf:
            X_train, X_test, y_train, y_test, results_test, keys_test = X[train], X[test], y[train], y[test], results[test], keys[test]
            clf_temp2 = EN(
                            l1_ratio = best_l1_ratio,
                            alpha = best_alpha,
                            fit_intercept = False)
            clf_temp2.fit(X_train,y_train)
            pred = clf_temp2.predict(X_test)
            clf.append(clf_temp2)
            R2.append(clf_temp2.score(X_test,y_test))
            coef.append(clf_temp2.coef_)
            prob.append(diff_to_prob(pred))
            score.append(lossFx(results_test,pred))
            group_keys.append(keys_test)
    else:
        clf_temp2 = EN(
                l1_ratio = best_l1_ratio,
                alpha = best_alpha,
                fit_intercept = False)
        clf_temp2.fit(X,y)
        pred = clf_temp2.predict(X)
        clf = clf_temp2
        R2 = clf_temp2.score(X,y)
        coef = clf_temp2.coef_
        prob = diff_to_prob(pred)
        score = lossFx(results,pred)
        group_keys = keys
    if num_cv > 1:
        return clf, R2, score, coef, prob, kf, group_keys
    else:
        return clf, R2, score, coef, prob, group_keys
        
def fit_linear_model2(X, y, results, keys,
                     num_cv = 5, 
                     verbose = False, 
                     plot_results = False
                     ):
    X = pp.scale(X)
    clf = []
    R2 = []
    coef = []
    prob = []
    score = []
    group_keys = []
    # Now do cross-validation to estimate accuracy
    if num_cv > 1:
        kf = KFold(n = len(y), n_folds = num_cv)
        for train, test in kf:
            X_train, X_test, y_train, y_test, results_test, keys_test = X[train], X[test], y[train], y[test], results[test], keys[test]
            clf_temp2 = linreg(
                            fit_intercept = False)
            clf_temp2.fit(X_train,y_train)
            pred = clf_temp2.predict(X_test)
            clf.append(clf_temp2)
            R2.append(clf_temp2.score(X_test,y_test))
            coef.append(clf_temp2.coef_)
            prob.append(diff_to_prob(pred))
            score.append(lossFx(results_test,pred))
            group_keys.append(keys_test)
    else:
        clf_temp2 = linreg(
                fit_intercept = False)
        clf_temp2.fit(X,y)
        pred = clf_temp2.predict(X)
        clf = clf_temp2
        R2 = clf_temp2.score(X,y)
        coef = clf_temp2.coef_
        prob = diff_to_prob(pred)
        score = lossFx(results,pred)
        group_keys = keys
    if num_cv > 1:
        return clf, R2, score, coef, prob, kf, group_keys
    else:
        return clf, R2, score, coef, prob, group_keys
    
# Convert point differential to probability of winning
def diff_to_prob(
                differentials,
                sigmoid_mean = 0,
                sigmoid_slope = .19
                ):
    # Ted's slope is 1/scale
    prob = logistic.cdf(differentials,loc=sigmoid_mean,scale=1/sigmoid_slope)
    return prob
    
# Loss function actually used in the Kaggle    
def lossFx(yTrue, scoreDiff, k=.19):    
    pProb = diff_to_prob(scoreDiff, sigmoid_slope = k)
    pPred = scoreDiff>0
    pPred = pPred.astype(int)
#    
    loss = -np.mean(yTrue*np.log(pProb)+(1-yTrue)*np.log(1-pProb))
    return loss
    
# Calculate score
def loss_score(yTrue,yPred):
    loss = -np.mean(yTrue*np.log(yPred)+(1-yTrue)*np.log(1-yPred))
    return loss
    
# Hybrid approach for training on both historical tournament data as well as regular season data
    