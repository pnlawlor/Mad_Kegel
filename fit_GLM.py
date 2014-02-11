#===============================================================================
# Import Stuff
#===============================================================================
from os import chdir
chdir("C:/Users/pnlawlor/Google Drive/Research/Projects/Mad_Kegel")
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cross_validation import cross_val_score as cv_score
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.grid_search import GridSearchCV as GSCV
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plot
from scipy.stats import logistic
import sklearn.preprocessing as pp

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
    X = pp.scale(X)
    # If regularization strength isn't specified, CV to find it
    if reg_strength == None:
        kf = KFold(y = y, n_folds = num_cv)
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
    kf2 = KFold(y = y, n_folds = num_cv)
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
#------------------------------------------------------------------------------ 
    # Plot results
    if plot_results:
        plot_logistic_fit(clf,X,kf2)
    # Returns model, scores of each CV, best C parameter, CV fold indices
    return clf, scores_to_return, reg_strength, kf2
        
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
        kf = KFold(y = y, n_folds = num_cv)
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
    kf2 = KFold(y = y, n_folds = num_cv)
    clf = []
    accuracy = []
    importances = []
    clf_temp = RF(n_estimators = num_estimators, 
             n_jobs = 1, 
             verbose = verbose)#,
#              compute_importances = True)
    for train, test in kf2:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf_temp.fit(X_train,y_train)
        clf.append(clf_temp)
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
    return clf, accuracy, num_estimators, kf2
    