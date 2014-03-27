from os import chdir
chdir("C:/Users/pnlawlor/Google Drive/Research/Projects/Mad_Kegel")
import numpy as np
from fit_GLM import fit_logistic_GLM, plot_logistic_fit, fit_RF, fit_linear_model, fit_linear_model2
import pandas
from Tkinter import Tk
from tkFileDialog import askopenfilename as uiopen
import sklearn.preprocessing as pp
import cPickle as p
from scipy.stats import logistic

# import data from CSV
#Tk().withdraw() 

#load_filename = uiopen()
#load_filename = 'C:\Users\pnlawlor\Dropbox\Data\\features_D.csv'

#load_filename = 'C:\Users\pnlawlor\Dropbox\Data\\features_all.csv'
load_filename = 'C:\Users\pnlawlor\Dropbox\Data\\tourneyFeatures_all.csv'
temp = pandas.io.parsers.read_csv(load_filename,header=None,index_col=0)
keys = temp.index
season_labels = np.array([keys[i][0] for i in range(len(keys))])

# Select individual seasons by letter
#data_points = np.any([[season_labels == 'J'],
#                                            ],axis=0).ravel() # always have to flatten indices...

# For selecting all seasons
X_data = temp.values[:,0:-1].astype(float) # UP TO BUT NOT INCLUDING
y_data = temp.values[:,-1].astype(float)

# Add sigmoid(seed1-seed2)
#seed_diff = np.atleast_2d((X_data[:,20] - X_data[:,21])).T # unravel and calc seed_diff
#sigm1 = 30*logistic.cdf(seed_diff,loc=0,scale=8) - 15
#sigm2 = 30*logistic.cdf(seed_diff,loc=0,scale=5) - 15
#sigm3 = 30*logistic.cdf(seed_diff,loc=0,scale=3) - 15
#sigm4 = 30*logistic.cdf(seed_diff,loc=0,scale=2) - 15
#X_data = np.concatenate((X_data, seed_diff, sigm2),axis=1)

# For selecting some seasons
#X_data = temp.values[data_points,0:-1].astype(float) # UP TO BUT NOT INCLUDING
#y_data = temp.values[data_points,-1].astype(float)
#season_labels = season_labels[data_points]

results = (.5*(np.sign(y_data)+1)).astype(int)
X_data = pp.scale(X_data)

#models3, R2_3, loss_scores3, coef3, prob3, kf3, group_keys3 = fit_linear_model(
#                                                                X_data,y_data,
#                                                                results,keys,
#                                                                labels=season_labels)

#models3, R2_3, loss_scores3, coef3, prob3, kf3, group_keys3 = fit_linear_model(
#                                                                X_data,y_data,
#                                                                results,keys,num_cv=10)

models3, R2_3, loss_scores3, coef3, prob3, group_keys3 = fit_linear_model(
                                                                X_data,y_data,
                                                                results,keys,num_cv=1)


# first 1/3 win pct, 
# mid 1/3 win pct, 
# last 1/3 win pct, 
# last 1/6 win pct, 
# away win pct, 
# ppg, 
# ppg against, 
# avg pt differential, 
# stdv points for*losing team, 
# stdv points against*losing team

save_model = False

# Save model
if save_model:
#    save_filename = 'C:\Users\pnlawlor\Dropbox\Data\\all_seasons_model.pickle'
    save_filename = 'C:\Users\pnlawlor\Dropbox\Data\\all_seasons_tourney_model.pickle'
    with open(save_filename, 'wb') as f:
        p.dump(models3,f)
    with open(save_filename, 'rb') as f:
        model3_loaded = p.load(f)        


# Old stuff

#models, scores, loss_scores, C, kf = fit_logistic_GLM(X_data, results,num_cv=5,verbose=True,plot_results=False)
##
#models2, scores2, loss_scores2, num_estimators2, kf2 = fit_RF(X_data,results,num_estimators=1000,num_cv=5)
#
#models4, R2_4, loss_scores4, coef4, prob4, group_keys4, kf4 = fit_linear_model2(X_data,y_data,results,keys,num_cv=5)