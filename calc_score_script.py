# -*- coding: utf-8 -*-

from os import chdir
chdir("C:/Users/pnlawlor/Google Drive/Research/Projects/Mad_Kegel")
import numpy as np
from fit_GLM import loss_score
import pandas 
from Tkinter import Tk
from tkFileDialog import askopenfilename as uiopen
import sklearn.preprocessing as pp
import cPickle as p
from scipy.stats import logistic
import scikits.bootstrap as boot

load_filename = 'C:\Users\pnlawlor\Dropbox\Data\comb_pracPredsResults.csv'
temp = pandas.io.parsers.read_csv(load_filename,header=None,index_col=0)

completed_games = ~temp[3].isnull()

results = temp.ix[completed_games,3]
conservative_preds = temp.ix[completed_games,1]
aggressive_preds = temp.ix[completed_games,2]

conservative_score = loss_score(results,conservative_preds)
#conservative_score_CI = boot.ci([results, conservative_preds],loss_score)
aggressive_score = loss_score(results,aggressive_preds)


print 'Conservative score: %s' % conservative_score
print 'Aggressive score: %s' % aggressive_score

simulate = False

def boot_sim(results,preds,nboot,acc):
    loss = []    
    for nb in range(nboot):
        n_shuffle = np.round(len(results)*(1-acc)).astype('int')
        np.random.shuffle(results)
        results = results.astype('bool')
#        results[:n_shuffle] = ~results[:n_shuffle]
        results[(len(results)-np.ceil(n_shuffle/2).astype('int')):(len(results)+np.ceil(n_shuffle/2).astype('int'))] = ~results[(len(results)-np.ceil(n_shuffle/2).astype('int')):(len(results)+np.ceil(n_shuffle/2).astype('int'))]
        results = results.astype('int')
        loss.append(loss_score(results,preds))
    return np.array(loss)
    
if simulate:
    acc = .75
    fakePreds = np.random.beta(2,2,1000)
    fakeResults = np.round(fakePreds)
    sim_losses = boot_sim(fakeResults,fakePreds,100,acc)
    print 'Simulation for %f accuracy...' % acc
    print 'Average loss: %f [%f %f]' % (np.median(sim_losses),np.percentile(sim_losses,5),np.percentile(sim_losses,95))