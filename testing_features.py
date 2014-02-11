from os import chdir
chdir("C:/Users/pnlawlor/Google Drive/Research/Projects/Mad_Kegel")
import numpy as np
from fit_GLM import fit_logistic_GLM, plot_logistic_fit, fit_RF
import pandas
from Tkinter import Tk
from tkFileDialog import askopenfilename as uiopen

# import data from CSV
Tk().withdraw() 
load_filename = uiopen()
temp = pandas.io.parsers.read_csv(load_filename)
X_data = temp.values[:-1,1:15].astype(float)
y_data = temp.values[:-1,-1].astype(bool)

models, scores, C, kf = fit_logistic_GLM(X_data, y_data,num_cv=10,verbose=True,plot_results=True)

models2, scores2, num_estimators2, kf2 = fit_RF(X_data,y_data,num_estimators=1000,num_cv=10)