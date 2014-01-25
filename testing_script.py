from os import chdir
chdir("C:/Users/pnlawlor/Google Drive/Research/Projects/Mad_Kegel")
import numpy as np
from fit_GLM import fit_logistic_GLM, plot_logistic_fit, fit_RF


X_data = np.random.rand(200,200)
# X_data = X_data.ravel()
y_data2 = np.round(X_data[:,0])
y_data = y_data2.ravel()

# models, scores, C, kf = fit_logistic_GLM(X_data, y_data,num_cv=5,verbose=True,plot_results=True)

models, scores, num_estimators, kf = fit_RF(X_data,y_data,num_estimators=10)