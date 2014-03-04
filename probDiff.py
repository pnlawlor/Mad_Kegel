import networkx as nx
import csv
from sets import Set
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# General Maintenance Functions

def loadResults(rfName):
    results = []
    with open(rfName,'rb') as ifile:
        f = csv.reader(ifile)
        header = f.next()
        for row in f:
            results.append(row)
    return results

def loadTeamNames(tfName):
    names = {}
    with open(tfName,'rb') as ifile:
        f = csv.reader(ifile)
        header = f.next()
        for row in f:
            names[row[0]] = row[1]
    return names

def selTeamGames(data,team):
    return [x for x in data if (x[2] == team or x[4] == team)]

def getTIDs(data):
    return list(Set([x[2] for x in data] + [x[4] for x in data]))

def flatten(l):
    return [x for v in l for x in v]
    
def getWL(data,t1,t2):
    games = selTeamGames(selTeamGames(data,t1),t2)
    nGames = len(games)
    nT1Win = 0
    if int(t1)<int(t2):
        for game in games:
            if game[2] == t1:
                nT1Win+=1
    else:
        print "teams in wrong order."
    if nGames == 0:
        return "NA"
    else:
        return nT1Win/float(nGames)
    
def getWLDiff(data,t1,t2):
    games = selTeamGames(selTeamGames(data,t1),t2)
    nGames = len(games)
    ptDiffT1 = 0
    if int(t1)<int(t2):
        for game in games:
            if game[2] == t1:
                ptDiffT1 += int(game[3]) - int(game[5])
            else:
                ptDiffT1 -= int(game[3]) - int(game[5])
    else:
        print "teams in wrong order."
    if nGames == 0:
        return "NA"
    else:
        return ptDiffT1/float(nGames)
    
def getMultiples(games,teams):
    nTeams = len(teams)
    pairs = {}
    for i in range(nTeams):
        print teams[i]
        for j in range(i+1,nTeams):
            t1 = teams[i]
            t2 = teams[j]
            pair = set([t1,t2])
            pairHandle = "{0}_{1}".format(t1,t2)
            for game in games:
                if pair == set([game[2],game[4]]):
                    if not pairs.has_key(pairHandle):
                        pairs[pairHandle] = {1:game}
                    else:
                        pairs[pairHandle][len(pairs[pairHandle].keys())+1] = game
    multiples = [xKey for xKey in pairs.keys() if len(pairs[xKey].keys()) > 1]
    print "Season has {0} pairs of teams that played multiple times.".format(len(multiples))
    multipleD = pairs
    for item in multipleD.keys():
        if item not in multiples:
            del multipleD[item]
    return multipleD
    
def genDiffPoints(pairs):
    results = []
    for pair in pairs:
        data = pairs[pair]
        teams = pair.split('_')
        diffs = np.array([int(x[3])-int(x[5]) for x in data.values()
                           if x[2] == teams[0]] + [int(x[5])-int(x[3]) 
                           for x in data.values() if x[4] == teams[0]])                
        print diffs
        for i in range(len(diffs)):
            mine = diffs[i]
            rest = diffs[0:i]+diffs[i+1:]
            winPct = np.mean([int(x>0) for x in rest])
            results.append((mine,winPct))
    return np.array(results)

def genAvgDiffPoints(pairs):
    results = []
    for pair in pairs:
        data = pairs[pair]
        teams = pair.split('_')
        diffs = np.array([int(x[3])-int(x[5]) for x in data.values()
                          if x[2] == teams[0]] + [int(x[5])-int(x[3])
                          for x in data.values() if x[4] == teams[0]])                
        avgDiff = np.mean(diffs)
        winPct = np.mean(diffs>0)
        results.append((avgDiff,winPct))
    return np.array(results)
    
def sigmoid(x,k):
    y = 1/(1+np.exp(-k*(x)))
    return y

def main():
    dbHeader = "C:/Users/Ted/Dropbox"
    rsfName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/regular_season_results.csv"
    teamfName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/teams.csv"
    trainName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/features_D.csv"
    featName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/all_D.csv"
    
    season = 'D'
    games = [game for game in loadResults(rsfName) if game[0]==season]
    
    tIDs = sorted(getTIDs(games))
        
    pairs = getMultiples(games,tIDs)
    #diffPts = genDiffPoints(pairs)
    diffPts = genAvgDiffPoints(pairs)
    
    x1data = diffPts[:,0]
    y1data = diffPts[:,1]
    x2data = xdata*-1
    y2data = 1-ydata
    
    xdata = np.append(x1data,x2data)
    ydata = np.append(y1data,y2data)
    
    popt, pcov = curve_fit(sigmoid,xdata,ydata)
    slope = popt[0]/4
    yprime = sigmoid(xdata,*popt)
    
    distinctDiffs = np.unique(xdata)
    binnedAvg = [np.mean(y[(xdata==ind).nonzero()]) for ind in distinctDiffs]
    
    plt.figure(1)
    plt.plot(diffPts[:,0],diffPts[:,1],'k.')
    plt.plot(xdata,yprime,'b.')
    plt.plot(distinctDiffs,binnedAvg,'g.')
    plt.show()