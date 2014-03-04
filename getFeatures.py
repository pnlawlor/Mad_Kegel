'''
Created on Jan 24, 2014

@author: Ted
'''

import networkx as nx
import csv
from sets import Set
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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

# Feature Functions

def winPct(data,team,startFrac,endFrac):
    tGames = selTeamGames(data,team)
    gameRange = [int(len(tGames)*startFrac),int(len(tGames)*endFrac)]
    nGames = float(gameRange[1]-gameRange[0])
    winN = len([x for x in tGames[gameRange[0]:gameRange[1]+1] if x[2] == team])
    return winN/nGames  

def winEarly(data,team):
    return winPct(data,team,0,0.33)

def winMid(data,team):
    return winPct(data,team,0.34,0.66)

def winLate(data,team):
    return winPct(data,team,0.67,1)

def pointsFor(data,team):
    tGames = selTeamGames(data,team)
    return ([int(x[3]) for x in tGames if x[2] == team] + 
            [int(x[5]) for x in tGames if x[4] == team])

def pointsAgainst(data,team):
    tGames = selTeamGames(data,team)
    return ([int(x[3]) for x in tGames if x[4] == team] + 
            [int(x[5]) for x in tGames if x[2] == team])

def ppg(data,team):
    return np.mean(pointsFor(data,team))

def ppga(data,team):
    return np.mean(pointsAgainst(data,team))

def diffpg(data,team):
    return ppg(data,team) - ppga(data,team)

def pfstd(data,team):
    return np.std(pointsFor(data,team))

def pastd(data,team):
    return np.std(pointsAgainst(data,team))

def getFeatureVector(data,t):
    return [winEarly(data,t), winMid(data,t), winLate(data,t), ppg(data,t), ppga(data,t), diffpg(data,t),
                    pfstd(data,t)*-1*np.sign(diffpg(data,t)),pastd(data,t)*-1*np.sign(diffpg(data,t))]
    
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
    
def main():
    dbHeader = "C:/Users/Ted/Dropbox"
    rsfName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/regular_season_results.csv"
    teamfName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/teams.csv"
    trainName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/features_D.csv"
    featName = dbHeader + "/Kording Lab/Projects/MarchMadness/Data/all_D.csv"
    
    season = 'D'
    games = [game for game in loadResults(rsfName) if game[0]==season]
    
    tIDs = sorted(getTIDs(games))
    names = loadTeamNames(teamfName)
    nTeams = len(tIDs)
    tData = {}
    for id in tIDs:
        tData[id] = getFeatureVector(games,id)
        
    of = open(trainName,'wb')
    featfile = csv.writer(of)
    for i in range(nTeams):
        for j in range(i+1,nTeams):
            id_string = "{0}_{1}_{2}".format(season,tIDs[i],tIDs[j])
            if j == i+1:
                print id_string
            if getWL(games,tIDs[i],tIDs[j])!="NA":
                featfile.writerow([id_string]+tData[tIDs[i]]+tData[tIDs[j]]+[getWLDiff(games,tIDs[i],tIDs[j])])
    of.close()
    
    
    
    