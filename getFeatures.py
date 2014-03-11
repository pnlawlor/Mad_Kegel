'''
Created on Jan 24, 2014

@author: Ted
'''

import numpy as np
import pandas as pd

#==============================================================================
# General Data Manipulation Functions
#==============================================================================

def selTeamGames(data,team):
    is_winner = data['wteam'] == team
    is_loser = data['lteam'] == team
    return data[is_winner | is_loser]

def getOpponents(data,team):
    tGames = selTeamGames(data,team)
    # find teams that were in TEAM's games but not TEAM
    notWin = tGames[tGames['wteam'] != team]['wteam']
    notLose = tGames[tGames['lteam'] != team]['lteam']
    return np.concatenate((notWin.values,notLose.values))

#==============================================================================
# Win Percent Calculation Code
#==============================================================================

def winPct(data,team,startFrac,endFrac):
    # Calculates win percent during a particular part of the season
    # First selects games for a particular team, then calculates #games in range
    tGames = selTeamGames(data,team)
    gameRange = [int(len(tGames)*startFrac),int(len(tGames)*endFrac)]
    nGames = float(gameRange[1]-gameRange[0])
    # Calculates number of games won
    gamesWon = tGames.iloc[range(*gameRange)]['wteam'].value_counts()
    if team in gamesWon.keys():
        winN = gamesWon[team]
    else:
        winN = 0
    return winN/nGames

def winEarly(data,team):
    return winPct(data,team,0,0.33)

def winMid(data,team):
    return winPct(data,team,0.34,0.66)

def winLate(data,team):
    return winPct(data,team,0.67,1)
    
#==============================================================================
# Other Assorted Single-Team Features
#==============================================================================

def pointsFor(data,team):
    tGames = selTeamGames(data,team)
    pf = tGames[tGames['wteam'] == team]['wscore'].append(
                tGames[tGames['lteam'] == team]['lscore'])
    return pf

def pointsAgainst(data,team):
    tGames = selTeamGames(data,team)
    pa = tGames[tGames['wteam'] == team]['lscore'].append(
                tGames[tGames['lteam'] == team]['wscore'])
    return pa

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
    
#==============================================================================
# Generate Single-Team Feature Vector
#==============================================================================

def getFeatureVector(data,t):
    return [winEarly(data,t), winMid(data,t), winLate(data,t), ppg(data,t), ppga(data,t), diffpg(data,t),
                    pfstd(data,t)*-1*np.sign(diffpg(data,t)),pastd(data,t)*-1*np.sign(diffpg(data,t))]
                    
#==============================================================================
# Generate Interaction Terms
#==============================================================================
    
def getWL(data,t1,t2):
    games = selTeamGames(selTeamGames(data,t1),t2)
    nGames = len(games)
    if nGames == 0:
        return "NA"
    else:
        gamesWon = games['wteam'].value_counts()
        if t1 in gamesWon.keys():
            nT1Win = gamesWon[t1]
        else:
            nT1Win = 0
        return nT1Win/float(nGames)
    
def getWLDiff(data,t1,t2):
    games = selTeamGames(selTeamGames(data,t1),t2)
    nGames = len(games)
    if nGames == 0:
        return "NA"
    else:
        ptDiffT1 = diffpg(games,t1)
        return ptDiffT1/float(nGames)

#==============================================================================
# Data Generator Functions
#==============================================================================
        
def regSeasFeatures(games,season,featfile):
    # Get list of all teams in season, sort them.
    tIDs = sorted(pd.unique(games[['wteam','lteam']].values.ravel()))
    tData = {}
    for id in tIDs:
        tData[id] = getFeatureVector(games,id)
    for t1 in tIDs:
        opps = getOpponents(games,t1)
        for t2 in np.unique(opps[opps>t1]):
            id_string = "{0}_{1}_{2}".format(season,str(t1),str(t2))
            featfile.writerow([id_string] + tData[t1] + tData[t2])


    
#==============================================================================
# Main code to execute
#==============================================================================

def main():
    dbHeader = "C:/Users/Ted/Dropbox"
    projHeader = "/Kording Lab/Projects/MarchMadness
    rsfName = dbHeader + projHeader + "/Data/regular_season_results.csv"
    teamfName = dbHeader + projHeader + "/Data/teams.csv"
    trainName = dbHeader + projHeader + "/Data/features_all.csv"
    tourneyName = dbHeader + projHeader + "/Data/tourney_results.csv"
    tourneyTrainName = dbHeader + projHeader + "/Data/tourneyFeatures_all.csv"
    
    seasons = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R']
    allData = pd.read_csv(rsfName)
#    tournData = pd.read_csv(tourneyName)
    names = pd.read_csv(teamfName)['name'].to_dict()
#    allExpDiff = "REPLACE WITH PATS DATA"
#    allSeeds = "READ IN SEEDS"
    
    of = open(trainName,'wb')
    featfile = csv.writer(of)
#    otf = open(tournTrainName,'wb')
#    tournfeatfile = csv.writer(otf)
    
    for season in seasons:
        print "Season: "+season
        # Find all games that were played this season
        games = allData[allData['season'] == season]
#        tournGames = tournData[tournData['season'] == season]
#        expDiff = allExpDiff[allExpDiff['season'] == season]
        regSeasFeatures(games,season,of)
#        tourneyFeatures(tournGames,games,expDiff,seeds)
    of.close()
    otf.close()