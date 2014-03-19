'''
Created on Jan 21, 2014

@author: Ted
'''

import networkx as nx
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==============================================================================
# Maintenance Code
#==============================================================================

def getTeams(data):
    # Returns list of unique teams in data
    return sorted(pd.unique(data[['wteam','lteam']].values.ravel()))

#==============================================================================
# Graph Initialization Code
#==============================================================================

def buildGraph(data):
    # Builds a graph with teams connected by games they played
    G = nx.MultiDiGraph()
    G.add_nodes_from(getTeams(data))
    G.add_weighted_edges_from(rawEdgesFromGames(data))
    normG = normEdges(G)
    return normG
    
def rawEdgesFromGames(data):
    localData = data.copy()
    localData['diffs'] = data['wscore'] - data['lscore']
    maxDiff = max(localData['diffs'])
    edges = [] 
    for i in range(len(data)):
        game = localData.iloc[i]
        winnerP = diffToP(game['diffs'],maxDiff)
        edges = (edges + [(game['wteam'],game['lteam'],(1-winnerP))]
                 + [(game['lteam'],game['wteam'],winnerP)])
    return edges

def diffToP(diff,maxDiff):
    #Assume we get sent the winning diff
    return 0.5 + 0.5*(diff/maxDiff)

def normEdges(G):
    normG = G.copy()
    for node in normG.nodes():
        normWeight = sum([x[2]['weight'] for x in normG.out_edges(node,False,True)])
        for edge in normG.out_edges(node,True,True):
            normG[node][edge[1]][edge[2]]['weight'] /= normWeight
    return normG
    
#==============================================================================
# Random Walk Code
#==============================================================================

def selectOutNode(edges):
    # Expects output of G.out_edges(node,False,True), i.e. no keys
    thresh = random.uniform(0,1)
    cumProb = 0.0
    weightList = [(x[1],x[2]['weight']) for x in edges]
    for node,weight in weightList:
        cumProb += weight
        if thresh < cumProb: break
    return node

def randomWalk(G,n):
    startNode = random.choice(G.nodes())
    counts = dict.fromkeys(G.nodes(),0)
    currentNode = startNode
    i = 0
    while i<n:
        counts[currentNode] += 1
        edges = G.out_edges(currentNode,False,True)
        currentNode = selectOutNode(edges)
        i+=1
    return counts
    
#==============================================================================
# Main Code
#==============================================================================

def main():
    dbHeader = "C:/Users/Ted/Dropbox"
    projHeader = "/Kording Lab/Projects/MarchMadness"
    rsfName = dbHeader + projHeader + "/Data/regular_season_results.csv"
    teamfName = dbHeader + projHeader + "/Data/teams.csv"
    tourneyName = dbHeader + projHeader + "/Data/tourney_results.csv"
    walkName = dbHeader + projHeader + "/Data/walk.csv"
    
    seasons = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S']
    nsteps = 100000000
    
    allData = pd.read_csv(rsfName)
#    names = pd.read_csv(teamfName)['name'].to_dict()
    names = pd.read_csv(teamfName)
    
    of = open(walkName,'wb')
    owrite = csv.writer(of)
    owrite.writerow(['team','season','rank'])
    
    for season in seasons:
        print "Season: "+season
        games = allData[allData['season'] == season]
        G = buildGraph(games)
        tempCts = randomWalk(G,nsteps)
        sortCts = sorted(tempCts,key=tempCts.get,reverse=True)
        for i in range(len(sortCts)):
            owrite.writerow([sortCts[i],season,str(i)])
    
    of.close()
            

if __name__ == "__main__":
    main()