'''
Created on Jan 21, 2014

@author: Ted
'''

import networkx as nx
import csv
from sets import Set
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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

def getTeams(data):
    return list(Set([x[2] for x in data] + [x[4] for x in data]))

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

def rawEdgesFromGames(data):
    edges = []
    diffs = [float(x[3])-float(x[5]) for x in data]
    maxDiff = max(diffs)
    for i in range(len(data)):
        winnerP = diffToP(diffs[i],maxDiff)
        edges = (edges + [(data[i][2],data[i][4],(1-winnerP))]
                 + [(data[i][4],data[i][2],winnerP)])
    return edges

def avgDupeEdges(G):
    return G

def buildGraph(data):
    G = nx.MultiDiGraph()
    G.add_nodes_from(getTeams(data))
    G.add_weighted_edges_from(rawEdgesFromGames(data))
    normG = normEdges(avgDupeEdges(G))
    return normG

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

def predictGames(data,cts):
    correct = 0.0
    for game in data:
        tw = game[2]
        tl = game[4]
        if cts[tw] > cts[tl]:
            correct += 1
    return correct/len(data)

def main():
    resName = "C:/Users/Ted/Dropbox/Kording Lab/Projects/MarchMadness/Data/regular_season_results.csv"
    teamfName = "C:/Users/Ted/Dropbox/Kording Lab/Projects/MarchMadness/Data/teams.csv"
    tournfName = "C:/Users/Ted/Dropbox/Kording Lab/Projects/MarchMadness/Data/tourney_results.csv"
    
    season = 'R'
    games = [game for game in loadResults(resName) if game[0]==season]
    G = buildGraph(games)
    
    # Plot the graph
    nx.draw(G)
    plt.show()

    # Walk the dinosaur
    nsteps = 1000000
    cts = randomWalk(G,nsteps)
    
    # Print rankings... or some other evaluation?
    rankings = sorted(cts,key=cts.get,reverse=True)
    names = loadTeamNames(teamfName)
    numTeams = 25
    for i in range(numTeams):
        team = rankings[i]
        print "{0}: {1}".format(i+1,names[team])
        
    # Evaluate on tourney data
    tourney = [game for game in loadResults(tournfName) if game[0]==season]
    print predictGames(tourney,cts)


if __name__ == "__main__":
    main()