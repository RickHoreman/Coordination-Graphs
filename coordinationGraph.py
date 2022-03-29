import random
import copy
from queue import Queue
from matplotlib import pyplot as plt
from time import time

class edge:

    def __init__(self, var1, var2, nactionsx, nactionsy):
        """
        Constructor for the edge class
        :param var1: the (index of the) first decision variable
        :param var2: the (index of the) second decision variable
        :param nactionsx: the number of possible values for var1
        :param nactionsy: the number of possible values for var1
        """
        self.rewards = [] #table with the local rewards
        self.x = var1
        self.y = var2
        for i in range(nactionsx):
            rew = []
            for j in range(nactionsy):
                rew.append( random.random() )
            self.rewards.append(rew)

    def localReward(self, xval, yval):
        """
        Return the local reward for this edge given the values of the connected decision variables
        :param xval: value of the first decision variable
        :param yval: value of the second decision variable
        :return: the local reward
        """
        return self.rewards[xval][yval]


class coordinationGraph:

    def __init__(self, noNodes, pEdge, noActions, seed=42):
        """
        Constructor for the coordination graph class. It generates a random graph based on a seed.

        :param noNodes: The number of vertices in the graph. Each vertex represents a decision variable.
        :param pEdge: the probability that an edge will be made
        :param noActions: the number of possible values (integers between 0 and noActions) for the decision variables
        :param seed: the pre-set seed for the random number generator
        """
        random.seed(seed)
        self.nodesAndConnections = dict() #for each node, a list of nodes it is connected to
        self.edges = dict() #A dictionary of tuples (of two decision variables) to an object of the edge class
        for i in range(noNodes): #First make sure that the entire graph is connected (connecting all nodes to the next one)
            if i == 0:
                self.nodesAndConnections[i] = [i + 1]
                self.nodesAndConnections[i+1] = [i]
                eddy = edge(i, i+1, noActions, noActions)
                self.edges[(i,i+1)] = eddy
            elif i <noNodes-1:
                self.nodesAndConnections[i].append(i + 1)
                self.nodesAndConnections[i + 1] = [i]
                eddy = edge(i, i + 1, noActions, noActions)
                self.edges[(i, i + 1)] = eddy
        tuplist = [(x, y) for x in range(noNodes) for y in range(x + 2, noNodes)]
        for t in tuplist: #Then, for each possible edge, randomly select which exist and which do not
            r = random.random()
            if r < pEdge:
                self.nodesAndConnections[t[0]].append(t[1])
                self.nodesAndConnections[t[1]].append(t[0])
                self.edges[t] = edge(t[0], t[1], noActions, noActions)
        #For reasons of structure, finally, let's sort the connection lists for each node
        for connections in self.nodesAndConnections.values():
            connections.sort()

        self.nNodes = noNodes
        self.nActions = noActions
    
    def randomSolution(self, seed=None):
        random.seed(seed)
        solution = []
        for _ in range(self.nNodes):
            solution.append(random.randrange(self.nActions))
        return solution

    def evaluateSolution(self, solution):
        """
        Evaluate a solution from scratch; by looping over all edges.

        :param solution: a list of values for all decision variables in the coordination graph
        :return: the reward for the given solution.
        """
        result = 0
        for i in range(len(solution)):
            for j in self.nodesAndConnections[i]:
                if(j>i):
                    print( "("+str(i)+","+str(j)+") -> "+str(self.edges[(i,j)].localReward(solution[i], solution[j])))
                    result += self.edges[(i,j)].localReward(solution[i], solution[j])
        return result

    def evaluateChange(self, oldSolution, variableIndex, newValue):
        """
        DONE: a function that evaluates a local change. 
        NB: Make sure NOT to evaluate the entire solution twice (that would be a waste of computation time!!!) 

        :param oldSolution: The original solution
        :param variableIndex: the index of the decision variable that is changing
        :param newValue: the new value for the decision variable
        :return: The difference in reward between the old solution and the new solution (with solution[variableIndex] set to newValue)
        """
        
        connectedNodes = self.nodesAndConnections[variableIndex]
        oldLocalReward = 0
        newLocalReward = 0
        for node in connectedNodes:
            if variableIndex < node:
                _edge = self.edges[(variableIndex, node)]
                oldLocalReward += _edge.localReward(oldSolution[variableIndex], oldSolution[node])
                newLocalReward += _edge.localReward(newValue, oldSolution[node])
            else:
                _edge = self.edges[(node, variableIndex)]
                oldLocalReward += _edge.localReward(oldSolution[node], oldSolution[variableIndex])
                newLocalReward += _edge.localReward(oldSolution[node], newValue)

        return newLocalReward - oldLocalReward

def getVars(coordinationGraph):
    """
    :param coordinationGraph: the coordination graph to yoink the nodes from
    :return: a shuffled queue of decision vars
    """
    nodeIndices = []
    queue = Queue()
    for key in coordinationGraph.nodesAndConnections.keys():
        nodeIndices.append(key)
    random.shuffle(nodeIndices)
    for i in nodeIndices:
        queue.put(i)
    return queue

def localSearch4CoG(coordinationGraph, initialSolution):
    """
    DONE: Implement local search
    :param coordinationGraph: the coordination graph to optimise for
    :param initialSolution: an initial solution for the coordination graph
    :return: a new solution (a local optimum)
    """
    solution = copy.copy(initialSolution)
    vars = getVars(coordinationGraph)
    while not vars.empty():
        i = vars.get()
        connectedNode = coordinationGraph.nodesAndConnections[i][0]
        if connectedNode > i:
            nValues = len(coordinationGraph.edges[(i, connectedNode)].rewards)
        else:
            nValues = len(coordinationGraph.edges[(connectedNode, i)].rewards[0])
        for ai in range(nValues):
            Δ = coordinationGraph.evaluateChange(solution, i, ai)
            if Δ > 0:
                solution[i] = ai
                vars = getVars(coordinationGraph)
                break
    return solution

def multiStartLocalSearch4CoG(coordinationGraph, noIterations):
    """
    TODO: Implement multi-start local search

    :param coordinationGraph: the coordination graph to optimise for
    :param noIterations:  the number of times local search is run
    :return: the best local optimum found and its reward
    """
    solution = None
    reward = -float('inf')
    for _ in range(noIterations):
        newSolution = coordinationGraph.randomSolution()
        newSolution = localSearch4CoG(coordinationGraph, newSolution)
        newReward = coordinationGraph.evaluateSolution(newSolution)
        if newReward > reward:
            print("yes")
            solution = newSolution
            reward = newReward
        else:
            print("no")
    return solution, reward


def iteratedLocalSearch4CoG(coordinationGraph, pChange, noIterations):
    """
    TODO: Implement iterated local search

    :param coordinationGraph: the coordination graph to optimise for
    :param pChange: the perturbation strength, i.e., when mutating the solution, the probability for the value of a given
                    decision variable to be set to a random value.
    :param noIterations:  the number of iterations
    :return: the best local optimum found and its reward
    """
    solution = None
    reward = 0
    return solution, reward

###TODO OPTIONAL: implement genetic local search.

nVars = 50
nActs = 3
teamRewards = []
totalRuntime = 0
for i in range(100):
    cog = coordinationGraph(nVars,1.5/nVars,nActs, i)
    solution = [2]*nVars
    startTime = time()
    solution = localSearch4CoG(cog, solution)
    totalRuntime += time() - startTime
    teamRewards.append(cog.evaluateSolution(solution))
print(f"Average localsearch runtime: {totalRuntime/100} seconds. (highest reward: {max(teamRewards)})")
plt.hist(teamRewards)
plt.show()

cog = coordinationGraph(nVars,1.5/nVars,nActs, i)
solution, reward = multiStartLocalSearch4CoG(cog, 100)
print(f"Multistart local search reward: {reward}")

# print(cog.nodesAndConnections)
# print(cog.edges)
# print(cog.evaluateSolution(solution))
# solution = localSearch4CoG(cog, solution)
# print(cog.evaluateSolution(solution))
