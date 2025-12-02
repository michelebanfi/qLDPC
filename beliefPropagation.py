import numpy as np

from drawUtils import plotGraph
    
def performBeliefPropagation(H, error, initialBelief, verbose = True, plotPath=None):

    H = np.array(H, dtype=float)

    # build mapping
    varNode_to_checkNode = {}
    checkNode_to_varNode = {}

    for checkNode in range(len(H)):
        temp = []
        for varNode in range(len(H[0])):
            if H[checkNode][varNode] == 1: temp.append(varNode)
        
        checkNode_to_varNode[checkNode] = temp

    H_T = H.T

    syndrome = (error @ H_T) % 2
    if verbose: print(f"Initial syndrome: {syndrome}")

    for varNode in range(len(H_T)):
        temp = []
        for checkNode in range(len(H_T[0])):
            if H_T[varNode][checkNode] == 1: temp.append(checkNode)
            
        varNode_to_checkNode[varNode] = temp

    if not plotPath == None: plotGraph(H, path=plotPath)

    Q = np.copy(H)
    R = np.copy(H)

    # Q ha the original beliefs from the "channel"
    for checkNode in range(len(H)):
        for varNode in range(len(H[0])):
            if H[checkNode][varNode] == 1: 
                Q[checkNode][varNode] = initialBelief[varNode]
                R[checkNode][varNode] = initialBelief[varNode]

    isSindromefound = False
    maxIter = 30
    currentIter = 0
    while not isSindromefound and currentIter < maxIter:
        
        # horizontal step, aggregate the entry message 
        for checkNode in range(len(H)): 
            
            syndromeSign = 1 if syndrome[checkNode] == 0 else -1
            
            variableNodesConnected = checkNode_to_varNode[checkNode]
            
            for variableNode_i in variableNodesConnected:
                temp = []
                for variableNode_j in variableNodesConnected:
                    if variableNode_i != variableNode_j:
                        temp.append(Q[checkNode][variableNode_j])
                
                R[checkNode][variableNode_i] = 2 * np.arctanh(np.clip(np.prod(np.tanh(np.array(temp) / 2)),-0.999, 0.999 ) * syndromeSign)
        
        # vertical step
        values = []
        for variableNode in range(len(H[0])):
            temp = []
            for vv in varNode_to_checkNode[variableNode]:            
                temp.append(R[vv][variableNode])
            values.append(np.sum(temp) + initialBelief[variableNode])
        
        
        # values update    
        for variableNode in range(len(H[0])):
            for checkNode in varNode_to_checkNode[variableNode]:
                Q[checkNode][variableNode] = values[variableNode] - R[checkNode][variableNode]
                
        candidateError = np.array([1 if i < 0 else 0 for i in values])
        calculateSyndrome = (candidateError @ H_T) % 2
        
        if np.array_equal(calculateSyndrome, syndrome): 
            isSindromefound = True
            if verbose: print(f"Error found at iteration {currentIter}: {candidateError}")
        
        
        currentIter += 1
        
    return candidateError, isSindromefound, values