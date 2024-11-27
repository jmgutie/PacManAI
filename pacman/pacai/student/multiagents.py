import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

# imports
# from pacai.core.search import heuristic
# from pacai.student import searchAgents
# from pacai.student import search
from pacai.core import distance
# from pacai.core import agentstate
# from pacai.agents.ghost import directional
# from pacai.core import directions

# from pacai.core.gamestate import AbstractGameState

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]
        
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        Distances = []
        evaluationScore = 0

        # Useful information you can extract.
        currPosition = currentGameState.getPacmanPosition()  # current Pacmans coordinates
        newPosition = successorGameState.getPacmanPosition()  # future pacmans coordinates
        # oldFood = currentGameState.getFood()  # a list of False/Trues
        newGhostStates = successorGameState.getGhostStates()  # an array of ghosts, type list
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates] time2eat

        # *** Your Code Here ***

        # consider both food locations and ghost locations
        # if food close, good thing, high number
        
        # if ghost close, bad thing, low number

        # if pellet close and ghost close, good thing, high number

        # if pellet close and no ghost, mid thing, medium number
        self.Food = currentGameState.getFood()
        GroceryList = self.Food.asList()
        
        # ############## Handling Food ###################################### #
        for item in GroceryList:  # for every food nearby
            manDistance = distance.manhattan(currPosition, item)  # calc manhattan distances
            Distances.append(manDistance)   # add to list
        ClosetFood = min(Distances)  # find closest food
        FoodIndex = Distances.index(ClosetFood)  # index of the closest food appended to Distances

        # calc manhattan distance, faster but gets stuck half the time
        # future = distance.manhattan(newPosition, GroceryList[FoodIndex])
        # past = distance.manhattan(currPosition, GroceryList[FoodIndex])

        # calc maze distance, reliable but slow
        futureFood = distance.maze(newPosition, GroceryList[FoodIndex], currentGameState)
        pastFood = distance.maze(currPosition, GroceryList[FoodIndex], currentGameState)

        # check if we are moving towards food by comparing future and current pac man
        if futureFood < pastFood:   # if distance of future > distance of past --
            evaluationScore += 200  # 200
            successorGameState.addScore(evaluationScore)
        if pastFood > futureFood:   # if distance of future < distance of past ++
            evaluationScore -= 100  # 100
            successorGameState.addScore(evaluationScore)
            
        # ################################################################### #

        # print("Pac man is ", currPosition)
        # print("The food index is: ", FoodIndex, "\nThe grocery list is: ",
        # GroceryList, "\nThe closest food is: ", ClosetFood, "Manhattan distance aways")
        # print("The coordinate of the closest food is :", GroceryList[FoodIndex])

        # calc distance for each ghost, get the closest one and check if moving towards or away
        for ghost in newGhostStates:
            futureGhost = distance.manhattan(newPosition, ghost.getPosition())
            pastGhost = distance.manhattan(currPosition, ghost.getPosition())

            if ghost.isBraveGhost():    # if ghost is brave, run away
                if futureGhost <= pastGhost:    # if distance of future < distance of past or ==, -
                    if pastGhost < 1:
                        evaluationScore -= 500  # 500
                    if pastGhost < 5:
                        evaluationScore -= 300  # 300
                    if pastGhost > 5:
                        evaluationScore -= 0  # 0
                    successorGameState.addScore(evaluationScore)
                if futureGhost > pastGhost and pastGhost < 3:  # if future > distance of past, +
                    evaluationScore += 100  # 100
                    successorGameState.addScore(evaluationScore)

            else:                       # if ghost is scared, run towards
                if futureGhost < pastGhost and pastGhost < 3:   # if future < distance of past, +
                    evaluationScore += 300  # 300
                    successorGameState.addScore(evaluationScore)
                if pastGhost > futureGhost:                    # if past > distance of future, -
                    evaluationScore += 0  # 0
                    successorGameState.addScore(evaluationScore)
        
        return successorGameState.getScore()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        depth = 0
        bestValue = -999999
        bestAction = 'Stop'
        
        newActions = state.getLegalActions(0)
        for action in newActions:
            if action == "Stop":
                continue
            newState = state.generateSuccessor(0, action)
            value = self.Max(newState, depth, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
                    
    # literally change state, by feeding in newState call min if ghost
    def Max(self, state, depth, turnCounter):
        maxScore = -999999
        # numOfAgents = state.getNumAgents()
        # agentIndex = turnCounter % numOfAgents
        totalDepth = self.getTreeDepth()
        
        # if a cycle was completed
        if depth >= totalDepth:
            return self.getEvaluationFunction()(state)
        
        # if terminal
        if state.isWin():
            return self.getEvaluationFunction()(state)
        
        # if terminal
        if state.isLose():
            return self.getEvaluationFunction()(state)
        
        # get every action
        newState = state.getLegalActions(0)
        # for each direction pacman can go in
        for direction in newState:
            if direction == 'Stop':
                continue
            # create successor, from each min value, for each direction
            new = state.generateSuccessor(0, direction)
            tempScore = self.Min(new, depth, turnCounter + 1)
           
            # if tempScore is better than bestScore, replace Score and Action to perform
            if tempScore > maxScore:
                maxScore = tempScore
        return maxScore
            
    def Min(self, state, depth, turnCounter):
        minScore = 999999
        numOfAgents = state.getNumAgents()
        
        agentIndex = turnCounter % numOfAgents
        totalDepth = self.getTreeDepth()

        # check if end state
        if state.isWin():
            return self.getEvaluationFunction()(state)
        
        if state.isLose():
            return self.getEvaluationFunction()(state)
        
        if depth >= totalDepth:
            return self.getEvaluationFunction()(state)
        
        newMoves = state.getLegalActions(agentIndex)
        for direction in newMoves:
            if direction == 'Stop':
                continue
            newState = state.generateSuccessor(agentIndex, direction)
            # if agent is a ghost
            if agentIndex + 1 < numOfAgents:
                nextTurnCounter = turnCounter + 1
                tempScore = self.Min(newState, depth, nextTurnCounter)
                if tempScore < minScore:  # if tempScore better than bestScore, replace Score&Action
                    minScore = tempScore
            # if agent is pacman
            else:
                nextTurnCounter = turnCounter + 1
                tempScore = self.Max(newState, depth + 1, nextTurnCounter)
                if tempScore < minScore:  # if tempScore better than bestScore, replace Score&Action
                    minScore = tempScore
        return minScore
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        # turnCounter = 0
        depth = 0
        # numOfAgents = state.getNumAgents()
        bestValue = -999999
        bestAction = 'Stop'
        alpha = -999999
        beta = 999999
        
        newActions = state.getLegalActions(0)
        for action in newActions:
            if action == "Stop":
                continue
            newState = state.generateSuccessor(0, action)
            value = self.Max(newState, depth, 0, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
                alpha = max(bestValue, alpha)
        return bestAction
                    
    # literally change state, by feeding in newState call min if ghost
    def Max(self, state, depth, turnCounter, alpha, beta):
        maxScore = -999999
        # move = 'Stop'
        # numOfAgents = state.getNumAgents()
        # agentIndex = turnCounter % numOfAgents
        totalDepth = self.getTreeDepth()
        
        # if a cycle was completed
        if depth >= totalDepth:
            return self.getEvaluationFunction()(state)
        
        # if terminal
        if state.isWin():
            return self.getEvaluationFunction()(state)
        
        # if terminal
        if state.isLose():
            return self.getEvaluationFunction()(state)
        
        # get every action
        newState = state.getLegalActions(0)
        # for each direction pacman can go in
        for direction in newState:
            if direction == 'Stop':
                continue
            # create successor, from each min value, for each direction
            new = state.generateSuccessor(0, direction)
            tempScore = self.Min(new, depth, turnCounter + 1, alpha, beta)
           
            # if tempScore is better than bestScore, replace Score and Action to perform
            if tempScore > maxScore:
                maxScore = tempScore
            if maxScore >= beta:
                break
            alpha = max(alpha, maxScore)
        return maxScore
            
    def Min(self, state, depth, turnCounter, alpha, beta):
        minScore = 999999
        numOfAgents = state.getNumAgents()
        # move = 'Stop'
        
        agentIndex = turnCounter % numOfAgents
        totalDepth = self.getTreeDepth()

        # check if end state
        if state.isWin():
            return self.getEvaluationFunction()(state)
        
        if state.isLose():
            return self.getEvaluationFunction()(state)
        
        if depth >= totalDepth:
            return self.getEvaluationFunction()(state)
        
        newMoves = state.getLegalActions(agentIndex)
        for direction in newMoves:
            if direction == 'Stop':
                continue
            newState = state.generateSuccessor(agentIndex, direction)
            # if agent is a ghost
            if agentIndex + 1 < numOfAgents:
                nextTurnCounter = turnCounter + 1
                tempScore = self.Min(newState, depth, nextTurnCounter, alpha, beta)
                if tempScore < minScore:  # if tempScore better than bestScore, replace Score&Action
                    minScore = tempScore
                if tempScore < alpha:
                    break
                beta = min(beta, tempScore)
            # if agent is pacman
            else:
                nextTurnCounter = turnCounter + 1
                tempScore = self.Max(newState, depth + 1, nextTurnCounter, alpha, beta)
                if tempScore < minScore:  # if tempScore better than bestScore, replace Score&Action
                    minScore = tempScore
                if tempScore >= beta:
                    break
                alpha = max(alpha, tempScore)
        return minScore

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        # init counters
        # get each action of pacman
        # generate successor for each action
        # if successor is terminal, return evaluationFunction
        # if successor is max(pacman), return Max
        # if successor is min(ghost), return Exp
        depth = 0
        bestValue = -999999
        bestAction = 'Stop'
        
        newActions = state.getLegalActions(0)
        for action in newActions:
            if action == "Stop":
                continue
            newState = state.generateSuccessor(0, action)
            value = self.Max(newState, depth, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
            
    # get each action of pacman
    # generate successor for each action
    # if successor is terminal, return evaluationFunction
   
    def Max(self, state, depth, turnCounter):
        maxScore = -999999
        totalDepth = self.getTreeDepth()
        
        # if a cycle was completed
        if depth >= totalDepth:
            return self.getEvaluationFunction()(state)
        
        # if terminal
        if state.isWin():
            return self.getEvaluationFunction()(state)
        
        # if terminal
        if state.isLose():
            return self.getEvaluationFunction()(state)
        
        # get every action
        newState = state.getLegalActions(0)
        # for each direction pacman can go in
        for direction in newState:
            if direction == 'Stop':
                continue
            # create successor, from each min value, for each direction
            new = state.generateSuccessor(0, direction)
            tempScore = self.Exp(new, depth, turnCounter + 1)
           
            # if tempScore is better than bestScore, replace Score and Action to perform
            if tempScore > maxScore:
                maxScore = tempScore  
        return maxScore
    
    def Exp(self, state, depth, turnCounter):
        tempValues = 0
        actionCounter = 0
        numOfAgents = state.getNumAgents()
        agentIndex = turnCounter % numOfAgents

        totalDepth = self.getTreeDepth()
        if state.isWin() or state.isLose() or depth >= totalDepth:
            return self.getEvaluationFunction()(state)
        
        actions = state.getLegalActions(agentIndex)
        for action in actions:
            actionCounter += 1
            if action == 'Stop':
                continue
            
            succ = state.generateSuccessor(agentIndex, action)
            if agentIndex + 1 < numOfAgents:
                nextTurnCounter = turnCounter + 1
                values = self.Exp(succ, depth, nextTurnCounter)
                tempValues += values
                
            else:
                nextTurnCounter = turnCounter + 1
                values = self.Max(succ, depth + 1, nextTurnCounter)
                tempValues += values
        # calc average
        weight = tempValues / actionCounter
        return weight


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    
    foodDistances = []
    food = currentGameState.getFood()
    groceryList = food.asList()
    currPosition = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()

    # determines how close is food
    for item in groceryList:  # for every food nearby
        foodManDistance = distance.manhattan(currPosition, item)  # calc manhattan distances
        foodDistances.append(foodManDistance)   # add to list
    # if list is empty
    if len(foodDistances) == 0:
        foodManDistance = 0
    else:
        closetFood = min(foodDistances)  # find closest food
        foodIndex = foodDistances.index(closetFood)  # index of closest food appended to Distance
        foodManDistance = distance.manhattan(currPosition, groceryList[foodIndex])

    # determines how close is ghost
    for ghost in newGhostStates:
        ghostManDistance = distance.manhattan(currPosition, ghost.getPosition())
        if ghost.isBraveGhost():    # if ghost is brave
            if ghostManDistance <= 3:  # 3
                ghostFactor = -100  # 100
            else:
                ghostFactor = 0
        else:   #ghost is scared, eat it
            if ghostManDistance < 3:  # 3
                ghostFactor = 150  # 150
            else:  
                ghostFactor = 0

    # using the distance as factors (food close = good)(ghost close = bad)
    return currentGameState.getScore() - foodManDistance + ghostFactor


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
