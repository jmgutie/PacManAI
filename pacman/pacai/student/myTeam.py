# from pacai.util import reflection
from pacai.agents.capture.reflex import CaptureAgent
# from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.core.distance import manhattan
import random
# from pacai.util import util

# TODO: TEST TO SEE IF THE MASTER BRANCH MERGES EVERYTIME WE PUSH

# TODO: do not import qulafied import THIS IS ONE OF THE CHECKS KAIA
# IDFK WHAT TO DO I WANNA CRYYYYYY
# TODO: make both agents offensive agents instead of defenisve offensive??

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # batu said get rid of qualified import
    # firstAgent = reflection.qualifiedImport(first)
    # secondAgent = reflection.qualifiedImport(second)

    return [
        defensiveAgent(firstIndex),
        offensiveAgent(secondIndex),
    ]

# reflex capture agent ???
class defensiveAgent(CaptureAgent):
    def getFeatures(self, gameState, action):
        features = {}
        # pacman
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)

        newGhostStates = successor.getGhostStates()
        # feature 1: score
        features['successorScore'] = self.getScore(successor)
        
        # feature 2: distance to nearest food
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            myPos = myState.getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # if winning send one ghost to defend
        if self.getScore() > 10:
            # ghost
            min = 999999
            enemyIndex = successor.getOpponents(gameState)
            for i in enemyIndex:
                enemyState = successor.getAgentState(enemyIndex)
                enemyPos = enemyState.getPosition(i)
                myPos = myState.getPosition()

                ghostDistance = self.getMazeDistance(myPos, enemyPos)
                if min > ghostDistance:
                    min = ghostDistance
            
            features['distanceToGhost'] = min
        # TODO: add more features here based on strategy
        return features

    def getWeights(self, gameState, action):
        # define weights for features 
        # adjust these values to reflect strategy
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            # TODO: add more feature weights as needed
        }

# reflex capture agent ???
class offensiveAgent(CaptureAgent):
    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)

        # feature 1: score (negative to encourage defensive play)
        features['successorScore'] = -self.getScore(successor)

        # feature 2: distance to nearest invader
        opponents = self.getOpponents(successor)
        invaders = [successor.getAgentState(i) for i in opponents if successor.getAgentState(i).isPacman]
        if len(invaders) > 0:
            myPos = myState.getPosition()
            minDistance = min([self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders])
            features['distanceToInvader'] = minDistance
        else:
            features['distanceToInvader'] = 0  # no invaders, no penalty

        # TODO: add more features here based on defensive strategy
        
        return features

    def getWeights(self, gameState, action):
        # define weights for features
        # adjust these values to reflect defensive strategy
        return {
            'successorScore': -100,
            'distanceToInvader': -1,
            # TODO: add more feature weights as needed
        }
    