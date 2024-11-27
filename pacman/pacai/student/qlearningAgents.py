from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
                
        # You can initialize Q-values here.
        self.QValues = {}
        
    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        
        # if exists return value, if not return 0
        value = self.QValues.get((state, action), 0)
        return value

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        bestValue = float('-inf')
        actions = self.getLegalActions(state)

        # if terminal state
        if len(actions) == 0:
            return 0.0
        
        # for each action
        for act in actions:
            # get QValue
            QValue = self.getQValue(state, act)
            # find max
            if QValue > bestValue:
                bestValue = QValue
        return bestValue

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        bestNum = float('-inf')
        bestAction = None
        actions = self.getLegalActions(state)
        
        if len(actions) == 0.0:
            return None
        
        QValue = 0.0
        for act in actions:
            QValue = self.getQValue(state, act)
            if QValue > bestNum:
                bestNum = QValue
                bestAction = act
        return bestAction
       
    def update(self, state, action, nextState, reward):
        
        oldQValue = self.getQValue(state, action)
        newQVal = self.getValue(nextState)
        
        Sample = reward + (self.discountRate * newQVal)
        newVal = (1 - self.alpha) * (oldQValue) + self.alpha * (Sample)
        self.QValues[(state, action)] = newVal

    def getAction(self, state):
        epsilon = self.getEpsilon()
        decision = probability.flipCoin(1 - epsilon)
        # if returns head
        if decision:
            return self.getPolicy(state)
        else:
            actions = self.getLegalActions(state)
            if actions:
                return random.choice(actions)
            else:
                return None
                
            
class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weight = {}
        
    def getQValue(self, state, action):
        # update QValues
        features = self.featExtractor().getFeatures(state, action)
        Qval = 0
        for key, value in features.items():
            Qval += value * self.weight.get(key, 0)
        return Qval
        
    def update(self, state, action, nextState, reward):
        # update the weight vector init in init func
        features = self.featExtractor().getFeatures(state, action)
        q = self.getQValue(state, action)
        error = (reward + self.discountRate * self.getValue(nextState)) - q
        for key, value in features.items():
            w = self.weight.get(key, 0)
            # for each feature, calc weight
            self.weight[key] = w + self.alpha * (error)

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass
            # raise NotImplementedError()
