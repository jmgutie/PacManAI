from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.
        
        # Compute the values here.
        for i in range(iters):
            newValues = self.values.copy()
            for state in self.mdp.getStates():
                # if terminal state, set to 0
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue
                else:
                    bestValue = -999999
                    for action in self.mdp.getPossibleActions(state):
                        QValue = 0
                        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, nextState)
                            # get value from dict, if key doesnt exist insert value of 0
                            valueOfNextState = self.values.get(nextState, 0)
                            # perform calculation
                            QValue += prob * (reward + (self.discountRate * valueOfNextState))
                        # get the max value
                        if QValue > bestValue:
                            bestValue = QValue
                    newValues[state] = bestValue
                    
            self.values = newValues
        # raise NotImplementedError()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getQValue(self, state, action):
        QValue = 0
        TStates = self.mdp.getTransitionStatesAndProbs(state, action)
        for newState, prob in TStates:
            # get the reward of that state
            reward = self.mdp.getReward(state, action, newState)
            # refer to dictionary value
            valueOfNextState = self.values.get(newState, 0)
            # perform calculation, Q = sum of T * (reward of current state * E * V*)
            QValue += prob * (reward + (self.discountRate * valueOfNextState))
            # print(QValue)
        return QValue
                
    def getPolicy(self, state):
        bestAction = 'North'
        bestValue = -999999
        # bestActionValue = -999999

        if self.mdp.isTerminal(state):
            bestAction = None
            return bestAction
        
        for action in self.mdp.getPossibleActions(state):
            QValue = 0
            QValue += self.getQValue(state, action)
            # get the max value, return best action
            if QValue > bestValue:
                bestValue = QValue
                bestAction = action
        return bestAction