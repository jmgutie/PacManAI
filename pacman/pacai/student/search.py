"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
import pacai.util.stack as Stack
import pacai.util.queue as Queue
import pacai.util.priorityQueue as PQ
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here **
    fringe = Stack.Stack()
    start = problem.startingState()
    fringe.push([start, [], 0])
    if problem.isGoal(start):
        return fringe
    explored = set()
    action = []
    cost = 0
   
    while 1:
        # if empty
        if fringe.isEmpty():
            return None
        state, action, cost = fringe.pop()  # pop and store fringe into var
        
        # add to our explored
        if problem.isGoal(state):          # if we are at goal, return action
            return action
        if state not in explored:
            explored.add(state)
        for child in problem.successorStates(state):
            # if problem.isGoal(state):
            if child[0] not in explored:
                explored.add(child[0])
                # update new state, append to list of actions, and increment cost
                state = child[0]
                cost += child[2]
                fringe.push([state, action + [child[1]], cost])
                        
    # raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    fringe = Queue.Queue()
    start = problem.startingState()
    fringe.push([start, [], 0])
    if problem.isGoal(start):
        return fringe
    explored = set()
    action = []
    cost = 0
   
    while 1:
        if fringe.isEmpty():  # if empty
            return None
        state, action, cost = fringe.pop()  # pop and store fringe into var
        
        # add to our explored
        if problem.isGoal(state):
            return action
        if state not in explored:
            explored.add(state)
        for child in problem.successorStates(state):
            # if problem.isGoal(state):
            # return action
            if child[0] not in explored:
                explored.add(child[0])
                # update new state, append to list of actions, and increment cost
                state = child[0]
                cost += child[2]
                fringe.push([state, action + [child[1]], cost])
    # raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    explored = set()
    action = []
    cost = 0
    fringe = PQ.PriorityQueue()
    start = problem.startingState()
    fringe.push([start, [], 0], cost)
    if problem.isGoal(start):
        return fringe
    
    while 1:
        if fringe.isEmpty():  # if empty
            return None
        state, action, cost = fringe.pop()  # pop and store fringe into var
        
        if problem.isGoal(state):          # if we are at goal, return action
            return action
        if state not in explored:
            explored.add(state)
        for child in problem.successorStates(state):
            if child[0] not in explored:
                explored.add(child[0])
                # update new state, append to list of actions, and increment cost
                state = child[0]
                cost += child[2]
                fringe.push([state, action + [child[1]], cost], cost)
    # raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    explored = set()
    action = []
    cost = 0
    h = 0
    fringe = PQ.PriorityQueue()
    start = problem.startingState()
    fringe.push([start, [], 0], cost + heuristic(start, problem))
    if problem.isGoal(start):
        return fringe
    
    while 1:
        if fringe.isEmpty():  # if empty
            return None
        state, action, cost = fringe.pop()  # pop and store fringe into var
        
        # add to our explored
        if problem.isGoal(state):          # if we are at goal, return action
            return action
        if state not in explored:
            explored.add(state)
        for child in problem.successorStates(state):
            if child[0] not in explored:
                explored.add(child[0])
                # update new state, update list of actions, and increment cost + h
                cost += child[2]
                h = heuristic(child[0], problem)
                fringe.push([child[0], action + [child[1]], cost], cost + h)
    # raise NotImplementedError()