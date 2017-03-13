# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from capture import GameState
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def maxValueMiniMax(self, gameState, treeDepth):        # the max value of the minimax tree with gameState and treeDepth
      if state.isOver():            # terminal state and if true, then return the evaluation function since it finished
        return self.evaluate(gameState, Directions.STOP) #goes to the scoreEvaluation Function
      
      value = float('-Infinity') #value is negative infinity
      for a in gameState.getLegalActions(0): #for a in the the legal actions of the pacman
        #if a != Directions.STOP:
          succ = gameState.generateSuccessor(0,a) #generate the successor of actions
          minValueMAX = minValueMiniMax(succ, treeDepth, 1)
          value = max(value, minValueMAX) #assign the value to the max of value and minValue containing succ
                                                                #tree depth, and a ghost
      return value  #returns the value

  #multiple min layers (one for each ghost) for every max layer
  def minValueMiniMax(self, gameState, treeDepth, ghost):  # the min value of the minimax tree with gamestate, treeDepth, and ghost
      if state.isOver() or treeDepth == 0:    #terminal state of the program and treeDepth == 0 since it is to prevent max recursion 
                                          # and if true, it return the evaluationFunction of gameState
        return self.evaluate(gameState, Directions.STOP) # goes to the scorceEvaluation function

      value = float('Infinity')                        # in min value, the value is infinity
      totalAgents = gameState.getNumAgents()-1         # the total agents is the total number of agents in the game
      for a in gameState.getLegalActions(ghost):       # for the actions in the legal actions of the ghost
        #if a != Directions.STOP:
        if ghost == totalAgents:                       # if the ghost is equal to totalAgents, then find the min of value and maxValue
          succ = self.getSuccessor(ghost,a)
          maxValue = maxValueMiniMax(succ, treeDepth-1) # the minus one prevent the exceeding the maximum recursion depth 
          value = min(value, maxValue) # the value is the min of value and maxValue with it succ, and treedepth-1
        else:                           #if the ghost doesn't equal to the total agents in the game
          succ = self.getSuccessor(ghost,a)
          minValue = minValueMiniMax(succ, treeDepth, ghost+1)
          value = min(value, minValue) #the value would be the min of value and minValue with succ, treedDepth, and ghost
      return value #returns the value

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor) #compute the score from the successor state
    myPos = successor.getAgentState(self.index).getPosition()

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    enemies = [successor.getAgentState(e) for e in self.getOpponents(successor)]
    ghosts = [g for g in enemies if not g.isPacman and g.getPosition() != None]
    invaders = [i for i in enemies if i.isPacman and i.getPosition() != None]
    
    if len(invaders) != 0:
      for i in invaders:
        invadePos = [i.getPosition()]
    else:
      for g in ghosts:
        ghostPos = [g.getPosition()]

    # if the other team paceman is a distance away to my pacman
    if len(invaders) != 0 and not successor.getAgentState(self.index).isPacman:
        for pacman in invadePos:
          distanceToPacman = min([self.getMazeDistance(myPos, pacman)])
        if distanceToPacman <= 1:
            features['distToInvader'] = distanceToPacman
    return features

  def getWeights(self, gameState, action):
    return {'distToInvader': 100,'successorScore': 100, 'distanceToFood': -1}

class DefensiveAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFood(successor).asList()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    #go to the nearest food when there are no invaders
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    
    #invade pacman
    ghosts = [g for g in enemies if not g.isPacman and g.getPosition() != None]
    if len(invaders) != 0:
      for i in invaders:
        invadePos = [i.getPosition()]
    else:
      for g in ghosts:
        ghostPos = [g.getPosition()]
    if len(invaders) != 0 and not successor.getAgentState(self.index).isPacman:
        for pacman in invadePos:
          distanceToPacman = min([self.getMazeDistance(myPos, pacman)])
        if distanceToPacman <= 5:
            features['distanceToInvader'] = distanceToPacman

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
