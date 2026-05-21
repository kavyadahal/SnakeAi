import torch
import numpy as np
import random
from collections import deque
from game import SnakeGameAI,Point,Direction

MAX_MEMORY = 100_000 #Store last 100k experience
BATCH_SIZE = 1000 #thousand samples
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0 #Count compoleted games
        """a hyperparameter that controls the probability that an AI agent takes a random action to discover its environment, 
        rather than picking the action it currently thinks is best"""
        self.epsilon = 0 #Exploration rate(0=always exploit)
       
        self.gama = 0.99 #Discount factor: 0.99 --> 0.9-->gent can see 10 steps aheaf, 0.99--> 100 steps
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() #stores experience
        self.model = None #TODO  #Neural network
        self.trainer = None #TODO #Training algorithm
       
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover)) # popleft if MAX_MEMORY is reached
        """Stores experiences for later learning. The deque automatically removes oldest entries when it reaches 100,000 items."""
    
        """Takes random batch of 1000 past experiences
Trains the model on these old experiences (improves stability)"""
    def train_long_memory(self):
    #IF MEMORY HAS MORE THAN 1000 EXPERIENCE THEN TAKE PICK 1000 EXPERIENCE RANDOMLY 
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #returns list of tuples
        else:
            #IF LESS THAN 1000 EXPERIENCE
            mini_sample = self.memory
        # unpacks and separates the batch of experiences into individual components.
        state, action, reward, next_state, gameover = zip(*mini_sample)
        self.trainer.train_step(state, action, reward, next_state,gameover)
     
#Trains immediately after each game step on this single experience.
    def train_short_memory(self,state,action,reward,next_state,gameover):
        self.trainer.train_step(state, action, reward, next_state,gameover)

        

    def get_action(self,state):
        #random moves : tradeoff exploration / exploitation

        self.epsilon = 80 - self.n_games

        #initialize action vector

        final_move = [0,0,0]
        """[0, 0, 0]  →  No action (straight)
[1, 0, 0]  →  Turn LEFT
[0, 1, 0]  →  Go STRAIGHT
[0, 0, 1]  →  Turn RIGHT"""

#Should the snake take a RANDOM action or a SMART action
#45 < 79?  → YES ✓ EXPLORE (take random action)
#150 < 79?  → YNO ✓ EXPLOIT (take model action)


        if random.randint(0,200)<self.epsilon:
            move = (0,2)
            final_move(move) = 1

        else:
            #state eturns: numpy array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1])
            #so use state 0 as The neural network expects a PyTorch tensor, but state is a numpy array.
            state0 = torch.tensor(state,dtype=torch.float)
            #Smart action:
            prediction = self.model.predict(state0)



          

def train():
    plot_scores = [] #List for score
    plot_mean_scores = [] #Mean of score
    total_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)


        final_move = agent.get_action(state_old)


        reward,gameover, score = game.play_step(final_move)

        state_new = agent.get_state(game)



        #Train short memory:
        agent.train_short_memory(state_old,final_move,reward,state_new,gameover)
        #remeber
        agent.remeber(state_old,final_move,reward,state_new,gameover)
        if gameover:
            #train the long memory(Expereince replay or replay memory)
            game.reset()
            agent.n_games +=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save() #Calls state_dict to store the weights and bias

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__=='main':
    train( )
