import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_Qnet(nn.Module):
    #nn.Module has nn.Linear whihch does matrix multiplication + bias addition. y = x * w^T + bias
    def __init__(self,input_size,hidden_layer,output_size,):
        super().__init__()
        #Takes 11 inputs, produces hidden_size outputs. This is a fully connected layer.
        self.linear1 = torch.nn.Linear(input_size,hidden_layer) #Transform from state to features
        #Takes hidden_size inputs, produces 4 outputs (Q-values for each action).
        self.linear2 = torch.nn.Linear(hidden_layer,output_size)
    def forward(self,x):
        #Turns negative numbers into 0, keeps positive numbers...Adds non-linearity so network can learn complex patterns
        x = F.relu(self.linear1(x))
        #Convert to output 4 values
        x = self.linear2(x)
        return x
    def save(self,file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path,file_name) # path = "/home" # Join various path components 
        #print(os.path.join(path, "User/Desktop", "file.txt"))

        torch.save(self.state_dict(),file_name)
        #state_dict() is a inbuilt pytorch function = "Give me ALL the trained weights and biases from the model

class Qtrainer:
  def __init__(self,model,lr,gamma):
      self.model = model
      self.lr = lr
      self.gamma = gamma
      self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
      self.criterion = nn.MSELoss()

  def train_step(self,state,action,reward,next_state,gameover):
      state = torch.tensor(state,dtype=torch.float)
      action = torch.tensor(action,dtype=torch.float)
      reward = torch.tensor(reward,dtype=torch.float)
      next_state = torch.tensor(next_state,dtype=torch.float)
      #(n,x)
      if len(state.shape)==1:
          #Checks if state is 1-dimensional (a single sample)   
          state = torch.unsqueeze(state,0)  #If shape(3,)-->(1,3) 1 sample with 3 feature
          next_state = torch.unsqueeze(next_state,0)
          reward = torch.unsqueeze(reward,0)
          action = torch.unsqueeze(action,0)
          #unsqueeze(dim=0) adds a batch dimension at position 0
          """Neural networks expect batches: (batch_size, features)
A single sample needs to be treated as a batch of size 1"""
          gameover = (gameover,) #Why a tuple? This allows it to be indexed later like an array. When processing batches:
#Single sample: done = (True,)
#Can access: done[0] = True
          pred = self.model(state)

          #For Bellman equation update: New Q-value = reward + γ × max(Q(next_state)) we need a clone (copy of pred)

          target = pred.clone()

          #update the target Q-values for each sample in a batch using the Bellman equation:

          for idx in range(len(gameover)):
              #If done = (True, False, False), then len(done) = 3 → loop runs 3 times
              Q_new = reward[idx]
              if not gameover[idx]:
                  Q_new = reward[idx]+self.gamma*torch.max(self.model(next_state[idx]))
                  #Q_new = immediate_reward + discount_factor × best_future_value

                  # Find Which Action Was Taken
                  

            
          


          


          
     






    
    
        