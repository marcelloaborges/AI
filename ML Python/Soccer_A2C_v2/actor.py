import torch

from memory import Memory

class Actor:

    def __init__(self,
        device,
        key,
        model, 
        batch_size):

        self.DEVICE = device
        self.KEY = key                

        # Neural model
        self.model = model        

        # Shared memory
        self.memory = Memory(batch_size)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
                
        self.model.eval()
        with torch.no_grad():
            action, prob, _, _ = self.model(state)            
        self.model.train()

        action = action.cpu().detach().numpy().squeeze()
        prob = prob.cpu().detach().numpy().squeeze()

        return action, prob

    def step(self, state, action, action_prob, reward, next_state):        
        # Save experience / reward
        self.memory.add(state, action, action_prob, reward, next_state)

    def enough_experiences(self):
        if self.memory.enough_experiences():
            return False

        return True

    def experiences(self):
        # Get the experiences and clear the memory
        return self.memory.experiences()        

