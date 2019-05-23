import torch

class Agent:

    def __init__(
        self, 
        device,
        model
        ):

        self.DEVICE = device

        # NEURAL MODEL
        self.model = model

    def act(self, state):
        state = torch.from_numpy(state.T.copy()).float().unsqueeze(0).to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():                
            action, log_prob, _, _ = self.model(state)                    
        self.model.train()

        action = action.cpu().detach().numpy().item()
        log_prob = log_prob.cpu().detach().numpy().item()

        return action, log_prob
