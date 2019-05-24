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

    def act(self, state, hx, cx):
        state = torch.from_numpy(state.T.copy()).float().unsqueeze(0).to(self.DEVICE)
        hx = torch.from_numpy( hx ).float().to(self.DEVICE)
        cx = torch.from_numpy( cx ).float().to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():                
            action, log_prob, _, _, hx, cx = self.model(state, hx, cx)
        self.model.train()

        action = action.cpu().detach().numpy().item()
        log_prob = log_prob.cpu().detach().numpy().item()
        hx = hx.cpu().data.numpy()
        cx = cx.cpu().data.numpy()

        return action, log_prob, hx, cx
