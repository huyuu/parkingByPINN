


class VehicleState():
    def __init__(self, x, y, theta, v, a, delta, omega, alpha):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.a = a
        self.delta = delta
        self.omega = omega
        self.alpha = alpha

    def to_tensor(self, device):
        return torch.tensor([self.x, self.y, self.theta, self.v, self.a, self.delta, self.omega, self.alpha], device=device)
    
    def from_tensor(self, tensor):
        self.x = tensor[0]
        self.y = tensor[1]
        self.theta = tensor[2]
        self.v = tensor[3]
        self.a = tensor[4]
        self.delta = tensor[5]
        self.omega = tensor[6]
        self.alpha = tensor[7]
        return self

