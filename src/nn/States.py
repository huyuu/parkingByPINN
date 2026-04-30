
import torch

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


class ObstacleState(VehicleState):
    def __init__(self, xMin, xMax, yMin, yMax):
        x = (xMin + xMax) / 2
        y = (yMin + yMax) / 2
        theta = 0.0
        v = 0.0
        a = 0.0
        delta = 0.0
        omega = 0.0
        alpha = 0.0
        super().__init__(x, y, theta, v, a, delta, omega, alpha)
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax

    def to_tensor(self, device):
        return torch.tensor([self.x, self.y, self.theta, self.v, self.a, self.delta, self.omega, self.alpha, self.xMin, self.xMax, self.yMin, self.yMax], device=device)
    
    def from_tensor(self, tensor):
        self.x = tensor[0]
        self.y = tensor[1]
        self.theta = tensor[2]
        self.v = tensor[3]
        self.a = tensor[4]
        self.delta = tensor[5]
        self.omega = tensor[6]
        self.alpha = tensor[7]
        self.xMin = tensor[8]
        self.xMax = tensor[9]
        self.yMin = tensor[10]
        self.yMax = tensor[11]
        return self