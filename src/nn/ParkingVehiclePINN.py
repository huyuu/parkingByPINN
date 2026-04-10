import torch
import torch.nn as nn
import numpy as np

from States import VehicleState


class KinematicNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: time (t)
        # Outputs:
        # x position
        # y position
        # theta vehicle orientation
        # v velocity
        # a acceleration
        # delta steering angle
        # omega angular velocity of steering angle
        # alpha angular acceleration of steering angle

        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 8)
        )

    def forward(self, t):
        out = self.net(t)
        x = out[:, 0:1]
        y = out[:, 1:2]
        theta = out[:, 2:3]
        v = out[:, 3:4]
        a = out[:, 4:5]
        delta = out[:, 5:6]
        omega = out[:, 6:7]
        alpha = out[:, 7:8]
        return x, y, theta, v, a, delta, omega, alpha


class ParkingVehiclePINN():
    def __init__(self):
        pass

    @staticmethod
    def get_gradient(output, input):
        return torch.autograd.grad(
            output, input, 
            grad_outputs=torch.ones_like(output), 
            create_graph=True
        )[0]

    def run(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = KinematicNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        wheelBase = 2.5


        start_state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0, a=0.0, delta=0.0, omega=0.0, alpha=0.0).to_tensor(device)
        target_state = VehicleState(x=10.0, y=10.0, theta=0.0, v=10.0, a=0.0, delta=0.0, omega=0.0, alpha=0.0).to_tensor(device)

        for i in range(1000):
            optimizer.zero_grad()

            # boundary conditions: start and end states
            t_start = torch.tensor([0.0], device=device, requires_grad=True)
            t_goal = torch.tensor([60.0], device=device, requires_grad=True)

            x0, y0, theta0, v0, a0, delta0, omega0, alpha0 = model(t_start)
            xT, yT, thetaT, vT, aT, deltaT, omegaT, alphaT = model(t_goal)

            loss_start = (x0 - start_state.x)**2 + (y0 - start_state.y)**2 + (theta0 - start_state.theta)**2 + (v0 - start_state.v)**2 + (a0 - start_state.a)**2 + (delta0 - start_state.delta)**2 + (omega0 - start_state.omega)**2 + (alpha0 - start_state.alpha)**2
            loss_goal = (xT - target_state.x)**2 + (yT - target_state.y)**2 + (thetaT - target_state.theta)**2 + (vT - target_state.v)**2 + (aT - target_state.a)**2 + (deltaT - target_state.delta)**2 + (omegaT - target_state.omega)**2 + (alphaT - target_state.alpha)**2

            loss_boundary = loss_start + loss_goal

            # physics constraints: kinematic equations

