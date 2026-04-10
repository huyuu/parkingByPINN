import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

    @staticmethod
    def capture_snapshot(model, device, epoch, t_dense):
        model.eval()
        with torch.no_grad():
            x, y, theta, v, a, delta, omega, alpha = model(t_dense)
        model.train()
        t_np = t_dense.cpu().numpy().flatten()
        return {
            'epoch': epoch,
            't': t_np,
            'x': x.cpu().numpy().flatten(),
            'y': y.cpu().numpy().flatten(),
            'theta': theta.cpu().numpy().flatten(),
            'v': v.cpu().numpy().flatten(),
            'a': a.cpu().numpy().flatten(),
            'delta': delta.cpu().numpy().flatten(),
            'omega': omega.cpu().numpy().flatten(),
            'alpha': alpha.cpu().numpy().flatten(),
        }

    @staticmethod
    def generate_gif(snapshots, save_path):
        state_names = ['x', 'y', 'theta', 'v', 'a', 'delta', 'omega', 'alpha']

        # Pre-compute axis limits across all snapshots
        xy_min = min(s['x'].min() for s in snapshots)
        xy_min = min(xy_min, min(s['y'].min() for s in snapshots))
        xy_max = max(s['x'].max() for s in snapshots)
        xy_max = max(xy_max, max(s['y'].max() for s in snapshots))
        margin = max(1.0, (xy_max - xy_min) * 0.1)
        xy_lim = (min(xy_min - margin, -2), max(xy_max + margin, 12))

        state_limits = {}
        for name in state_names:
            lo = min(s[name].min() for s in snapshots)
            hi = max(s[name].max() for s in snapshots)
            pad = max(0.1, (hi - lo) * 0.1)
            state_limits[name] = (lo - pad, hi + pad)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

        ax_xy = fig.add_subplot(gs[:, :2])
        axes_state = []
        for i in range(4):
            for j in range(2):
                axes_state.append(fig.add_subplot(gs[i, 2 + j]))

        def update(frame_idx):
            snap = snapshots[frame_idx]

            # Left panel: x-y trajectory
            ax_xy.clear()
            ax_xy.plot(snap['x'], snap['y'], 'b-', linewidth=1.5)
            ax_xy.plot(snap['x'][0], snap['y'][0], 'go', markersize=10, label='Start')
            ax_xy.plot(10.0, 10.0, 'r*', markersize=15, label='Target')
            ax_xy.set_xlim(xy_lim)
            ax_xy.set_ylim(xy_lim)
            ax_xy.set_xlabel('x')
            ax_xy.set_ylabel('y')
            ax_xy.set_title(f'Trajectory (Epoch {snap["epoch"]})')
            ax_xy.set_aspect('equal', adjustable='box')
            ax_xy.legend(loc='upper left')
            ax_xy.grid(True, alpha=0.3)

            # Right panel: state variables vs time
            for k, name in enumerate(state_names):
                axes_state[k].clear()
                axes_state[k].plot(snap['t'], snap[name], 'b-', linewidth=1.0)
                axes_state[k].set_title(name, fontsize=9)
                axes_state[k].set_ylim(state_limits[name])
                axes_state[k].tick_params(labelsize=7)
                axes_state[k].grid(True, alpha=0.3)
                if k >= 6:
                    axes_state[k].set_xlabel('t', fontsize=8)

            fig.suptitle(f'Training Progress — Epoch {snap["epoch"]}', fontsize=14)

        anim = FuncAnimation(fig, update, frames=len(snapshots), interval=200)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=5))
        plt.close(fig)
        print(f"Saved training animation to {save_path}")

    def run(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = KinematicNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        wheelBase = 2.5

        start_state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0, a=0.0, delta=0.0, omega=0.0, alpha=0.0)
        target_state = VehicleState(x=10.0, y=10.0, theta=0.0, v=10.0, a=0.0, delta=0.0, omega=0.0, alpha=0.0)

        # Dense time grid for visualization snapshots
        t_dense = torch.linspace(0, 60, 200, device=device).unsqueeze(1)
        snapshots = []
        total_epochs = 10000
        capture_interval = 200

        for epoch in range(total_epochs):
            optimizer.zero_grad()

            # boundary conditions: start and end states
            t_start = torch.tensor([[0.0]], device=device, requires_grad=True)
            t_goal = torch.tensor([[60.0]], device=device, requires_grad=True)

            x0, y0, theta0, v0, a0, delta0, omega0, alpha0 = model(t_start)
            xT, yT, thetaT, vT, aT, deltaT, omegaT, alphaT = model(t_goal)

            loss_start = (x0 - start_state.x)**2 + (y0 - start_state.y)**2 + (theta0 - start_state.theta)**2 + (v0 - start_state.v)**2 + (a0 - start_state.a)**2 + (delta0 - start_state.delta)**2 + (omega0 - start_state.omega)**2 + (alpha0 - start_state.alpha)**2
            loss_goal = (xT - target_state.x)**2 + (yT - target_state.y)**2 + (thetaT - target_state.theta)**2 + (vT - target_state.v)**2 + (aT - target_state.a)**2 + (deltaT - target_state.delta)**2 + (omegaT - target_state.omega)**2 + (alphaT - target_state.alpha)**2

            loss_boundary = loss_start + loss_goal

            # physics constraints: kinematic equations
            t_colloc = torch.rand((100, 1), requires_grad=True, device=device)
            x, y, theta, v, a, delta, omega, alpha = model(t_colloc)

            dx_dt = self.get_gradient(x, t_colloc)
            dy_dt = self.get_gradient(y, t_colloc)
            dtheta_dt = self.get_gradient(theta, t_colloc)
            dv_dt = self.get_gradient(v, t_colloc)
            da_dt = self.get_gradient(a, t_colloc)
            ddelta_dt = self.get_gradient(delta, t_colloc)
            domega_dt = self.get_gradient(omega, t_colloc)
            dalpha_dt = self.get_gradient(alpha, t_colloc)

            # Formulate residuals (Left side minus Right side = 0)
            res_x = dx_dt - (v * torch.cos(theta))
            res_y = dy_dt - (v * torch.sin(theta))
            res_theta = dtheta_dt - ((v / wheelBase) * torch.tan(delta))
            res_v = dv_dt - a
            res_a = da_dt - 0.0
            res_delta = ddelta_dt - omega
            res_omega = domega_dt - alpha

            loss_physics = torch.mean(res_x**2 + res_y**2 + res_theta**2 + res_v**2 + res_a**2 + res_delta**2 + res_omega**2 + dalpha_dt**2)
            
            # penalize extreme steering angles to prevent tan(delta) from blowing up
            loss_constraints = torch.mean(torch.relu(torch.abs(delta) - 0.6))

            loss = (100.0 * loss_boundary) + \
                (1.0 * loss_physics) + \
                (1.0 * loss_constraints)

            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Total Loss: {loss.item():.4f} | "
                      f"Physics: {loss_physics.item():.4f} | "
                      f"Boundary: {loss_boundary.item():.4f} | "
                      f"Constraints: {loss_constraints.item():.4f}")

            # Capture snapshots for visualization
            if epoch == 0 or epoch % capture_interval == 0 or epoch == total_epochs - 1:
                snapshots.append(self.capture_snapshot(model, device, epoch, t_dense))

        # Generate animated GIF after training
        gif_path = os.path.join(os.path.dirname(__file__), 'training_progress.gif')
        self.generate_gif(snapshots, gif_path)


if __name__ == "__main__":
    parking_vehicle_pinn = ParkingVehiclePINN()
    parking_vehicle_pinn.run()