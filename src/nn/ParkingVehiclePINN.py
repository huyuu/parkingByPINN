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
    def generate_vehicle_gif(model, device, save_path, wheelBase=2.5,
                             vehicle_length=4.5, vehicle_width=2.0):
        """Generate a GIF showing the vehicle body moving along the final trajectory."""
        model.eval()
        t_dense = torch.linspace(0, 60, 300, device=device).unsqueeze(1)
        with torch.no_grad():
            x, y, theta, v, a, delta, omega, alpha = model(t_dense)
        x = x.cpu().numpy().flatten()
        y = y.cpu().numpy().flatten()
        theta = theta.cpu().numpy().flatten()
        delta = delta.cpu().numpy().flatten()

        # Compute axis limits with margin
        all_coords = np.concatenate([x, y])
        lo, hi = all_coords.min(), all_coords.max()
        margin = max(2.0, (hi - lo) * 0.15)
        lim = (lo - margin, hi + margin)

        fig, ax = plt.subplots(figsize=(8, 8))
        trail_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.4)

        # rear_axle_offset: distance from vehicle center to rear axle
        rear_axle_offset = vehicle_length / 2 - 0.5  # rear axle ~0.5m from rear

        def draw_vehicle_rect(ax, cx, cy, th):
            """Return a rotated rectangle patch centered at (cx, cy) with heading th."""
            cos_t, sin_t = np.cos(th), np.sin(th)
            # corners relative to center
            hw, hl = vehicle_width / 2, vehicle_length / 2
            corners = np.array([
                [-hl, -hw],
                [ hl, -hw],
                [ hl,  hw],
                [-hl,  hw],
            ])
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated = corners @ rot.T + np.array([cx, cy])
            from matplotlib.patches import Polygon
            return Polygon(rotated, closed=True, facecolor='skyblue',
                           edgecolor='black', linewidth=1.5, zorder=5)

        def update(frame):
            ax.clear()
            cx, cy, th, dt = x[frame], y[frame], theta[frame], delta[frame]

            # Trail
            ax.plot(x[:frame+1], y[:frame+1], 'b-', linewidth=1, alpha=0.4)
            # Full path ghost
            ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.2)

            # Start and target markers
            ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
            ax.plot(10.0, 10.0, 'r*', markersize=14, label='Target')

            # Vehicle body
            body = draw_vehicle_rect(ax, cx, cy, th)
            ax.add_patch(body)

            cos_t, sin_t = np.cos(th), np.sin(th)

            # Heading arrow from center
            arrow_len = vehicle_length * 0.6
            ax.annotate('', xy=(cx + arrow_len * cos_t, cy + arrow_len * sin_t),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        zorder=10)

            # Front axle position
            front_x = cx + (vehicle_length / 2 - 0.5) * cos_t
            front_y = cy + (vehicle_length / 2 - 0.5) * sin_t

            # Steering angle indicator (front wheel direction)
            steer_angle = th + dt
            steer_len = 1.2
            ax.annotate('', xy=(front_x + steer_len * np.cos(steer_angle),
                                front_y + steer_len * np.sin(steer_angle)),
                        xytext=(front_x, front_y),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        zorder=10)

            # Rear axle marker
            rear_x = cx - (vehicle_length / 2 - 0.5) * cos_t
            rear_y = cy - (vehicle_length / 2 - 0.5) * sin_t
            ax.plot(rear_x, rear_y, 'ko', markersize=4, zorder=10)

            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            t_val = frame / (len(x) - 1) * 60.0
            ax.set_title(f'Vehicle Motion  t = {t_val:.1f}s')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Sample every few frames to keep GIF size reasonable
        step = max(1, len(x) // 100)
        frames = list(range(0, len(x), step))
        if frames[-1] != len(x) - 1:
            frames.append(len(x) - 1)

        anim = FuncAnimation(fig, update, frames=frames, interval=100)
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close(fig)
        print(f"Saved vehicle animation to {save_path}")

    @staticmethod
    def plot_loss_history(loss_history, save_path):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        epochs = np.arange(len(loss_history['total']))

        # Top: total loss
        ax1.plot(epochs, loss_history['total'], 'k-', linewidth=0.5, alpha=0.3, label='Total')
        # Smoothed version
        window = min(500, len(epochs) // 10)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(loss_history['total'], kernel, mode='valid')
            ax1.plot(np.arange(len(smoothed)) + window // 2, smoothed, 'k-', linewidth=1.5, label=f'Total (smoothed)')
        ax1.set_ylabel('Total Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Training Loss')

        # Bottom: individual components
        colors = {'physics': 'blue', 'boundary': 'red', 'constraints': 'green'}
        for name, color in colors.items():
            ax2.plot(epochs, loss_history[name], color=color, linewidth=0.5, alpha=0.3)
            if window > 1:
                smoothed = np.convolve(loss_history[name], kernel, mode='valid')
                ax2.plot(np.arange(len(smoothed)) + window // 2, smoothed, color=color, linewidth=1.5, label=name)
            else:
                ax2.plot(epochs, loss_history[name], color=color, linewidth=1.0, label=name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved loss plot to {save_path}")

    @staticmethod
    def generate_gif(snapshots, save_path, start_state=None, target_state=None):
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
            ax_xy.plot(snap['x'][-1], snap['y'][-1], 'bs', markersize=8, label='End')
            if target_state is not None:
                ax_xy.plot(target_state.x, target_state.y, 'r*', markersize=15, label='Target')
            else:
                ax_xy.plot(10.0, 10.0, 'r*', markersize=15, label='Target')

            # Heading arrows for start and target states
            arrow_len = max(1.0, (xy_max - xy_min) * 0.08)
            if start_state is not None:
                ax_xy.annotate('', xy=(start_state.x + arrow_len * np.cos(start_state.theta),
                                       start_state.y + arrow_len * np.sin(start_state.theta)),
                               xytext=(start_state.x, start_state.y),
                               arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
                               zorder=10)
            if target_state is not None:
                ax_xy.annotate('', xy=(target_state.x + arrow_len * np.cos(target_state.theta),
                                       target_state.y + arrow_len * np.sin(target_state.theta)),
                               xytext=(target_state.x, target_state.y),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                               zorder=10)

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
        target_state = VehicleState(x=10.0, y=10.0, theta=0.0, v=0.0, a=0.0, delta=0.0, omega=0.0, alpha=0.0)

        # Dense time grid for visualization snapshots
        t_dense = torch.linspace(0, 60, 200, device=device).unsqueeze(1)
        snapshots = []
        total_epochs = 100000
        capture_interval = 200

        loss_history = {'total': [], 'physics': [], 'boundary': [], 'constraints': []}

        for epoch in range(total_epochs):
            optimizer.zero_grad()

            # boundary conditions: start and end states
            t_start = torch.tensor([[0.0]], device=device, requires_grad=True)
            t_goal = torch.tensor([[60.0]], device=device, requires_grad=True)

            x0, y0, theta0, v0, a0, delta0, omega0, alpha0 = model(t_start)
            xT, yT, thetaT, vT, aT, deltaT, omegaT, alphaT = model(t_goal)

            loss_start = (x0 - start_state.x)**2 + (y0 - start_state.y)**2 + 100 *(theta0 - start_state.theta)**2 + (v0 - start_state.v)**2 + (a0 - start_state.a)**2 + (delta0 - start_state.delta)**2 + (omega0 - start_state.omega)**2 + (alpha0 - start_state.alpha)**2
            loss_goal = (xT - target_state.x)**2 + (yT - target_state.y)**2 + 100 * (thetaT - target_state.theta)**2 + (vT - target_state.v)**2

            loss_boundary = loss_start + loss_goal

            # physics constraints: kinematic equations
            t_colloc = torch.rand((100, 1), device=device) * 60.0
            t_colloc = t_colloc.requires_grad_(True)
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

            loss_history['total'].append(loss.item())
            loss_history['physics'].append(loss_physics.item())
            loss_history['boundary'].append(loss_boundary.item())
            loss_history['constraints'].append(loss_constraints.item())

            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Total Loss: {loss.item():.4f} | "
                      f"Physics: {loss_physics.item():.4f} | "
                      f"Boundary: {loss_boundary.item():.4f} | "
                      f"Constraints: {loss_constraints.item():.4f}")

            # Capture snapshots for visualization
            if epoch == 0 or epoch % capture_interval == 0 or epoch == total_epochs - 1:
                snapshots.append(self.capture_snapshot(model, device, epoch, t_dense))

        # Plot loss history
        loss_plot_path = os.path.join(os.path.dirname(__file__), 'loss_history.png')
        self.plot_loss_history(loss_history, loss_plot_path)

        # Generate animated GIF after training
        gif_path = os.path.join(os.path.dirname(__file__), 'training_progress.gif')
        self.generate_gif(snapshots, gif_path, start_state=start_state, target_state=target_state)

        # Generate vehicle motion animation
        vehicle_gif_path = os.path.join(os.path.dirname(__file__), 'vehicle_motion.gif')
        self.generate_vehicle_gif(model, device, vehicle_gif_path, wheelBase=wheelBase)


if __name__ == "__main__":
    parking_vehicle_pinn = ParkingVehiclePINN()
    parking_vehicle_pinn.run()