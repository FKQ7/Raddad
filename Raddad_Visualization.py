import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


"""

    Satellite and Debris Visualization
    Please ensure you have all the dependencies installed:
    pip install numpy matplotlib
    Ensure you are using Python 3.8 as tested.
    Python 3.12 Won't work as tested

"""


# --------------------------
# Simulation parameters
# --------------------------
np.random.seed(45)
timesteps = 200
dt = 1
safety_radius = 1.0 # collision risk distance

# Satellite starts at center, small constant velocity
sat_pos = np.array([0.0, 0.0])
sat_vel = np.array([0.05, 0.02])

# Debris starts FAR away (right side)
debris_pos = np.array([15.0, 5.0])
debris_vel = np.array([-0.04, -0.009])

# Store trajectories
sat_traj = [sat_pos.copy()]
debris_traj = [debris_pos.copy()]

# --------------------------
# Helper: noisy update
# --------------------------
def noisy_update(pos, vel, noise_scale=0.05):
    noise = np.random.normal(0, noise_scale, size=2)
    return pos + vel * dt + noise

# --------------------------
# Simulate trajectories
# --------------------------
for _ in range(timesteps):
    sat_pos = sat_pos + sat_vel * dt
    debris_pos = noisy_update(debris_pos, debris_vel)

    sat_traj.append(sat_pos.copy())
    debris_traj.append(debris_pos.copy())

sat_traj = np.array(sat_traj)
debris_traj = np.array(debris_traj)


# --------------------------
# Visualization
# --------------------------

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-20, 20)
ax.set_ylim(-10, 15)
ax.set_title("Satellite & Debris (Prediction + Collision Detection)")

# Paths
sat_line, = ax.plot([], [], 'b-', lw=2, label="Satellite Path")
debris_line, = ax.plot([], [], 'r--', lw=2, label="Debris Path")
sat_dot, = ax.plot([], [], 'bo', markersize=6)
debris_dot, = ax.plot([], [], 'ro', markersize=6)


# Safety radius circle
safety_circle = plt.Circle((0, 0), safety_radius, color='gray', alpha=0.3)
ax.add_patch(safety_circle)

ax.legend()

# --------------------------
# Animation functions
# --------------------------
def init():
    sat_line.set_data([], [])
    debris_line.set_data([], [])
    sat_dot.set_data([], [])
    debris_dot.set_data([], [])
    return sat_line, debris_line, sat_dot, debris_dot, safety_circle

def animate(i):
    # Update paths
    sat_line.set_data(sat_traj[:i, 0], sat_traj[:i, 1])
    debris_line.set_data(debris_traj[:i, 0], debris_traj[:i, 1])

    # Update positions
    sat_dot.set_data(sat_traj[i, 0], sat_traj[i, 1])
    debris_dot.set_data(debris_traj[i, 0], debris_traj[i, 1])

    # Move safety circle with satellite
    safety_circle.center = (sat_traj[i, 0], sat_traj[i, 1])


    return sat_line, debris_line, sat_dot, debris_dot, safety_circle

# --------------------------
# Run animation
# --------------------------
ani = animation.FuncAnimation(fig, animate, frames=timesteps,
                              init_func=init, blit=True, interval=80)

plt.show()
