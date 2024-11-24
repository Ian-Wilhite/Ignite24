import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import control as ct
import scipy.integrate 

# --- Constants Section ---
g = 9.81  # Acceleration due to gravity in m/s^2
# mass = 150*1000  # Mass of the rocket in kilograms
mass = 685000 # mass of spacex superheavy
Cd = 1.427  # Drag coefficient (dimensionless)
rho = 1.225  # Air density at sea level in kg/m^3
A = 1.0  # Cross-sectional area of the rocket in square meters
engine_thrust = 3.2*1000000 # thurst per engine
num_engines = 3 # number of engines 
max_thrust = engine_thrust * num_engines # Maximum thrust force available, in Newtons 
min_thrust = 0 # the rocket cannot generate a downwards force

# --- Gimble angles ---
theta = 0  # Tilt angle in degrees
phi = 0    # Azimuth angle in degrees

# --- Simulation Parameters ---
dt = 0.1  # Time step in seconds
total_time = 120  # Total time to simulate in seconds

# --- Initial Rocket Conditions ---
altitude = 1000  # Starting altitude in meters above ground
init_velocity = -50  # Initial velocity in m/s (negative indicates falling down)
thrust = 0  # Initial thrust applied by engines in Newtons

braking_altitude = 100 # [m]
stopping_time = 5 # [s]


def calculate_drag(velocity):
    """
    Calculate the drag force based on the current velocity.
    
    Arguments:
        velocity (float): The current velocity of the rocket (m/s).
    
    Returns:
        float: The drag force exerted by the atmosphere.
    """
    return 0.5 * Cd * rho * A * velocity**2 * np.sign(velocity)

def thrust_vector(thrust, theta, phi, max_thrust):
    thrust = np.clip(thrust, min_thrust, max_thrust)
       
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    return np.array([
        thrust * np.sin(theta_rad) * np.cos(phi_rad),  # x-component
        thrust * np.sin(theta_rad) * np.sin(phi_rad),  # y-component
        thrust * np.cos(theta_rad)                    # z-component
    ])

def adjust_gimbal(torque, moment_of_inertia, dt):
    angular_acceleration = torque / moment_of_inertia
    # Update gimbal angles based on desired angular acceleration (simplified)
    theta += angular_acceleration[0] * dt
    phi += angular_acceleration[1] * dt
    return theta, phi

def plot_1d():
        
    # --- Post-Landing Calculations ---
    stopping_distance = 1  # Assume a stopping distance of 1 meter for simplicity
    touchdown_velocity = velocity_list[-1][2]
    if altitude <= 0:
        deceleration = (touchdown_velocity**2) / (2 * stopping_distance)
        impact_force = mass * (deceleration + g)
        normal_force = mass * g + impact_force
    else:
        deceleration = 0
        impact_force = 0
        normal_force = mass * g

    # Display the calculated forces after landing
    print("Rocket Landing Forces:")
    print(f"Touchdown Velocity: {abs(touchdown_velocity):.2f} m/s")
    print(f"Deceleration during landing: {deceleration:.2f} m/s^2")
    print(f"Impact Force: {impact_force:.2f} N")
    print(f"Normal Force at landing: {normal_force:.2f} N")

    # --- Plotting Section ---
    plt.figure(figsize=(10, 8))

    # Altitude Plot
    plt.subplot(4, 1, 1)
    # print(np.shape(np.array(position_list).transpose()[:][2]))
    plt.plot(time_list, np.array(position_list).transpose()[:][2], label='Altitude (m)')
    plt.plot(time_list, np.array(targ_alts), label='targ Altitude (m)')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    # plt.ylim(0, 1000)
    plt.title('Rocket Altitude vs. Time')
    plt.grid()
    plt.legend()

    # Velocity Plot
    plt.subplot(4, 1, 2)
    plt.plot(time_list, np.array(velocity_list).transpose()[:][2], label='Velocity (m/s)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Rocket Velocity vs. Time')
    plt.grid()
    plt.legend()

    # int Plot
    plt.subplot(4, 1, 3)
    plt.plot(time_list, int_pos, color='g')

    plt.xlabel('Time (s)')
    plt.ylabel('int of pos (m*m)')
    # plt.ylim(-max_thrust, max_thrust)
    plt.title('Rocket Thrust vs. Time')
    plt.grid()
    plt.legend()
    
    # Thrust Plot
    plt.subplot(4, 1, 4)
    plt.plot(time_list, np.array(thrust_list).transpose()[:][2], label='Thrust (N)', color='g')
    plt.plot(time_list, np.ones(len(thrust_list)) * mass * g, label='gravity (N)', color='r')
    plt.plot(time_list, np.ones(len(thrust_list)) * max_thrust, label='max thrust (N)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    # plt.ylim(-max_thrust, max_thrust)
    plt.title('Rocket Thrust vs. Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    # plt.show()

def plot_rocket_static(position, thrust):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert position and thrust to NumPy arrays
    position = np.array(position)
    thrust = np.array(thrust)
    
    # Normalize thrust for visualization
    # thrust_magnitude = np.linalg.norm(thrust, axis=1)
    # scaled_thrust = (thrust / thrust_magnitude[:, None]) * (10 * thrust_magnitude / max_thrust)[:, None]
    
    # Create thrust line segments for plotting
    thrust_start = position  # Starting points are the positions
    thrust_end = position + thrust  # End points are scaled thrust vectors
    
    # Plot each thrust vector
    for start, end in zip(thrust_start, thrust_end):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-', label="Thrust Vector")
    
    # Axes limits and labels
    # ax.set_xlim([-100, 100])
    # ax.set_ylim([-100, 100])
    # ax.set_zlim([0, 1100])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_rocket_animated(position, thrust, dt):
    position = np.array(position)
    thrust = np.array(thrust)

    # Normalize and scale thrust for visualization
    thrust_magnitude = np.linalg.norm(thrust, axis=1)
    thrust_magnitude[thrust_magnitude == 0] = 1  # Prevent division by zero
    scaled_thrust = (thrust / thrust_magnitude[:, None]) * 10  # Scale for visibility

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up axes limits and labels
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([0, 1200])  # Adjust based on max altitude
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initialize lines for the rocket path and thrust vector
    rocket_line, = ax.plot([], [], [], 'r-', label="Rocket Path")
    thrust_line, = ax.plot([], [], [], 'b-', label="Thrust Vector")
    ax.legend()

    def update(frame):
        # Current position and thrust for this frame
        pos = position[frame]
        thrust_vec = scaled_thrust[frame]

        # Update rocket path
        rocket_line.set_data(position[:frame + 1, 0], position[:frame + 1, 1])
        rocket_line.set_3d_properties(position[:frame + 1, 71.628])

        # Update thrust vector
        thrust_start = pos
        thrust_end = pos + thrust_vec
        thrust_line.set_data([thrust_start[0], thrust_end[0]],
                             [thrust_start[1], thrust_end[1]])
        thrust_line.set_3d_properties([thrust_start[2], thrust_end[2]])

        return rocket_line, thrust_line

    # Create the animation
    frames = len(position)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt * 1000, blit=False)

    plt.show()

def generate_quadratic_with_derivatives(begin, end, num_points):
    """
    Generate a quadratic function with specified start/end points and derivatives.

    Parameters:
    - begin: Tuple (x0, y0, dy0), the starting point and its derivative.
    - end: Tuple (x1, y1, dy1), the ending point and its derivative.
    - num_points: Integer, the number of points to generate (including begin and end).

    Returns:
    - x_values: numpy array of x values.
    - y_values: numpy array of y values corresponding to the quadratic function.
    """
    x0, y0, dy0 = begin
    x1, y1, dy1 = end

    # Solve for the coefficients a, b, c of the quadratic y = ax^2 + bx + c
    A = np.array([
        [x0**2, x0, 1],  # y(x0) = y0
        [x1**2, x1, 1],  # y(x1) = y1
        [2 * x0, 1, 0],  # y'(x0) = dy0
    ])
    B = np.array([y0, y1, dy0])
    a, b, c = np.linalg.solve(A, B)

    # Generate x values and compute corresponding y values
    x_values = np.linspace(x0, x1, num_points)
    y_values = a * x_values**2 + b * x_values + c

    return x_values, y_values

def generate_cubic_with_derivatives(begin, end, num_points):
    """
    Generate a cubic spline function with specified start/end points and derivatives.

    Parameters:
    - begin: Tuple (x0, y0, dy0), the starting point and its derivative.
    - end: Tuple (x1, y1, dy1), the ending point and its derivative.
    - num_points: Integer, the number of points to generate (including begin and end).

    Returns:
    - x_values: numpy array of x values.
    - y_values: numpy array of y values corresponding to the cubic function.
    """
    # Unpack inputs
    x0, y0, dy0 = begin
    x1, y1, dy1 = end

    # Solve for the coefficients a, b, c, d of the cubic y = ax^3 + bx^2 + cx + d
    A = np.array([
        [x0**3, x0**2, x0, 1],  # y(x0) = y0
        [x1**3, x1**2, x1, 1],  # y(x1) = y1
        [3*x0**2, 2*x0, 1, 0],  # y'(x0) = dy0
        [3*x1**2, 2*x1, 1, 0],  # y'(x1) = dy1
    ])
    B = np.array([y0, y1, dy0, dy1])
    a, b, c, d = np.linalg.solve(A, B)

    # Generate x values and compute corresponding y values
    x_values = np.linspace(x0, x1, num_points)
    y_values = a * x_values**3 + b * x_values**2 + c * x_values + d

    return x_values, y_values

# class PIDController:
#     def __init__(self, Kp, Ki, Kd, setpoint):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.setpoint = setpoint
#         self.prev_error = 0
#         self.integral = 0

#     def compute(self, current_value, dt):
#         error = self.setpoint - current_value
#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt
#         self.prev_error = error

#         # Control signal
#         return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
 
#  # --- PID Controller Parameters ---

# Define the PID controller

# Initial conditions
position = np.array([0, 0, altitude], dtype=float)  # [x, y, z] in meters
velocity = np.array([0, 0, init_velocity], dtype=float)
target_altitude = 0  # Desired altitude in meters

time_list = [0]
position_list = [position]
targ_alts = [1000]
velocity_list = [velocity]
int_pos = [0]
acceleration_list = [np.array([0, 0, 0], dtype=float)]
thrust_list = [np.array([0, 0, 0], dtype=float)]

# --- PID Tuning Section ---
# Create a transfer function for the rocket's vertical dynamics
numerator = [1]
denominator = [mass, Cd * rho * A, g]  # Assuming mass, drag, and gravity affect vertical motion
system = ct.TransferFunction(numerator, denominator)

# Target altitude step response
time_steps = np.linspace(0, total_time, int(total_time / dt))
desired_altitude = np.ones(len(time_steps)) * target_altitude

# Tune PID controller
k_scale = 18200
K_p, K_i, K_d = 10 * k_scale, 0.5 * k_scale, -1 * k_scale  # Initial guesses for gains
pid = ct.TransferFunction([K_d, K_p, K_i], [1, 0])  # PID transfer function
closed_loop_system = ct.feedback(pid * system)

print(time_steps)
# Use simulation to tune PID parameters
tuned_response, tuned_time = ct.forced_response(closed_loop_system, time_steps, desired_altitude)

# Visualize the tuning (optional)
plt.figure()
# plt.plot(tuned_time, tuned_response, label="Tuned Altitude Response")
plt.plot(tuned_time, label="Tuned time")
# plt.plot(tuned_response, label="Tuned Response")
# plt.plot(time_steps, tuned_response, label="Tuned Altitude Response2")
plt.plot(time_steps, desired_altitude, '--', label="Target Altitude")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("PID Tuning")
plt.grid()
# plt.show()
print((total_time * dt, altitude, velocity))
print((total_time / dt, 0, 0))
print(total_time / dt)
model = generate_cubic_with_derivatives((total_time * dt, altitude, init_velocity), (total_time / dt, 0, 0), int(total_time / dt))
plt.plot(model)
plt.show()

for t in np.arange(0, total_time, dt):
    # Compute thrust adjustment from PID
    error = position[2] - model[total_time] 
    print(f'thrust math: ({K_p} * {error} = {K_p * error}) + ({K_i} * {np.sum(error * dt)} = {K_i * np.sum(error * dt)}) + ({K_d} * {(error / dt)} = {K_d * (error / dt)})')
    thrust_adjustment = K_p * error + K_i * np.sum(error * dt) + K_d * (error / dt)
    print(f'thrust: {thrust_adjustment}')
    # Compute thrust vector
    T = thrust_vector(thrust_adjustment, theta, phi, max_thrust)
    print(f'thrust v: {T}')

    # Forces
    weight = np.array([0, 0, -mass * g])  # Gravity acts downward
    drag = np.array([0, 0, calculate_drag(velocity[2])])  # Drag affects z-axis
    net_force = T + weight - drag  # Net forces
    print(f'force: {net_force} = T:{T} + weight:{weight} - drag:{drag}')

    # Acceleration
    acceleration = net_force[-1] / mass

    # Update velocity and position
    velocity += acceleration * dt
    position += velocity * dt
    

    # Prevent ground penetration
    if position[2] <= 0:
        position[2] = 0
        velocity[2] = 0
        print(f"Landed at time {t:.2f}s")
        break

    # Record data
    time_list.append(t)
    position_list.append(position.copy())
    velocity_list.append(velocity.copy())
    acceleration_list.append(acceleration.copy())
    int_pos.append(np.sum(error * dt))
    # targ_alts.append(get_target_altitude(total_time, position))
    thrust_list.append(T.copy())


plot_1d()
# # plot_rocket_static(position_list, thrust_list)
plot_rocket_animated(position_list, thrust_list, dt)


