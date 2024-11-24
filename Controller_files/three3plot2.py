import numpy as np
import matplotlib.pyplot as plt
import control as ct

# --- Constants Section ---
g = 9.81  # Acceleration due to gravity in m/s^2
mass = 685000  # Mass of SpaceX Superheavy in kilograms
Cd = 0.75  # Drag coefficient (dimensionless)
rho = 1.225  # Air density at sea level in kg/m^3
A = 1.0  # Cross-sectional area of the rocket in square meters
engine_thrust = 3.2 * 1000000  # Thrust per engine (N)
num_engines = 3  # Number of engines
max_thrust = engine_thrust * num_engines  # Maximum thrust force available (N)
min_thrust = 0  # The rocket cannot generate a downwards force

# --- Gimble angles ---
theta = 0  # Tilt angle in degrees
phi = 0    # Azimuth angle in degrees

# --- Simulation Parameters ---
dt = 0.1  # Time step in seconds
total_time = 120  # Total time to simulate in seconds

# --- Initial Rocket Conditions ---
altitude = 1000  # Starting altitude in meters
velocity = -50  # Initial velocity in m/s (negative indicates falling down)
thrust = 0  # Initial thrust applied by engines in Newtons

# Initialize lists to store time, position, velocity, etc.
time_list = [0]
position_list = [[0, 0, altitude]]  # Starting position [x, y, z]
velocity_list = [[0, 0, velocity]]  # Initial velocity [vx, vy, vz]
thrust_list = [0]

# --- Drag Force Calculation ---
def calculate_drag(velocity):
    """Calculate the drag force based on the current velocity."""
    return 0.5 * Cd * rho * A * velocity**2 * np.sign(velocity)

# --- Thrust Vector Calculation ---
def thrust_vector(thrust, theta, phi, max_thrust):
    thrust = np.clip(thrust, min_thrust, max_thrust)
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    return np.array([
        thrust * np.sin(theta_rad) * np.cos(phi_rad),  # x-component
        thrust * np.sin(theta_rad) * np.sin(phi_rad),  # y-component
        thrust * np.cos(theta_rad)                    # z-component
    ])

# --- PID Controller Implementation ---
def pid_control(setpoint, current_value, integral, derivative, dt, Kp, Ki, Kd):
    error = setpoint - current_value
    integral += error * dt
    derivative = (error - derivative) / dt
    return Kp * error + Ki * integral + Kd * derivative, integral, derivative

# --- Main Simulation Loop ---
# PID tuning variables
k_scale = 18200
Kp, Ki, Kd = 10 * k_scale, 0.5 * k_scale, -1 * k_scale  # Initial guesses for gains

# Initialize PID parameters
integral = 0
derivative = 0
error = 900

# Run simulation for each time step
for t in np.arange(0, total_time, dt):
    # Update thrust with PID control based on altitude error
    thrust_adjustment, integral, derivative = pid_control(altitude, position_list[-1][2], integral, derivative, dt, Kp, Ki, Kd)
    
    # Calculate thrust vector based on gimbal angles and thrust adjustment
    thrust_vector_adjusted = thrust_vector(thrust_adjustment, theta, phi, max_thrust)
    
    # Calculate the forces on the rocket (thrust, drag, gravity)
    drag = calculate_drag(velocity_list[-1][2])
    gravity = mass * g
    net_force = thrust_vector_adjusted[2] - drag - gravity  # Vertical forces
    
    # Compute acceleration and update velocity
    acceleration = net_force / mass
    velocity_new = velocity_list[-1][2] + acceleration * dt
    
    # Update position using velocity
    position_new = position_list[-1][2] + velocity_new * dt
    
    # Append updated values to lists
    position_list.append([0, 0, position_new])
    velocity_list.append([0, 0, velocity_new])
    thrust_list.append(thrust_vector_adjusted[2])
    time_list.append(t)
    
    if (position_new <= 0):
        break

# --- Plot Results ---
plt.figure(figsize=(12, 8))

# Altitude vs Time
plt.subplot(3, 1, 1)
plt.plot(time_list, np.array(position_list)[:, 2], label='Altitude (m)')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Rocket Altitude vs. Time')
plt.grid()

# Velocity vs Time
plt.subplot(3, 1, 2)
plt.plot(time_list, np.array(velocity_list)[:, 2], label='Velocity (m/s)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Rocket Velocity vs. Time')
plt.grid()

# Thrust vs Time
plt.subplot(3, 1, 3)
plt.plot(time_list, thrust_list, label='Thrust (N)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.title('Rocket Thrust vs. Time')
plt.grid()

plt.tight_layout()
plt.show()
