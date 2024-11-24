import numpy as np
import matplotlib.pyplot as plt

# --- Constants Section ---
g = 9.81  # Acceleration due to gravity in m/s^2
mass = 685000 # mass in kg
Cd = 1.427  # Drag coefficient from CFD simulation
rho = 1.225  # Air density at sea level in kg/m^3
A = 0.5 # Cross-sectional area of the rocket in square meters
engine_thrust = 3.2*1000000 # thurst per engine
num_engines = 3 # number of engines 
max_thrust = engine_thrust * num_engines # Maximum thrust in Newtons 
min_thrust = 0 # the rocket cannot generate an upward thrust

# --- Simulation Parameters ---
dt = 0.1  # Time step in seconds.
total_time = 100  # Total time to simulate in seconds.

# --- Initial Rocket Conditions ---
altitude = 1000  # Starting altitude in meters above ground.
velocity = -50  # Initial velocity in m/s (negative indicates falling down).
thrust = 0  # Initial thrust applied by engines in Newtons.
ending_velocity = 0

# --- Data Storage ---
time_list = []        # Store the time at each simulation step.
altitude_list = []    # Track the rocket's altitude over time.
velocity_list = []    # Track the rocket's velocity over time.
thrust_list = []      # Track how thrust changes over time.

# --- PID Controller Variables ---
pid_scale = 9934.21091
Kp = pid_scale * 1.0  # Proportional gain
Ki = pid_scale * 0.1  # Integral gain
Kd = pid_scale * 0.01  # Derivative gain

prev_error = 0  # Previous error for derivative calculation
integral = 0    # Integral of error

# --- Function Definitions ---

def calculate_drag(velocity):
    """
    Calculate the drag force based on the current velocity.
    """
    return 0.5 * Cd * rho * A * velocity**2 if velocity < 0 else 0  # Drag is only calculated when falling.

# --- Main Simulation Loop ---

for t in np.arange(0, total_time, dt):
    
    # --- Force Calculations ---
    weight = mass * g  # Gravitational force (downward)
    drag = calculate_drag(velocity)  # Drag force (depends on the current velocity)
    
    # --- PID Control for Thrust ---
    # Error term: we want to minimize both altitude and velocity to reach a gentle landing.
    # altitude_error = altitude - 0  # Desired altitude is 0 (ground level)
    velocity_error = ending_velocity - velocity   # Desired velocity is 0 (no speed at landing)
    
    # Proportional term
    proportional = Kp * velocity_error
    
    # Integral term (accumulating error)
    integral += velocity_error * dt
    integral_term = Ki * integral
    
    # Derivative term (change in error over time)
    derivative = (velocity_error - prev_error) / dt
    derivative_term = Kd * derivative
    
    # PID control signal (combined control actions)
    print(f'pid: {proportional + integral_term + derivative_term}, p:{proportional}, i:{integral_term}, d:{derivative_term}')
    pid_output = proportional + integral_term + derivative_term
    
    # Apply thrust using PID output to control velocity
    thrust = min(max_thrust, max(0, weight + drag + pid_output))  # Thrust should not exceed max or be negative
    print(f'Thrust from: model: {weight + drag} + error: {pid_output}')
    # --- Net Force Calculation ---
    net_force = thrust - weight - drag  # Total force acting on the rocket

    # --- Update Rocket State ---
    acceleration = net_force / mass  # Using F = ma to calculate acceleration
    velocity += acceleration * dt  # Update velocity
    altitude += velocity * dt  # Update altitude

    # --- Data Recording ---
    time_list.append(t)
    altitude_list.append(altitude)
    velocity_list.append(velocity)
    thrust_list.append(thrust)
    print(f'altitude: {altitude}, velocity: {velocity}, thrust: {thrust}')

    # --- Check Landing Condition ---
    if altitude <= 0:
        touchdown_velocity = velocity
        break  # Exit the loop when the rocket has landed

    prev_error = velocity_error  # Update previous error for next iteration's derivative term

# --- Post-Landing Calculations ---
if altitude <= 0:
    # Calculate the required deceleration to bring the rocket to a full stop
    stopping_distance = 1.0  # Assume a small stopping distance after touchdown for simplicity
    deceleration = (touchdown_velocity**2) / (2 * stopping_distance)  # Deceleration = v^2 / (2 * d)
    impact_force = mass * (deceleration + g)  # Impact force with gravity considered
    normal_force = mass * g + impact_force  # Normal force during impact

    # Display the results
    print("Rocket Landing Forces:")
    print(f"Time to touchdown: {time_list[-1]:.2f} [s]")
    print(f"Touchdown Velocity: {abs(touchdown_velocity):.3f} [m/s]")
    print(f"Deceleration during landing: {deceleration:.3f} [m/s^2]")
    print(f"Impact Force: {impact_force / 10 ** 6:.3f} [MN]")
    print(f"Normal Force at landing: {normal_force / 10 ** 6:.3f} [MN]")
    # print(f'Net Impulse Applied {np.trapz(thrust_list, dx=dt) }[N*s]')
    print(f'Net Power Applied {-1 * np.dot(thrust_list, velocity_list) / 10 ** 9 :.3f} [GW]')

# --- Plotting Section ---
# Plot the results to visualize the rocket's descent.

plt.figure(figsize=(10, 6))

# Altitude Plot
plt.subplot(3, 1, 1)
plt.plot(time_list, altitude_list, label="Altitude (m)")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Rocket Altitude Over Time")
plt.grid(True)

# Velocity Plot
plt.subplot(3, 1, 2)
plt.plot(time_list, velocity_list, label="Velocity (m/s)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Rocket Velocity Over Time")
plt.grid(True)

# Thrust Plot
plt.subplot(3, 1, 3)
plt.plot(time_list, thrust_list, label="Thrust (N)", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.title("Rocket Thrust Over Time")
plt.grid(True)

# plt.tight_layout()
plt.show()
