import numpy as np
import matplotlib.pyplot as plt

# --- Constants Section ---
g = 9.81  # Acceleration due to gravity in m/s^2
mass = 500  # Mass of the rocket in kilograms
Cd = 0.75  # Drag coefficient (dimensionless)
rho = 1.225  # Air density at sea level in kg/m^3
A = 1.0  # Cross-sectional area of the rocket in square meters
max_thrust = 15000  # Maximum thrust force available, in Newtons
touchdown_velocity = -1

# --- Simulation Parameters ---
dt = 0.1  # Time step in seconds
total_time = 10  # Total time to simulate in seconds

# --- Initial Rocket Conditions ---
altitude = 1000  # Starting altitude in meters above ground
velocity = -50  # Initial velocity in m/s (negative indicates falling down)
thrust = 0  # Initial thrust applied by engines in Newtons

# --- Data Storage ---
time_list = []
altitude_list = []
velocity_list = []
thrust_list = []

def thrust_vector(thrust, theta, phi):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    return np.array([
        thrust * np.sin(theta_rad) * np.cos(phi_rad),  # x-component
        thrust * np.sin(theta_rad) * np.sin(phi_rad),  # y-component
        thrust * np.cos(theta_rad)                    # z-component
    ])

# --- Function Definitions ---
def calculate_drag(velocity):
    """
    Calculate the drag force based on the current velocity.
    
    Arguments:
        velocity (float): The current velocity of the rocket (m/s).
    
    Returns:
        float: The drag force exerted by the atmosphere.
    """
    return 0.5 * Cd * rho * A * velocity**2 * np.sign(velocity)

# --- Main Simulation Loop ---
for t in np.arange(0, total_time, dt):
    # --- Force Calculations ---
    weight = mass * g  # Gravitational force
    drag = calculate_drag(velocity)  # Drag force

    # --- Thrust Control Logic ---
    if altitude > 50:
        thrust = min(max_thrust, weight + abs(drag) + (0.5 * mass * abs(velocity)))
    else:
        thrust = min(max_thrust, weight + abs(drag) + (0.1 * mass * abs(velocity)))

    # --- Net Force Calculation ---
    net_force = thrust - weight - drag

    # --- Update Rocket State ---
    acceleration = net_force / mass
    velocity += acceleration * dt
    altitude += velocity * dt

    # --- Data Recording ---
    time_list.append(t)
    altitude_list.append(max(altitude, 0))  # Ensure altitude doesn't go below zero
    velocity_list.append(velocity)
    thrust_list.append(thrust)

    # --- Check Landing Condition ---
    if altitude <= 0:
        touchdown_velocity = velocity
        break

# --- Post-Landing Calculations ---
stopping_distance = 1  # Assume a stopping distance of 1 meter for simplicity
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
plt.subplot(3, 1, 1)
plt.plot(time_list, altitude_list, label='Altitude (m)')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.ylim(0,1000)
plt.title('Rocket Altitude vs. Time')
plt.grid()
plt.legend()

# Velocity Plot
plt.subplot(3, 1, 2)
plt.plot(time_list, velocity_list, label='Velocity (m/s)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Rocket Velocity vs. Time')
plt.grid()
plt.legend()

# Thrust Plot
plt.subplot(3, 1, 3)
plt.plot(time_list, thrust_list, label='Thrust (N)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.title('Rocket Thrust vs. Time')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
