import numpy as np
import matplotlib.pyplot as plt

# --- Constants Section ---
# These constants will remain fixed throughout the simulation.

g =   # Acceleration due to gravity in m/s^2. What would happen if you were on a different planet?
mass =   # Mass of the rocket in kilograms. How would a different mass affect landing?
Cd =   # Drag coefficient (dimensionless). How does changing shape alter this?
rho =   # Air density at sea level in kg/m^3. What if the air was thinner?
A =   # Cross-sectional area of the rocket in square meters. What if the rocket was larger?
max_thrust =   # Maximum thrust force available, in Newtons. Is this enough for a safe landing?

# --- Simulation Parameters ---
# These control the resolution and duration of the simulation.

dt =   # Time step in seconds. What happens if you make it smaller or larger?
total_time =   # Total time to simulate in seconds. Does the rocket need the full time?

# --- Initial Rocket Conditions ---
# Set the initial state of the rocket before simulation starts.

altitude = 1000  # Starting altitude in meters above ground.
velocity = -50  # Initial velocity in m/s (negative indicates falling down).
thrust = 0  # Initial thrust applied by engines in Newtons.

# --- Data Storage ---
# Lists to keep track of the simulation data over time for plotting.

time_list = []        # Store the time at each simulation step.
altitude_list = []    # Track the rocket's altitude over time.
velocity_list = []    # Track the rocket's velocity over time.
thrust_list = []      # Track how thrust changes over time.

# --- Function Definitions ---
# Define any helper functions needed for the simulation.

def calculate_drag(velocity):
    """
    Calculate the drag force based on the current velocity.
    
    Arguments:
        velocity (float): The current velocity of the rocket (m/s).

    Returns:
        float: The drag force exerted by the atmosphere.
    """
    # Why does the drag depend on the velocity squared? Think about how resistance changes with speed.
    return #drag force equation

# --- Main Simulation Loop ---
# This is where the physics calculations happen, step-by-step.

# We will loop over each time step and update the rocket's state.
# What is being updated? Think about position, velocity, and applied forces.

for t in np.arange(0, total_time, dt):
    
    # --- Force Calculations ---
    # Compute the forces acting on the rocket at the current moment.
    
    # Gravitational force (downward)
    weight =   # What is the rocket's weight? What direction does this force act?

    # Drag force (depends on the current velocity)
    drag = calculate_drag(velocity)  # How does drag change with increasing or decreasing velocity?
    
    # --- Thrust Control Logic ---
    # Decide how much thrust the rocket engines should apply.
    
    # If the altitude is above 50 meters, maintain high thrust to slow down.
    if altitude > 50:
        # Why do we add drag and part of the velocity here? What is the purpose?
        thrust = min(max_thrust, weight + abs(drag) + (0.5 * mass * abs(velocity)))
    else:
        # Reduce thrust as the rocket nears the ground to ensure a gentle landing.
        thrust = min(max_thrust, weight + abs(drag) + (0.1 * mass * abs(velocity)))
    
    # --- Net Force Calculation ---
    # Compute the total force on the rocket (net force).
    
    net_force =   # What happens if this force becomes negative?

    # --- Update Rocket State ---
    # Use the forces calculated to update the rocket's acceleration, velocity, and position.
    
    # Calculate the rocket's acceleration using Newton's second law.
    acceleration =   # How does the net force affect acceleration?
    
    # Update the velocity based on the acceleration.
    velocity +=   # Why do we multiply by the time step?

    # Update the altitude based on the current velocity.
    altitude +=   # What would happen if dt was too large?

    # --- Data Recording ---
    # Store the updated values for later analysis.

    time_list.append(t)              # Save the current time step.
    altitude_list.append(altitude)   # Save the current altitude.
    velocity_list.append(velocity)   # Save the current velocity.
    thrust_list.append(thrust)       # Save the current thrust.

    # --- Check Landing Condition ---
    # Determine if the rocket has reached the ground.
    
    if altitude <= 0:
        # The rocket has landed! What was the velocity just before landing?
        touchdown_velocity = 
        break  # Exit the loop since the simulation is complete.

# --- Post-Landing Calculations ---
# Now that the rocket has landed, calculate the forces experienced on impact.

# Calculate the required deceleration to bring the rocket to a full stop.
# How does the deceleration depend on the stopping distance?

if altitude <= 0:
    # Why is the deceleration formula based on velocity and distance?
    deceleration = (touchdown_velocity**2) / (2 * stopping_distance)
    
    # Calculate the impact force using Newton's second law.
    impact_force =   # Why do we add gravity to the deceleration?
    
    # Calculate the normal force exerted by the ground.
    normal_force =   # Why is the weight added to the impact force?
else:
    # If the simulation didn't reach the ground, set default values.
    deceleration = 0
    impact_force = 0
    normal_force = mass * g

# Display the calculated forces after landing.
print("Rocket Landing Forces:")
print(f"Touchdown Velocity: {abs(touchdown_velocity):.2f} m/s")
print(f"Deceleration during landing: {deceleration:.2f} m/s²")
print(f"Impact Force: {impact_force:.2f} N")
print(f"Normal Force at landing: {normal_force:.2f} N")

# --- Plotting Section ---
# Visualize the results to analyze the rocket's descent.



# Altitude Plot


# Velocity Plot


# Thrust Plot

