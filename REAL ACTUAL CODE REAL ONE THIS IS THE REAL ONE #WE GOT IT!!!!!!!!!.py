import datetime
from meteostat import Point, Hourly
import yaml
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from scipy.interpolate import interp1d

 # Step 1: Data Collection with Meteostat
# Libraries
from datetime import datetime
from meteostat import Point, Hourly

# Get the weather data from meteostat for Edinburgh
location       = Point(55.9533, -3.1883)
start          = datetime(2024, 10, 7, 0)
end            = datetime(2024, 10, 8, 0)
meteostat_data = Hourly(location, start, end)
weather_data   = meteostat_data.fetch()


# Step 2: Data Fitting Using Manufacturer COP Data
# Libraries
import yaml
import numpy as np
from scipy.optimize import curve_fit

# Read the YAML file to access COP data
with open(r"C:\Users\thodi\Desktop\CMM3\heat_pump_cop_synthetic_full.yaml", 'r') as file:
    cop_data = yaml.safe_load(file)

# Extract outdoor temperature and COP values from 'heat_pump_cop_data' YAML file
cop_temps = []
cop_values   = []

for entry in cop_data['heat_pump_cop_data']:
    cop_temps.append(entry['outdoor_temp_C'])
    cop_values.append(entry['COP_noisy'])

# Convert to numpy arrays for later calculations
cop_temps = np.array(cop_temps)
cop_values = np.array(cop_values)

delta_T = 60 - cop_temps

# Define the COP model: COP = a + (b / delta_T)
def cop_model(delta_T, a, b):
    return a + (b / delta_T)

# Fit the curve to find 'a' and 'b'
popt, pcov = curve_fit(cop_model, delta_T, cop_values)

a, b = popt
print(f'Fitted values: a = {a}, b = {b}')


# Step 3: Heat Load Calculation
# Libraries
import yaml

# Read the inputs YAML file to access data
with open(r"C:\Users\thodi\Desktop\CMM3\inputs.yaml", 'r') as file:
    input_data = yaml.safe_load(file)

# Extract building properties
Awall  = input_data['building_properties']['wall_area']['value']
Uwall  = input_data['building_properties']['wall_U_value']['value']
Aroof  = input_data['building_properties']['roof_area']['value']
Uroof  = input_data['building_properties']['roof_U_value']['value']
Tsetpoint = input_data['building_properties']['indoor_setpoint_temperature_K']['value']

# Extract other necessary values
sim_time       = input_data['simulation_parameters']['total_time_seconds']['value']
time_points    = input_data['simulation_parameters']['time_points']['value']

interval = int(sim_time / time_points)

# Using meteostat data, get the ambient temperatures at each hour
T_amb_C = weather_data['temp'].tolist()
T_amb = [t + 273.15 for t in T_amb_C]

# Q_load calculation
def Q_load_calc_current(Awall, Uwall, Aroof, Uroof, T_amb_current, Tsetpoint):
    Q_load = -((Awall * Uwall * (T_amb_current - Tsetpoint)) + (Aroof * Uroof * (T_amb_current - Tsetpoint)))
    return Q_load


# Step 4: Heat Pump
# Libraries

# Extract heat pump properties
Ucond = input_data['heat_pump']['overall_heat_transfer_coefficient']['value']
Acond = input_data['heat_pump']['heat_transfer_area']['value']
Tcond = input_data['heat_pump']['fixed_condenser_temperature_K']['value']
Tmin  = input_data['heat_pump']['on_temperature_threshold_K']['value']
Tmax  = input_data['heat_pump']['off_temperature_threshold_K']['value']
Ttankinitial = input_data['initial_conditions']['initial_tank_temperature_K']['value']

def Q_hp_calc_current(Ucond, Acond, Tcond, T_tank, tolerance): 
    heat_pump = False
    if T_tank >= Tmax - tolerance:
        heat_pump = False
    elif T_tank <= Tmin:
        heat_pump = True
        
    if heat_pump == True:
        Q_hp = Ucond * Acond * (Tcond - T_tank)
    elif heat_pump == False:
        Q_hp = 0 
    return Q_hp


# Step 5: ODE Setup for Tank Dynamics
# Libraries
import numpy as np

Utank = input_data['hot_water_tank']['heat_loss_coefficient']['value']
Ct = input_data['hot_water_tank']['total_thermal_capacity']['value']
water_mass = input_data['hot_water_tank']['mass_of_water']['value']
water_density = 1000   # Density of water

vol_w = 1 * (water_mass / water_density)   # m^3
# Assuming a cylinder, and that its height is 3x the radius. Currently tank is 100% full, no air gaps (change the '1' to a different multiplier for air gaps)
r_tank = (np.cbrt(vol_w / np.pi)) / 3
h_tank = 3 * r_tank
A_tank = (2 * np.pi * r_tank**2) + (2 * np.pi * r_tank * h_tank)   # Surface area of the tank

def Q_loss_calc_current(Utank, A_tank, T_tank, T_amb_current):
    Q_loss = Utank * A_tank * (T_tank - T_amb_current)
    return Q_loss

def dT_tank_calc_current(Q_hp_current, Q_load_current, Q_loss_current, Ct):
    dT_tank = (Q_hp_current - Q_load_current - Q_loss_current) / Ct
    return dT_tank


# Step 6: Solving the ODE System
# Libraries
from scipy.integrate import solve_ivp

tolerance = 1   # [°C]
def system_calc_current(time, temp):
    x, hour    = 0, 0
    T_tank     = temp
    for i in range(0, 86000, 86):
        x = x + 86.4
        if round(x / 3600) > hour:
            hour = hour + 1
        T_amb_current = T_amb[hour]
        q_hp    = Q_hp_calc_current(Ucond, Acond, Tcond, T_tank, tolerance)
        q_load  = Q_load_calc_current(Awall, Uwall, Aroof, Uroof, T_amb_current, Tsetpoint)
        q_loss  = Q_loss_calc_current(Utank, A_tank, T_tank, T_amb_current)
        dt_tank = dT_tank_calc_current(q_hp, q_load, q_loss, Ct)
        T_tank  = T_tank + dt_tank
        return dt_tank

times = np.linspace(0, sim_time, 1000)
solution = solve_ivp(system_calc_current, [0, sim_time], [Ttankinitial], t_eval = times, method = 'RK45', dense_output = False)
temps = solution.y[0]
temps_C = [t - 273.15 for t in temps]


# Plotting
# Libraries
import matplotlib.pyplot as plt

# Arrays to store heat pump power values for plotting
Q_hp_values = []

# Iterate over the time steps to calculate Q_hp
for t in range(len(times)):
    # Current tank temperature
    T_tank = temps[t]
    
    # Calculate Q_hp for the current time step
    Q_hp = Q_hp_calc_current(Ucond, Acond, Tcond, T_tank, tolerance)
    Q_hp_values.append(Q_hp)

# Plotting Tank Temperature
plt.figure(figsize=(15, 10))
plt.plot(times / 3600, temps_C, label='Tank Temperature (°C)', color='tab:blue')
plt.axhline(y=40, color='green', linestyle='--', label='Heat Pump On Threshold (40°C)')
plt.axhline(y=60, color='orange', linestyle='--', label='Heat Pump Off Threshold (60°C)')
plt.xlabel('Time (hours)')
plt.ylabel('Tank Temperature (°C)')
plt.title('Heat Pump and Hot Water Tank Dynamics')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Heat Pump Power (Q_hp)
plt.figure(figsize=(15, 10))
plt.plot(times / 3600, Q_hp_values, label='Heat Pump Power (Q_hp)', color='tab:red')
plt.xlabel('Time (hours)')
plt.ylabel('Heat Pump Power (W)')
plt.title('Heat Pump Power (Q_hp) Over Time')
plt.legend()
plt.grid(True)
plt.show()