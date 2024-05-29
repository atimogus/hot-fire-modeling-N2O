from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from functions import *

time_step = 0.03 #if program fails to execute randomly, change this number, it is because of sequence for plotting a graph
temp_degrees = 20 #initial temperature of an oxidizer
oxidizer_mass_l_kg =19
oxidizer, fuel, fuel_CEA = 'N2O' , 'Acetone', 'C3H6O,acetone' #for fuel has two different names because CEApy has special name for acetone
o_f = 3
pc = 28e5 #pa, initial chamber pressure
P_atmosphere = 1e5
liquid , gas = 0 , 1
gamma = 1.3 #for N2O vapour
R = PropsSI('GAS_CONSTANT', 'NitrousOxide') / PropsSI('M', 'NitrousOxide')  # Specific gas constant in J/(kg*K) for N2O
temp = 273.15 + (temp_degrees) #pressure of a tank is dependant on a tank temperature
pressure_frictionLoss  = 2e5 #pressure loss due to piston friction in fuel tank needs to be determined experimentaly
C_d = 0.5 #this injector property has to be determined experimentaly usually between 0.5 and 0.77
d_mm = 7.5
A = (d_mm/1000)**2 * np.pi/4
density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, gas, liquid, oxidizer)
density_f = PropsSI('D', 'P', pressure_l, 'T', temp, fuel)

#tank modeling
V_liquid = oxidizer_mass_l_kg/density_l  # additional 10 percent for gas
V_gas =0.01*V_liquid
V_tank = V_liquid + V_gas
mass_v = V_gas * density_v
mass_l = oxidizer_mass_l_kg
#vidjeti nacin kako smanjiti kolicinu goriva
mass_f = oxidizer_mass_l_kg / (o_f * 2)
mass_f_initial = mass_f
V_fuel = mass_f / density_f

#injector modeling (pintle)
ml_dot = C_d * A * np.sqrt(2 * density_l * (pressure_l - pc))
mf_dot = ml_dot / o_f
A_fuel = mf_dot / (C_d * np.sqrt(2 * density_f * (pressure_l - pressure_frictionLoss - pc)))
D_fuel_mm = np.sqrt(A_fuel * 4/np.pi) * 1000

#combustion modeling
df = CEA_start(oxidizer, fuel_CEA, temp, pc, o_f)
At_m2, Dt = Dt_mm(df["gam"][0], df["t"][0],df["mw"][0], ml_dot + ml_dot/o_f, pc, df['R'][0])

liquid_pressure = []
liquid_massflow = []
vapor_pressure = []
vapor_mass_flow = []
mass_liquid = []
mass_vapour = []
O_F_shift = []
fuel_massflow = []
fuel_pressure = []
dp_dt = []
pressure_c = []
temperature_c = []
total_thrust = []
o_f_old = o_f
i=0

while mass_l > ml_dot * 1.5:
    mass_liquid , mass_vapour =  np.append(mass_liquid, mass_l) , np.append(mass_vapour, mass_v)
    pressure_old = pressure_l
    density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, gas, liquid, oxidizer)
        
    if  mass_l < oxidizer_mass_l_kg *0.5  and pressure_old - pressure_l < np.mean(dp_dt[-5:])*0.5:
        temp = PropsSI('T','P',pressure_old - np.mean(dp_dt[-5:]),'Q', gas, oxidizer)
        density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, gas, liquid, oxidizer)

    ml_dot, mass_l, mass_l_old = mdot_liquid_oxidizer_SPI(C_d, A, density_l, pressure_l, pc, time_step, mass_l)
    
    if mass_f > mf_dot * 1.5:
        pressure_f = pressure_l - pressure_frictionLoss
        density_f = PropsSI('D', 'P', pressure_l, 'T', temp , fuel)
        mf_dot = C_d * A_fuel * np.sqrt(2 * density_f * (pressure_f - pc)) * time_step
        mass_f -= mf_dot
        
        V_tank +=  mf_dot / density_f
        mass_l, mass_v, mass_vapourised = mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l)
        
        o_f = ml_dot / mf_dot
        if int(o_f * 100) % 10 != int(o_f_old * 100) % 10:
            i += 1
            print(i)
            df = CEA_start(oxidizer, fuel_CEA, temp, pc, o_f)
            pc = pc_func(df["gam"][0], df["t"][0], df["mw"][0], (ml_dot + mf_dot)/time_step, P_atmosphere, At_m2, df['R'][0])
        o_f_old = o_f  # Update O_F_old
        
        thrust = (ml_dot + mf_dot)/time_step * Ve_func(df["gam"][0], df["t"][0], df["mw"][0], P_atmosphere, pc)

        fuel_pressure = np.append(fuel_pressure, pressure_f)
        O_F_shift = np.append(O_F_shift, o_f)
        fuel_massflow = np.append(fuel_massflow, mf_dot)
        total_thrust.append(thrust)
        pressure_c.append(pc)
        
    else:
        mf_dot = 0
        fuel_massflow, fuel_pressure = np.append(fuel_massflow, mf_dot), np.append(fuel_pressure, mf_dot)
        mass_l, mass_v, mass_vapourised = mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l)
        
    Heat_of_vapourisation = PropsSI('H','T',temp,'Q',gas,oxidizer) - PropsSI('H','T',temp,'Q',liquid,oxidizer)
    heat_removed_deltaQ = mass_vapourised * Heat_of_vapourisation 
    deltaT = -heat_removed_deltaQ/(mass_l * PropsSI('C','T',temp,'Q', liquid ,oxidizer))
    temp += deltaT #promjena temperature tekucine usljed hladjenja tekucine zbog isparavanja   
    dp_dt = np.append(dp_dt, pressure_old - pressure_l)
    liquid_pressure = np.append(liquid_pressure, pressure_v)
    liquid_massflow.append(ml_dot)    
    temperature_c.append(pc)
    
# Initial settings for vapor loop
mass_v_initial = mass_v
# Modeling for vapor pressure as ideal gas
while pressure_v > P_atmosphere * 1 and mass_v > mass_v_initial * 0.05: 
    fuel_massflow, fuel_pressure = np.append(fuel_massflow, mf_dot), np.append(fuel_pressure, mf_dot)
    mass_liquid , mass_vapour =  np.append(mass_liquid, mass_l) , np.append(mass_vapour, mass_v)
    O_F_shift = np.append(O_F_shift, mf_dot)

    mv_dot = C_d * A * np.sqrt(2 * density_v * (pressure_v - P_atmosphere)) * time_step
    mass_old = mass_v
    mass_v -= mv_dot

    # Update temperature and pressure using polytropic relations
    T_2 = temp * (mass_v / mass_old) ** (gamma - 1)
    P_2 = pressure_v * (T_2 / temp) ** (gamma / (gamma - 1))
    temp = T_2
    pressure_v = P_2

    vapor_pressure.append(P_2)
    vapor_mass_flow.append(mv_dot)

tank_pressure = np.concatenate((liquid_pressure , vapor_pressure))
tank_massflow = np.concatenate((liquid_massflow, vapor_mass_flow)) / time_step
fuel_massflow = fuel_massflow / time_step
sequence = np.arange(0, len(tank_massflow)*time_step,time_step)  
pressure_c = np.array(pressure_c)
sequence2 = np.arange(0, len(pressure_c)*time_step,time_step)

import os
script_dir = os.path.dirname(os.path.realpath(__file__))
print(f"Script directory: {script_dir}")
os.chdir(script_dir)
print(f"Current working directory after change: {os.getcwd()}")
csv_file_path = os.path.join(script_dir, 'izgaranje.csv')

try:
    # Write to CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'Tank Pressure (Pa)', 'Mass Flow Rate (kg/s)'])
        for i in range(len(sequence)):
            writer.writerow([sequence[i], tank_pressure[i], tank_massflow[i]])
    print(f"CSV file created successfully at: {csv_file_path}")
except Exception as e:
    print(f"Error creating CSV file: {e}")
    
# Plotting results
fig = 10
plt.figure(figsize=(fig * 2 , fig))

# Plot for vapor tank pressure
plt.subplot(2, 2, 1)
plt.plot(sequence, tank_pressure/1e5, color='red')
# plt.plot(dataset['timeShift'], dataset['pressureShift']-3, label='Dataset Pressure', color='blue')
plt.plot(sequence2, pressure_c / 1e5, color='blue')
plt.plot(sequence, fuel_pressure/1e5, color='green')
plt.xlabel('Time (s)')
plt.ylabel('tank pressure (Pa)')
plt.legend(['ox cold flow model', 'chamber pressure', 'fuel cold flow model'])
plt.grid(True)

# Plot for vapor mass flow
plt.subplot(2, 2, 2)
plt.plot(sequence, tank_massflow, color='red')
plt.plot(sequence, fuel_massflow, color='green')
plt.xlabel('Time (s)')
plt.ylabel('massflow (kg/s)')
plt.legend(['N2O mass flow rate', 'Fuel mass flow rate'])
plt.grid(True)

# Plot for vapor tank pressure
plt.subplot(2, 2, 3)
plt.plot(sequence, mass_liquid, color='red')
plt.plot(sequence, mass_vapour, color='green')
plt.xlabel('Time (s)')
plt.ylabel('mass in tank (kg)')
plt.legend(['mass_liquid', 'mass_vapour'])
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(sequence2, total_thrust, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Total thrust (N)')

plt.tight_layout()
plt.show()

print(f"mean ox mass flow = {np.mean(liquid_massflow) / time_step:.2f}, mean fuel mass flow = {np.mean(fuel_massflow[fuel_massflow != 0]):.2f}, average thrust = {np.mean(total_thrust)}")
