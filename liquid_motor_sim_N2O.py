import numpy as np
import matplotlib.pyplot as plt
import logging
from CoolProp.CoolProp import PropsSI
import cantera as ct
import rocketpy as rp
import pandas as pd

# Constants
TIME_STEP = 0.01
INITIAL_TEMP_DEGREES = 22
OXIDIZER_MASS_L_KG = 21
OXIDIZER, FUEL, FUEL_CEA = 'N2O', 'Acetone', 'C3H8'
YAML_FILE = 'gri30.yaml'
O_F = 2.1
PC = 28e5 # pascals
P_ATMOSPHERE = 101325 # pascals
P_EXIT = 95555
PRESSURE_FRICTION_LOSS = 2e5 # pascals
FUEL_DURATION = 1.25 # value between 1 and 2
C_D = 0.66
D_MM = 3.5
N_ORFICE_OX = 8
N_ORFICE_F = 8
DRY_WEIGHT = 12 #kilograms
LAUNCH_RAIL_ANGLE = 84 #degrees
DRAG_COEFF = 0.75 # for mach 1.2 Cd is = https://www.nakka-rocketry.net/RD_body.html
LOSS = 7 # percent
L_D_ratio = 15 # tank lenght / body diameter ratio

#fuel and oxidizer pressure and according to this change it calculates for thrust
LIQUID , GAS = 0 , 1
PINTLE_ANGLE = 60
GAMMA = 1.3
G = 9.81

# Set up logging
logging.basicConfig(level=logging.INFO)

data = {
    'liquid_pressure': [],
    'liquid_massflow': [],
    'vapor_pressure': [],
    'vapor_mass_flow': [],
    'mass_liquid': [],
    'mass_vapour': [],
    'mass_fuel': [],
    'O_F_shift': [],
    'fuel_massflow': [],
    'fuel_pressure': [],
    'dp_dt': [],
    'pressure_c': [],
    'temperature_c': [],
    'total_thrust': [],
    'density_ex_gas': [],
    'temperature_tank': [],
    'rockett_acceleration': [],
    'rocket_velocity': [],
    'rocket_distance': [],
    'TW_ratio': []
}

# Functions
def calculate_specific_gas_constant():
    return PropsSI('GAS_CONSTANT', 'NitrousOxide') / PropsSI('M', 'NitrousOxide')

def calculate_initial_properties(temp_degrees, oxidizer, gas):
    temp = 273.15 + temp_degrees
    density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, GAS, LIQUID, oxidizer)
    return temp, density_v, pressure_v, density_l, pressure_l

def calculate_tank_properties(density_l, density_v):
    V_liquid = OXIDIZER_MASS_L_KG / density_l
    V_gas = 0.05 * V_liquid
    V_tank = V_liquid + V_gas
    mass_v = V_gas * density_v
    mass_l = OXIDIZER_MASS_L_KG
    return V_liquid, V_gas, V_tank, mass_v, mass_l

def calculate_fuel_properties(pressure_l, temp):
    density_f = PropsSI('D', 'P', pressure_l, 'T', temp, FUEL)
    mass_f = OXIDIZER_MASS_L_KG / (O_F * FUEL_DURATION)
    V_fuel = mass_f / density_f
    return mass_f, V_fuel

def calculate_injector_properties(C_d, A_OX_injector, density_l, pressure_l, pc, O_F, temp):
    ml_dot = N_ORFICE_OX * C_d * A_OX_injector * np.sqrt(2 * density_l * (pressure_l - pc))
    mf_dot = ml_dot / O_F
    density_f = PropsSI('D', 'P', pressure_l, 'T', temp, FUEL)
    A_fuel = mf_dot / (N_ORFICE_F * C_d * np.sqrt(2 * density_f * (pressure_l - PRESSURE_FRICTION_LOSS - pc)))
    D_fuel_mm = np.sqrt(A_fuel * 4 / np.pi) * 1000
    speed_OxInjector = ml_dot / (density_l * A_OX_injector)
    speed_FInjector = mf_dot / (density_f * A_fuel)
    #calculate momentum equation for impinging injectors
    #splashhead injector?
    return ml_dot, mf_dot, A_fuel, D_fuel_mm, speed_OxInjector, speed_FInjector

def run_liquid_phase_simulation(pressure_l, A_ox_injector):
    global pressure_v, temp, pressure_old, temperature_tank, pressure_combustion, df, mass_l, ml_dot, mass_v, density_v, density_l, pc, o_f, fuel_pressure, O_F_shift, fuel_massflow, total_thrust, pressure_c, density_ex_gas, o_f_old, dp_dt, liquid_pressure, liquid_massflow, temperature_c, V_tank, mass_f, mf_dot, A_fuel
    
    density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, GAS, LIQUID, OXIDIZER)
    
    # This if statement prevents oscillations in calculations (fine-tune if you have oscillations)
    if mass_l < OXIDIZER_MASS_L_KG * 0.5 and pressure_old - pressure_l < np.mean(data['dp_dt'][-5:]) * 0.5:
        if mass_f > mf_dot:
            temp = PropsSI('T', 'P', pressure_old - np.mean(data['dp_dt'][-5:]), 'Q', GAS, OXIDIZER)
            density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, GAS, LIQUID, OXIDIZER)
        else:
            temp = PropsSI('T', 'P', pressure_old - np.mean(data['dp_dt'][-5:]), 'Q', GAS, OXIDIZER)
            density_v, pressure_v, density_l, pressure_l = get_n2o_properties_T(temp, GAS, LIQUID, OXIDIZER)
    
    ml_dot, mass_l, mass_l_old = mdot_liquid_oxidizer_SPI(A_ox_injector, density_l, pressure_l, pressure_combustion, mass_l)
    
    if mass_f > mf_dot:
        pressure_f = pressure_l - PRESSURE_FRICTION_LOSS
        density_f = PropsSI('D', 'P', pressure_l, 'T', temp, FUEL)
        mf_dot = N_ORFICE_F * C_D * A_fuel * np.sqrt(2 * density_f * (pressure_f - pressure_combustion)) * TIME_STEP
        mass_f -= mf_dot

        V_tank += mf_dot / density_f
        mass_l, mass_v, mass_vapourised = mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l)

        o_f = ml_dot / mf_dot
        if np.isnan(o_f):
            print("your chamber pressure is too high or your initial temperature is too low \n \n")

        if int(o_f * 100) % 10 != int(o_f_old * 100) % 10:
            df = CEA_cantera(YAML_FILE, o_f, FUEL_CEA, OXIDIZER, temp, PC)
            pressure_combustion = pc_func(df["gam"], df["t"], (ml_dot + mf_dot) / TIME_STEP, At_m2, df['R'])
        
        o_f_old = o_f
        
        thrust = (ml_dot + mf_dot) / TIME_STEP * Ve_func(df["gam"], df["t"], df["mw"], P_EXIT, pressure_combustion)
        
        data['fuel_pressure'].append(pressure_f)
        data['O_F_shift'].append(o_f)
        data['fuel_massflow'].append(mf_dot)
        data['pressure_c'].append(pressure_combustion)
        data['density_ex_gas'].append(df["rho"])

    else:
        mf_dot, mass_f, thrust = 0, 0, 0
        data['fuel_massflow'].append(mf_dot)
        data['fuel_pressure'].append(mf_dot)
        mass_l, mass_v, mass_vapourised = mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l)
        
    Heat_of_vapourisation = PropsSI('H', 'T', temp, 'Q', GAS, OXIDIZER) - PropsSI('H', 'T', temp, 'Q', LIQUID, OXIDIZER)
    heat_removed_deltaQ = mass_vapourised * Heat_of_vapourisation
    deltaT = -heat_removed_deltaQ / (mass_l * PropsSI('C', 'T', temp, 'Q', LIQUID, OXIDIZER))
    temp += deltaT  # Change in temperature due to evaporation cooling
    
    data['dp_dt'].append(pressure_old - pressure_l)
    data['liquid_pressure'].append(pressure_v)
    data['liquid_massflow'].append(ml_dot)
    data['mass_fuel'].append(mass_f)
    data['mass_liquid'].append(mass_l)
    data['mass_vapour'].append(mass_v)
    data['temperature_tank'].append(temp)
    data['temperature_c'].append(df["t"])
    pressure_old = pressure_l
    
    return pressure_old, thrust


def run_vapor_phase_simulation():
    global A_ox_injector, density_v, mass_v, temp, pressure_v, mf_dot, fuel_massflow, fuel_pressure, mass_liquid, mass_vapour, O_F_shift, vapor_pressure, vapor_mass_flow
    
    data['fuel_massflow'].append(mf_dot)
    data['fuel_pressure'].append(mf_dot)
    data['mass_liquid'].append(mass_l)
    data['mass_vapour'].append(mass_v)
    data['O_F_shift'].append(mf_dot)
    data['mass_fuel'].append(mass_f)

    mv_dot = N_ORFICE_OX * C_D * A_ox_injector * np.sqrt(2 * density_v * (pressure_v - P_ATMOSPHERE)) * TIME_STEP
    mass_old = mass_v
    mass_v -= mv_dot

    # Update temperature and pressure using polytropic relations
    T_2 = temp * (mass_v / mass_old) ** (GAMMA - 1)
    P_2 = pressure_v * (T_2 / temp) ** (GAMMA / (GAMMA - 1))
    temp = T_2
    pressure_v = P_2
    
    data['vapor_pressure'].append(P_2)
    data['vapor_mass_flow'].append(mv_dot)
    # total_thrust.append(thrust)
    
    return 0 # we return value zero, this is intended to be a thrust gained on capour phase

def plot_results():
    global body_diameter, tank_height, fuel_tank_diameter, mass_l, ml_dot, mass_v, temp, pressure_l, density_v, density_l, pc, o_f, fuel_pressure, O_F_shift, fuel_massflow, total_thrust, pressure_c, density_ex_gas, o_f_old, dp_dt, liquid_pressure, liquid_massflow, temperature_c, V_tank

    # Convert lists to numpy arrays for consistency
    tank_pressure = np.concatenate((data["liquid_pressure"], data["vapor_pressure"]))
    tank_massflow_array = np.concatenate((data["liquid_massflow"], data["vapor_mass_flow"])) / TIME_STEP
    fuel_massflow_array = np.array(data["fuel_massflow"]) / TIME_STEP
    total_thrust_array = np.array(data["total_thrust"])
    pressure_c_array = np.array(data['pressure_c'])

    # Time sequences for plotting
    sequence = np.arange(0, len(tank_massflow_array) * TIME_STEP, TIME_STEP)
    sequence2 = np.arange(0, len(pressure_c_array) * TIME_STEP, TIME_STEP)

    # Plotting results
    fig = 10
    plt.figure(figsize=(fig * 2, fig))

    # Plot for vapor tank pressure
    plt.subplot(2, 2, 1)
    plt.plot(sequence, tank_pressure / 1e5, color='red')
    plt.plot(sequence2, pressure_c_array / 1e5, color='blue')
    plt.plot(sequence, np.array(data['fuel_pressure']) / 1e5, color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Tank Pressure (bar)')
    plt.legend(['Oxidizer Pressure', 'Chamber Pressure', 'Fuel Pressure'])
    plt.grid(True)

    # Plot for mass flow rates
    plt.subplot(2, 2, 2)
    plt.plot(sequence, tank_massflow_array, color='red')
    plt.plot(sequence, fuel_massflow_array, color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Flow Rate (kg/s)')
    plt.legend(['N2O Mass Flow Rate', 'Fuel Mass Flow Rate'])
    plt.grid(True)

    # Plot for mass in tank
    plt.subplot(2, 2, 3)
    plt.plot(sequence, data["mass_liquid"], color='red')
    plt.plot(sequence, data["mass_vapour"], color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass in Tank (kg)')
    plt.legend(['Mass Liquid', 'Mass Vapour'])
    plt.grid(True)

    # Plot for total thrust
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(0, len(total_thrust_array) * TIME_STEP, TIME_STEP), data["total_thrust"], color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Thrust (N)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"Mean oxidizer mass flow = {np.mean(data['liquid_massflow']) / TIME_STEP:.2f} kg/s, Mean fuel mass flow = {np.mean(fuel_massflow_array[fuel_massflow_array != 0]):.2f} kg/s")
    print(f" Average thrust = {np.mean(total_thrust_array[total_thrust_array != 0]):.2f} N, Impulse - {np.mean(total_thrust_array[total_thrust_array != 0]) * len(total_thrust_array[total_thrust_array != 0]) * TIME_STEP:.2f} Ns")
    print(f"L* = {l_star:.2f}, Volume of combustion chamber = {volume_chamb * 1000:.4f} liters")
    print(f"Mix time = {mix_time:.4f} s, Burn time = {burn_time:.4f} s")
    print(f"body_diameter  = {body_diameter*1000:.2f} mm, tank_height  = {tank_height*1000:.2f} mm, fuel_tank_diameter  = {fuel_tank_diameter*1000:.2f} mm \n \n")

    # Length of the arrays
    sequence3 = np.arange(0, len(data['rockett_acceleration']) * TIME_STEP, TIME_STEP)
    plt.figure(figsize=(10, 6))

    # Plot acceleration and velocity on primary y-axis
    fig, ax1 = plt.subplots()

    ax1.plot(sequence3, data['rockett_acceleration'], label='Rocket Acceleration', color='b')
    ax1.plot(sequence3, data['rocket_velocity'], label='Rocket Velocity', color='g')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Acceleration / Velocity')
    ax1.legend(loc='upper left')
    ax1.set_title('Rocket Acceleration, Velocity, and Distance over Time')

    # Create secondary y-axis for distance
    ax2 = ax1.twinx()
    ax2.plot(sequence3, data['rocket_distance'], label='Rocket Distance', color='r')
    ax2.set_ylabel('Distance')
    ax2.legend(loc='upper right')

    # Show plot
    plt.show()

    print(f'apogee = {data["rocket_distance"][-1]:.0f} m = {data["rocket_distance"][-1]/1000:.1f}km,  max speed = {np.max(data["rocket_velocity"]):.2f} m/s, mean TW ratio = {np.mean(data["TW_ratio"][data["TW_ratio"] != 0]):.2f}')
    print(f'apogee with = {LOSS}% loss is {data["rocket_distance"][-1] * (1-LOSS/100):.0f} m ')

    
def generate_csv_files():

    # Save the DataFrame to a CSV file
    pd.DataFrame(data['mass_liquid']).to_csv('data/mass_liquid.csv', index=False)
    pd.DataFrame(data['mass_vapour']).to_csv('data/mass_vapour.csv', index=False)
    pd.DataFrame(data['mass_fuel']).to_csv('data/mass_fuel.csv', index=False)
    
    mass_liquid_array = np.array(data['mass_liquid'])
    mass_vapour_array = np.array(data['mass_vapour'])
    mass_fuel_array = np.array(data['mass_fuel'])
    
    total_mass = mass_liquid_array + mass_vapour_array + mass_fuel_array
    
    pd.DataFrame(total_mass).to_csv('data/total_mass.csv', index=False)
    pd.DataFrame(data['total_thrust']).to_csv('data/total_thrust.csv', index=False)


    
def mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l):
    mass_l = (V_tank-(mass_l_old+mass_v)/density_v)/(1/density_l-1/density_v)
    mass_vapourised = mass_l_old - mass_l
    mass_v += mass_vapourised
    return mass_l, mass_v, mass_vapourised

def get_n2o_properties_T(temp, gas, liquid, fluid):
    density_v, pressure_v = PropsSI('D','T',temp,'Q',gas,fluid), PropsSI('P','T',temp,'Q',gas,fluid) # promjena gustoce plina
    density_l, pressure_l = PropsSI('D','T',temp,'Q',liquid,fluid), PropsSI('P','T',temp,'Q',liquid,fluid)# promjena gustoce tekucine, pressure vapour = pressure liquid
    return density_v, pressure_v, density_l, pressure_l

def mdot_liquid_oxidizer_SPI(A_ox_injector, density_l, pressure_l, pressure_combustion, mass_l):
    ml_dot = N_ORFICE_OX * C_D * A_ox_injector * np.sqrt(2 * density_l * (pressure_l - pressure_combustion)) * TIME_STEP
    mass_l -= ml_dot 
    mass_l_old = mass_l
    return ml_dot, mass_l, mass_l_old

def CEA_cantera(yaml_file, o_f, fuel, oxidizer, init_temp, pressure_chamber):
    
    # Create gas object
    gas = ct.Solution(yaml_file)
    
    gas.TPX = init_temp, pressure_chamber, {fuel:1} # odabire svojstva mjesavine na 300K i 1atmosferi
    # density_fuel = gas.density
    gas.TPX = init_temp, pressure_chamber, {oxidizer:1}
    # density_ox = gas.density
    
    # Calculate mass fractions
    m_fuel = 1 / (1 + o_f)
    m_ox = o_f / (1 + o_f)
    gas.TPY = init_temp, pressure_chamber, {fuel: m_fuel, oxidizer: m_ox}
    
    # Perform equilibrium calculation at constant pressure and enthalpy
    gas.equilibrate('HP')
        
    results = {
        't': gas.T,
        'rho': gas.density,
        'h': gas.enthalpy_mass,
        'g': gas.gibbs_mass,
        's': gas.entropy_mass,
        'mw': gas.mean_molecular_weight,
        'cp': gas.cp_mass,
        'gam': gas.cp_mass / gas.cv_mass,
        'R': ct.gas_constant / gas.mean_molecular_weight,
        'mu' : gas.viscosity
    }
    
    return results 

def gas_fuelNox_density(yaml_file, fuel, oxidizer, init_temp, pressure_chamber):
    # Create gas object
    gas = ct.Solution(yaml_file)
    
    gas.TPX = init_temp, pressure_chamber, {fuel:1} # odabire svojstva mjesavine na 300K i 1atmosferi
    density_fuel = gas.density
    gas.TPX = init_temp, pressure_chamber, {oxidizer:1}
    density_ox = gas.density

    return density_fuel, density_ox

def Ve_func(k, Tc, M, pe, pc):
    Rmolar = 8314  # J/(kmol*K)
    return np.sqrt(2 * (Rmolar * k) / (k - 1) * Tc / M * (1 - (pe / pc) ** ((k - 1) / k)))

def Dt_mm(k, Tc, M, m_dot, pc, R):
    At =  m_dot / ((pc * np.sqrt(k)) / np.sqrt(R * Tc) * (2 / (k + 1))**((k + 1) / (2 * (k - 1))))
    return At, np.sqrt(4*At/np.pi) * 1000

def De_mm(k, Tc, m_dot, pe, pc, R):
    At =  m_dot / ((pc * np.sqrt(k)) / np.sqrt(R * Tc) * (2 / (k + 1))**((k + 1) / (2 * (k - 1))))
    Ae = At * np.sqrt((k - 1) / 2) * (2 / (k + 1))**((k + 1) / (2 * (k - 1))) * 1 / ((pe / pc)**(1 / k) * np.sqrt(1 - (pe / pc)**((k - 1) / k)))
    return np.sqrt(4*Ae/np.pi) * 1000

def pc_func(k, Tc, m_dot, At, R):
    return m_dot/At * np.sqrt(Tc) *np.sqrt(R/k) * ((k+1)/2)**((k + 1) / (2 * (k - 1)))

def BurnTime(yaml_file, mass_flow_rate, O_F_ratio, temp, pressure, fuel, oxidizer, tolerance = 3e-2):
    # Load the gas model excluding Argon
    gas = ct.Solution(yaml_file)
    
    # Initial values
    combustor_volume = 0.01  # m^3 - 10 litara (maksimalno- polako ce smanjivat)
    temperature_threshold = 500  # K
    combustor_update = 0 
    initial_change_rate = None # Variable to store the initial change rate

    # Define the fuel and oxidizer mixture
    m_fuel = 1 / (1 + O_F_ratio)
    m_ox = O_F_ratio / (1 + O_F_ratio)
    gas.TPY = temp, pressure, {fuel: m_fuel, oxidizer: m_ox}

    inlet = ct.Reservoir(gas)    # Create the inlet reservoir with the initial gas state
    exhaust = ct.Reservoir(gas)    # Create the exhaust reservoir

    # Prepare to store the results
    states = ct.SolutionArray(gas, extra=['volume', 'residence_time'])

    # Run the loop over decreasing volumes until the combustor is extinguished
    while True:
        # Create the combustor, and fill it initially with a mixture consisting of the equilibrium products of the inlet mixture.
        gas.equilibrate('HP')
        combustor = ct.IdealGasReactor(gas)
        combustor.volume = combustor_volume

        # Define the mass flow rate function
        def mdot(t):
            return mass_flow_rate

        inlet_mfc = ct.MassFlowController(inlet, combustor, mdot=mdot)
        outlet_mfc = ct.PressureController(combustor, exhaust, primary=inlet_mfc, K=0.01)

        # The simulation only contains one reactor
        sim = ct.ReactorNet([combustor])

        # Run the simulation to steady state
        sim.initial_time = 0.0  # reset the integrator
        sim.advance_to_steady_state()

        # Calculate residence time
        residence_time = combustor.volume / mass_flow_rate

        # Print current status
        # print(f'Volume = {combustor_volume:.2e} m^3; Temperature = {combustor.T:.1f} K; Residence time = {residence_time:.2e} s')

        # Store the state
        states.append(combustor.thermo.state, volume=combustor_volume, residence_time=residence_time)

        # Exit loop if temperature drops below the threshold (combustor is extinguished)
        if combustor.T < temperature_threshold:
            break

        # Calculate the change in heat release rate
        if combustor_update != 0:
            current_change_rate = 100 * abs(combustor.thermo.heat_release_rate - combustor_update) / combustor_update
            # print(f'Change in heat release rate: {current_change_rate:.2f}%')

            # Set the initial change rate if it's not already set
            if initial_change_rate is None:
                initial_change_rate = current_change_rate

            # Exit loop if the change in heat release rate is different from the initial value
            if abs(current_change_rate - initial_change_rate) > tolerance:  # Small tolerance for floating-point comparison
                # print("Change in heat release rate is different from the initial value. Stopping simulation.")
                break

        # Update combustor heat release rate for the next iteration
        combustor_update = combustor.thermo.heat_release_rate

        # Decrease the combustor volume for the next iteration
        combustor_volume *= 0.9

    # # Plot results: Heat release rate and temperature vs combustor volume
    # fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

    # # Plot heat release rate and temperature vs combustor volume
    # ax1.plot(states.volume, states.heat_release_rate, '.-', color='C0')
    # ax2 = ax1.twinx()
    # ax2.plot(states.volume, states.T, '.-', color='C1')
    # ax1.set_xlabel('Combustor volume [m$^3$]')
    # ax1.set_ylabel('Heat release rate [W/m$^3$]', color='C0')
    # ax2.set_ylabel('Temperature [K]', color='C1')

    # # Plot residence time vs combustor volume
    # ax3.plot(states.volume, states.residence_time, '.-', color='C2')
    # ax3.set_xlabel('Combustor volume [m$^3$]')
    # ax3.set_ylabel('Residence time [s]', color='C2')

    # fig.tight_layout()
    # plt.show()

    return states.residence_time[-1]

def tank_dimensions():
    
    global L_D_ratio, V_fuel, V_gas, V_liquid
    
    Volume_tank = V_fuel + V_gas + V_liquid
    
    body_diameter = ((Volume_tank * 4) / (3.14*15))**(1/3)
    tank_height = L_D_ratio * body_diameter
    fuel_tank_diameter = np.sqrt((V_fuel*4)/(tank_height * 3.14))
    
    ROCKET_BODY_AREA = (body_diameter) ** 2 * np.pi / 4
    
    return body_diameter, tank_height, fuel_tank_diameter, ROCKET_BODY_AREA


def flight_dynamics(thrust, M=28.96, R=8314, L=0.008):
    global mass_f, mass_l, mass_v, velocity_old, distance_old, P_old
    
    if thrust > 0:  
        data['total_thrust'].append(thrust) 
        Ae = 3.14 * (De/1000)**2 / 4
        thrust = thrust + (P_EXIT - P_old) * Ae

    total_mass = DRY_WEIGHT + mass_l + mass_v + mass_f
    TW_ratioo = thrust / (total_mass * 9.81)
    
    if total_mass * 9.81 > thrust and thrust != 0:
        print("this rocket cannot fly")
    if thrust / (total_mass * 9.81) < 3 and thrust != 0:
        print(f"thrust to weight is low and is equal to {(thrust/(total_mass * 9.81)):.2f}")
        
    T = (INITIAL_TEMP_DEGREES + 273) - L * distance_old
    P = P_ATMOSPHERE * np.exp(-M * G * distance_old / (R * T))
    air_density = (P * M) / (R * T)
    drag_force = 1 / 2 * air_density * velocity_old ** 2 * DRAG_COEFF * ROCKET_BODY_AREA
        
    rocket_acceleration = (np.sin(LAUNCH_RAIL_ANGLE) * thrust) / total_mass - G - drag_force / total_mass
    velocity_new = velocity_old + rocket_acceleration * TIME_STEP
    distance_new = distance_old + velocity_new * TIME_STEP
    
    data['rockett_acceleration'].append(rocket_acceleration)
    data['rocket_velocity'].append(velocity_new)
    data['rocket_distance'].append(distance_new)
    data['TW_ratio'].append(TW_ratioo)    
    
    velocity_old, distance_old, P_old = velocity_new, distance_new, P

    
def rocketpy_sim():
    
    temp, density_v, pressure_v, density_l, pressure_l = calculate_initial_properties(INITIAL_TEMP_DEGREES, OXIDIZER, GAS)
    density_f = PropsSI('D', 'P', pressure_l, 'T', temp, FUEL)
    dinitrogen_oxide_liquid = rp.Fluid(name="N2O", density=density_l)
    dinitrogen_oxide_vapour = rp.Fluid(name="N2O_V", density=density_v)
    Ethanol = rp.Fluid(name="Ethanol", density=density_f)
    
    print(rp.TankGeometry.total_volume(V_gas+V_liquid))
    ox_volume = rp.TankGeometry.total_volume(V_gas+V_liquid)
    ox_tank = rp.MassBasedTank(name="oxtenk", geometry=ox_volume, flux_time=20 , liquid=dinitrogen_oxide_liquid, gas=dinitrogen_oxide_vapour, liquid_mass=mass_liquid, gas_mass=mass_vapour)
    print(ox_tank)
   

#main code starts here

R = calculate_specific_gas_constant()
temp, density_v, pressure_v, density_l, pressure_l = calculate_initial_properties(INITIAL_TEMP_DEGREES, OXIDIZER, GAS)
A_ox_injector = (D_MM / 1000) ** 2 * np.pi / 4
                                                                                                            # pressure vessel properties
V_liquid, V_gas, V_tank, mass_v, mass_l = calculate_tank_properties(density_l, density_v)
                                                                                                            # fuel properties
mass_f, V_fuel = calculate_fuel_properties( pressure_l, temp)
                                                                                                            #pressure vessel dimensions
body_diameter, tank_height, fuel_tank_diameter , ROCKET_BODY_AREA= tank_dimensions()
                                                                                                            # injector properties
ml_dot, mf_dot, A_fuel, D_fuel_mm, speed_OxInjector, speed_FInjector = calculate_injector_properties(
    C_D, A_ox_injector, density_l, pressure_l, PC, O_F, temp)


#geometry
df = CEA_cantera(YAML_FILE, O_F, FUEL_CEA, OXIDIZER, temp, PC)
At_m2, Dt = Dt_mm(df["gam"], df["t"],df["mw"], ml_dot + ml_dot/O_F, PC, df['R'])
De = De_mm(df["gam"], df["t"], ml_dot + ml_dot/O_F, P_EXIT, PC, df['R'])
# Ac_m2 , Dc_mm = Dc_mm(ml_dot + mf_dot, df["rho"], v3y) # teorethical
Dc_mm = 3.5 * Dt #dobiveno iz iskustvenih podataka
Ac = (Dc_mm/1000)**2 * np.pi/4


#initial conditions
burn_time = BurnTime(YAML_FILE, ml_dot + mf_dot, O_F, temp, PC, FUEL_CEA, OXIDIZER)
mix_time = 50/(1000 * speed_FInjector) #30mm - 100mm is average mix range for pintle
stay_time = burn_time + mix_time
volume_chamb = (mf_dot + ml_dot) * 1/df['rho'] * stay_time
l_star = volume_chamb / At_m2
l_chamber = volume_chamb / Ac

i, o_f_old , pressure_combustion, pressure_old = 0, 0, PC, 0
velocity_old, distance_old, P_old= 0, 0 , P_ATMOSPHERE
# Main simulation loop for liquid phase
while mass_l > ml_dot:
    pressure_old, thrust = run_liquid_phase_simulation(pressure_l, A_ox_injector)
    flight_dynamics(thrust)
    
mass_v_initial = mass_v

# Simulation loop for vapor phase
while mass_v > mass_v * 0.05 and pressure_v > 2* P_ATMOSPHERE:
    # print(mass_v)
    thrust = run_vapor_phase_simulation()
    flight_dynamics(thrust=0)

#rest of flight dynamics to apogee
while velocity_old > 0:
    flight_dynamics(thrust = 0)

# rocketpy_sim()

# Plot results
plot_results()
generate_csv_files()


