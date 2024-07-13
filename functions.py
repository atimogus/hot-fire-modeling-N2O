from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cantera as ct

def mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l):
    mass_l = (V_tank-(mass_l_old+mass_v)/density_v)/(1/density_l-1/density_v)
    mass_vapourised = mass_l_old - mass_l
    mass_v += mass_vapourised
    return mass_l, mass_v, mass_vapourised

def get_n2o_properties_T(temp, gas, liquid, fluid):
    density_v, pressure_v = PropsSI('D','T',temp,'Q',gas,fluid), PropsSI('P','T',temp,'Q',gas,fluid) # promjena gustoce plina
    density_l, pressure_l = PropsSI('D','T',temp,'Q',liquid,fluid), PropsSI('P','T',temp,'Q',liquid,fluid)# promjena gustoce tekucine, pressure vapour = pressure liquid
    return density_v, pressure_v, density_l, pressure_l

def mdot_liquid_oxidizer_SPI(C_d, A, density_l, pressure_l, P_downstream_injector, time_step, mass_l, n_orfices_ox):
    ml_dot = n_orfices_ox * C_d * A * np.sqrt(2 * density_l * (pressure_l - P_downstream_injector)) * time_step
    mass_l -= ml_dot 
    mass_l_old = mass_l
    return ml_dot, mass_l, mass_l_old

def CEA_cantera(yaml_file, O_F_ratio, fuel, oxidizer, init_temp, pressure_chamber):
    
    # Create gas object
    gas = ct.Solution(yaml_file)
    
    gas.TPX = init_temp, pressure_chamber, {fuel:1} # odabire svojstva mjesavine na 300K i 1atmosferi
    density_fuel = gas.density
    gas.TPX = init_temp, pressure_chamber, {oxidizer:1}
    density_ox = gas.density
    
    # Calculate mass fractions
    m_fuel = 1 / (1 + O_F_ratio)
    m_ox = O_F_ratio / (1 + O_F_ratio)
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

def pc_func(k, Tc, M, m_dot, pe, At, R):
    return m_dot/At * np.sqrt(Tc) *np.sqrt(R/k) * ((k+1)/2)**((k + 1) / (2 * (k - 1)))

def thrust(mdot, Ve):
    return mdot * Ve

def Dc_mm(mdot, density, speed):
    Ac_m2 = mdot / (density * speed)
    Dc_mm = np.sqrt(4*Ac_m2/np.pi) * 1000
    return Ac_m2 , Dc_mm

def BurnTime(yaml_file, mass_flow_rate, O_F_ratio, temp, pressure, fuel, oxidizer, tolerance = 3e-2):
    # Load the gas model excluding Argon
    gas = ct.Solution(yaml_file)
    
    # Initial values
    combustor_volume = 0.01  # m^3 - 10 litara
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
        print(f'Volume = {combustor_volume:.2e} m^3; Temperature = {combustor.T:.1f} K; Residence time = {residence_time:.2e} s')

        # Store the state
        states.append(combustor.thermo.state, volume=combustor_volume, residence_time=residence_time)

        # Exit loop if temperature drops below the threshold (combustor is extinguished)
        if combustor.T < temperature_threshold:
            break

        # Calculate the change in heat release rate
        if combustor_update != 0:
            current_change_rate = 100 * abs(combustor.thermo.heat_release_rate - combustor_update) / combustor_update
            print(f'Change in heat release rate: {current_change_rate:.2f}%')

            # Set the initial change rate if it's not already set
            if initial_change_rate is None:
                initial_change_rate = current_change_rate

            # Exit loop if the change in heat release rate is different from the initial value
            if abs(current_change_rate - initial_change_rate) > tolerance:  # Small tolerance for floating-point comparison
                print("Change in heat release rate is different from the initial value. Stopping simulation.")
                break

        # Update combustor heat release rate for the next iteration
        combustor_update = combustor.thermo.heat_release_rate

        # Decrease the combustor volume for the next iteration
        combustor_volume *= 0.9

    # Plot results: Heat release rate and temperature vs combustor volume
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot heat release rate and temperature vs combustor volume
    ax1.plot(states.volume, states.heat_release_rate, '.-', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(states.volume, states.T, '.-', color='C1')
    ax1.set_xlabel('Combustor volume [m$^3$]')
    ax1.set_ylabel('Heat release rate [W/m$^3$]', color='C0')
    ax2.set_ylabel('Temperature [K]', color='C1')

    # Plot residence time vs combustor volume
    ax3.plot(states.volume, states.residence_time, '.-', color='C2')
    ax3.set_xlabel('Combustor volume [m$^3$]')
    ax3.set_ylabel('Residence time [s]', color='C2')

    fig.tight_layout()
    plt.show()

    return states.residence_time[-1]
