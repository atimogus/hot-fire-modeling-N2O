from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CEApy import CEA

def mass_evapouration(V_tank, mass_l_old, mass_v, density_v, density_l):
    mass_l = (V_tank-(mass_l_old+mass_v)/density_v)/(1/density_l-1/density_v)
    mass_vapourised = mass_l_old - mass_l
    mass_v += mass_vapourised
    return mass_l, mass_v, mass_vapourised

def get_n2o_properties_T(temp, gas, liquid, fluid):
    density_v, pressure_v = PropsSI('D','T',temp,'Q',gas,fluid), PropsSI('P','T',temp,'Q',gas,fluid) # promjena gustoce plina
    density_l, pressure_l = PropsSI('D','T',temp,'Q',liquid,fluid), PropsSI('P','T',temp,'Q',liquid,fluid)# promjena gustoce tekucine, pressure vapour = pressure liquid
    return density_v, pressure_v, density_l, pressure_l

def mdot_liquid_oxidizer_SPI(C_d, A, density_l, pressure_l, P_downstream_injector, time_step, mass_l):
    ml_dot = C_d * A * np.sqrt(2 * density_l * (pressure_l - P_downstream_injector)) * time_step
    mass_l -= ml_dot 
    mass_l_old = mass_l
    return ml_dot, mass_l, mass_l_old

def CEA_start(oxidizer, fuel_CEA, temp, pc, O_F):
    Rmolar = 8314  # J/(kmol*K)
    combustion = CEA()
    combustion.settings()
    
    oxid = [[oxidizer, 100, temp]]
    fuel = [[fuel_CEA, 100, temp]]
    combustion.input_propellants(oxid=oxid, fuel=fuel)
    combustion.input_parameters(chamber_pressure=[pc], of_ratio=[O_F, O_F])
    combustion.output_parameters(user_outputs=['t', 'rho', 'h', 'g', 's', 'mw', 'cp', 'gam'])
    
    # Running analyses
    try:
        combustion.run()
    except Exception as e:
        print(f"Error running CEA analysis: {e}")
    # Getting results
    df = combustion.get_results()
    df['R'] = Rmolar/df["mw"][0]
    
    return df

def Ve_func(k, Tc, M, pe, pc):
    Rmolar = 8314  # J/(kmol*K)
    return np.sqrt(2 * (Rmolar * k) / (k - 1) * Tc / M * (1 - (pe / pc) ** ((k - 1) / k)))

def Dt_mm(k, Tc, M, m_dot, pc, R):
    At =  m_dot / ((pc * np.sqrt(k)) / np.sqrt(R * Tc) * (2 / (k + 1))**((k + 1) / (2 * (k - 1))))
    return At, np.sqrt(4*At/np.pi) * 1000

def De_mm(k, Tc, M, m_dot, pe, pc, R):
    At =  m_dot / ((pc * np.sqrt(k)) / np.sqrt(R * Tc) * (2 / (k + 1))**((k + 1) / (2 * (k - 1))))
    Ae = At * np.sqrt((k - 1) / 2) * (2 / (k + 1))**((k + 1) / (2 * (k - 1))) * 1 / ((pe / pc)**(1 / k) * np.sqrt(1 - (pe / pc)**((k - 1) / k)))
    return np.sqrt(4*Ae/np.pi) * 1000

def pc_func(k, Tc, M, m_dot, pe, At, R):
    return m_dot/At * np.sqrt(Tc) *np.sqrt(R/k) * ((k+1)/2)**((k + 1) / (2 * (k - 1)))

def thrust(mdot, Ve):
    return mdot * Ve
