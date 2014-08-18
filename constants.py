import numpy as np

pi = np.pi

#-- Unit conversions
eV_per_J   = 6.24150965e18
J_per_eV   = 1/eV_per_J # Conversion factor from [eV] to [J]
J_per_cal  = 4.186     # Conversion factor from calories to Joules
J_per_kcal = 4186.0    # Conversion factor from Calories [kcal] to Joules
Pa_per_atm = 1.013e5   # Pascals per 1 Atmosphere

#-- Constants
kB_eV    = 8.617e-5          # Boltzmann's constant [eV/K]
kB       = 1.381e-23         # Boltzmann's constant [J/K]
h        = 6.62606896e-34    # Planck's constant [J s]
hbar     = h/(2*np.pi)       # reduced Planck's constant [J s]
h_eV     = h/J_per_eV        # Planck's constant [J s]
N_A      = 6.022e23          # Number, N_A [unitless]
c        = 2.99792458e8      # speed of light [m/s]
sigma    = 5.670400e-8       # stefan-boltzman constant [W m^-2 K^-4]
m_sun    = 2e30              # mass of sun [kg]
P_sun    = 3.9e26            # power output of sun [W]
epsilon0 = 8.854187817*1e-12 # [a s / (V m)] or [F/m]
mu0      = 4*np.pi*1e-7      # [H/m] or [N/A^2]
Z0       = 376.730313461     # Characteristic impedance of vacuum [Ohm]
K_J      = 483597.891e9      # Josephson constant [Hz/V]

m_e      = 9.10938215e-31    # mass of electron [kg]
m_p      = 1.672621637e-27   # mass of electron [kg]
q_e      = 1.602176487e-19   # elementary charge (charge of electron) [C]
a0       = 52.917720859e-12  # bohr radius [m]
alpha    = q_e**2/(4*pi*epsilon0*hbar*c) # fine-structure constant
g_d      = 0.8574382308      # g-factor for deuteron (neutron + proton)
g_e      = -2.0023193043622  # g-factor for electron
g_p      = 5.585694713       # g-factor for single proton (helium nucleus)
R_inf    = 10973731.568527   # Rydberg constant [1/m]
mu_N     = 5.05078324e-27    # Nuclear magneton [J/T]
mu_B     = 927.400915e-26    # Bohr magneton [J/T]

Mm  = np.float('7.3477e22')     # mass of moon [kg]
Me  = np.float('5.9742e24')     # mass of earth [kg]
Ms  = np.float('1.9891e30')     # mass of sun [kg]
Dm  = np.float('384403.0e3')    # dist from earth center to moon center [m]
Ds0 = np.float('147102790.0e3') # distance from earth to sun, perihelion [m]
Ds1 = np.float('152102302.0e3') # distance from earth to sun, aphelion [m]
Ds  = np.float('149.6e9')       # mean distance, earth to sun [m]
G   = np.float('6.67428e-11')   # grav constant [m^3 kg^-1 s^-2]
g   = np.float('9.80665')       # accepted standard accel of gravity
rE0 = np.float('6357.0e3')      # earth min radius [m]
rE1 = np.float('6378.0e3')      # earth max radius [m]
rE  = np.float('6.37e6')        # earth averaged radius [m]
wEd = np.float(2*np.pi/(24*3600)) # earth angular velocity w.r.t. sun [rad s^-1]
wE  = np.float('7.2921159e-5')  # earth ang vel w.r.t. fixed stars [rad s^-1]

atm = 101325                    # adopted standard atmosphere [Pa]
N_A = 6.02214179e23             # Avogadro constant [1/mol]
