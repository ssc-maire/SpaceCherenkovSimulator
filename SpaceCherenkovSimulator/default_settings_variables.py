import pandas as pd
import numpy as np
import pkg_resources
from . import spectra_running_tools as SRT

def convert_integral_values_to_differential(integral_DF):

    differential_DF_coords = integral_DF.iloc[:,:2]

    differential_DF_energy_vals = (integral_DF.iloc[:,2:-1].columns +integral_DF.iloc[:,3:].columns)/2
    differential_DF_values = -1 * integral_DF.iloc[:,2:].diff(axis=1).iloc[:,1:] / np.diff(integral_DF.iloc[:,2:].columns)
    differential_DF_values.columns = differential_DF_energy_vals

    return pd.concat([differential_DF_coords,differential_DF_values],axis=1)

def get_IRENE8_electron_spectra(path_to_electron_IRENE8_file, spacecraft_coords_DF):
    IRENE8_450_electrons = pd.read_csv(path_to_electron_IRENE8_file,skiprows=76,header=None,skipfooter=1)
    IRENE8_450_electrons.columns = [
        "B Gauss",
        "L !8R!3!dE!n",
        0.04,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25,
        3.5,
        3.75,
        4,
        4.25,
        4.5,
        4.75,
        5,
        5.5,
        6,
        6.5,
        7
    ]
    IRENE8_450_electrons["B Gauss"] = spacecraft_coords_DF["Latitude_deg"]
    IRENE8_450_electrons.rename(columns={"B Gauss":"Latitude_deg"},inplace=True)
    
    IRENE8_450_electrons["L !8R!3!dE!n"] = spacecraft_coords_DF["Longitude_deg"]
    IRENE8_450_electrons.rename(columns={"L !8R!3!dE!n":"Longitude_deg"},inplace=True)
    
    IRENE8_450_electrons_differential = convert_integral_values_to_differential(IRENE8_450_electrons)
    
    IRENE8_450_electrons_differential_shortened = IRENE8_450_electrons_differential.drop(columns=IRENE8_450_electrons_differential.columns[2:11])
    
    IRENE8_450_electrons_differential_shortened_single_orbit = IRENE8_450_electrons_differential_shortened.iloc[790:850].sample(60,random_state=1).sort_index()
    
    return IRENE8_450_electrons_differential_shortened_single_orbit

def get_IRENE8_proton_spectra(path_to_proton_IRENE8_file, spacecraft_coords_DF):
    IRENE8_450_protons = pd.read_csv(path_to_proton_IRENE8_file,skiprows=76,header=None,skipfooter=1)
    IRENE8_450_protons.columns = [
        "B Gauss",
        "L !8R!3!dE!n",
        0.1, 
        0.15, 
        0.2, 
        0.3, 
        0.4, 
        0.5, 
        0.6, 
        0.7, 
        1, 
        1.5, 
        2, 
        3, 
        4, 
        5, 
        6, 
        7, 
        10, 
        15, 
        20, 
        30, 
        40, 
        50, 
        60, 
        70, 
        100, 
        150, 
        200, 
        300, 
        400
    ]
    IRENE8_450_protons["B Gauss"] = spacecraft_coords_DF["Latitude_deg"]
    IRENE8_450_protons.rename(columns={"B Gauss":"Latitude_deg"},inplace=True)
    
    IRENE8_450_protons["L !8R!3!dE!n"] = spacecraft_coords_DF["Longitude_deg"]
    IRENE8_450_protons.rename(columns={"L !8R!3!dE!n":"Longitude_deg"},inplace=True)
    
    IRENE8_450_protons_differential = convert_integral_values_to_differential(IRENE8_450_protons)
    
    IRENE8_450_protons_differential_single_orbit = IRENE8_450_protons_differential.iloc[790:850].sample(60,random_state=1).sort_index()
    
    return IRENE8_450_protons_differential_single_orbit

global default_path_to_electron_IRENE8_file
default_path_to_electron_IRENE8_file = pkg_resources.resource_filename(__name__,"IRENE8_input_data/IRENE8_450_circ_orbit_electrons.txt")

global default_path_to_proton_IRENE8_file
default_path_to_proton_IRENE8_file = pkg_resources.resource_filename(__name__,"IRENE8_input_data/IRENE8_450_circ_orbit_protons.txt")

global default_spacecraft_coords_DF
default_spacecraft_coords_DF = pd.read_csv(pkg_resources.resource_filename(__name__,"IRENE8_input_data/IRENE8_450_circ_orbit_coords.txt"),
                                   skiprows=73,header=None,skipfooter=1)
default_spacecraft_coords_DF.columns = ['ModifiedJulianDay',
'Altitude_km',
'Latitude_deg',
'Longitude_deg',
'LocalTime_hrs',
'PitchAngle_deg']
default_spacecraft_coords_DF["Minutes"] = (default_spacecraft_coords_DF["ModifiedJulianDay"] - default_spacecraft_coords_DF["ModifiedJulianDay"].iloc[0]) * (24 * 60)

global default_trajectory_DF
default_trajectory_DF = default_spacecraft_coords_DF.iloc[790:850]

global IRENE8_450_electrons_differential_shortened_single_orbit
IRENE8_450_electrons_differential_shortened_single_orbit = get_IRENE8_electron_spectra(default_path_to_electron_IRENE8_file, default_spacecraft_coords_DF)

global IRENE8_450_protons_differential_shortened_single_orbit
IRENE8_450_protons_differential_shortened_single_orbit = get_IRENE8_proton_spectra(default_path_to_proton_IRENE8_file, default_spacecraft_coords_DF)

global GLE21_spectrum
GLE21_spectrum = SRT.particle_spectrum(particle_species=SRT.particle("proton"),
                                        spectrum_file_path="interplanetary_spectra/GLE21spectrum_multipliedby1.csv",
                                        incoming_particles_per_s_per_cm2=SRT.GLE21_integral_count_rate)

global GLE05_spectrum
GLE05_spectrum = SRT.particle_spectrum(particle_species=SRT.particle("proton"),
                                        spectrum_file_path="interplanetary_spectra/GLE05spectrum_multipliedby1.csv",
                                        incoming_particles_per_s_per_cm2=SRT.GLE05_integral_count_rate)
