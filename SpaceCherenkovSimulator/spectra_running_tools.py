from .tools_for_Cherenkov_runs import gras_Cherenkov_runner, Cherenkov_gdml_geometry_generator, gras_single_energy_macro, gras_spectrum_macro, gras_isotropic_macro, gras_input_macro_file_generator, single_particle_gras_Cherenkov_runner
import os
import numpy as np
from CosRayModifiedISO import CosRayModifiedISO
import datetime as dt
import matplotlib.pyplot as plt
import ParticleRigidityCalculationTools as PRCT

from uncertainties import ufloat
from uncertainties.umath import isnan

import joblib

mem = joblib.Memory('./Cherenkov_run_cache', verbose=1)

class orbit_type():

    def __init__(self,label:str,list_of_relevant_particle_spectra:list):

        self.label = label
        self.list_of_relevant_spectra = list_of_relevant_particle_spectra

class particle():

    def __init__(self,particle_name:str):
        self.particle_name = particle_name

class particle_spectrum():

    def __init__(self,particle_species:particle,spectrum_file_path:str,incoming_particles_per_s_per_cm2=None):

        self.particle_species_name = particle_species.particle_name
        self.spectrum_file_path = spectrum_file_path
        self.incoming_particles_per_s_per_cm2 = incoming_particles_per_s_per_cm2

    def get_particle_spectrum_macro(self):

        return gras_spectrum_macro(self.spectrum_file_path)
    
class single_energy_particle_spectrum():

    def __init__(self, particle_species:particle,single_particle_energy_MeV:float,incoming_particles_per_s_per_cm2=None):

        self.particle_species_name = particle_species.particle_name
        self.single_particle_energy_MeV = single_particle_energy_MeV
        self.incoming_particles_per_s_per_cm2 = incoming_particles_per_s_per_cm2

    def get_particle_spectrum_macro(self):

        return gras_single_energy_macro(self.single_particle_energy_MeV)
    
class spacecraft_shielding_geometry():

    def __init__(self,aluminium_thickness_in_mm=0.0,tantalum_thickness_in_mm=0.0):

        self.aluminium_thickness_in_mm = aluminium_thickness_in_mm
        self.tantalum_thickness_in_mm = tantalum_thickness_in_mm

class gras_Cherenkov_runner_from_objects(gras_Cherenkov_runner):

    generation_surface_radius = 2.0 #cm
    generation_surface_area_cm2 = 4*np.pi*(generation_surface_radius**2)

    def __init__(self, 
                 particle_spectrum_to_use:particle_spectrum,
                 shielding_geometry:spacecraft_shielding_geometry,
                 number_of_particles=10_000,verbose_output=False,
                 radiator_geometry="cube",
                 **kwargs):
        
        super().__init__(number_of_particles=number_of_particles,
                 particle_type=particle_spectrum_to_use.particle_species_name,
                 spectrum_file_to_use=particle_spectrum_to_use.spectrum_file_path,
                 incoming_particles_per_s=particle_spectrum_to_use.incoming_particles_per_s_per_cm2 * self.generation_surface_area_cm2,
                 verbose_output=verbose_output,
                 aluminium_thickness_in_mm=shielding_geometry.aluminium_thickness_in_mm, tantalum_thickness_in_mm=shielding_geometry.tantalum_thickness_in_mm,
                 radiator_geometry=radiator_geometry, 
                 **kwargs)

    def plot_pulse_height_distribution(self):
        self.outputted_tuple.plot_pulse_height_distribution()

    def get_total_photon_inducing_event_count_per_second(self,**kwargs):
        return self.outputted_tuple.pulse_height_distribution_monte_carlo.get_total_photon_inducing_event_count_per_second(**kwargs)
    
class single_particle_gras_Cherenkov_runner_from_objects(gras_Cherenkov_runner_from_objects,single_particle_gras_Cherenkov_runner):

    def __init__(self, 
                 particle_type:str,
                 energy_in_MeV:float,
                 incoming_particles_per_s_per_cm2:float,
                 shielding_geometry:spacecraft_shielding_geometry,
                 number_of_particles=10_000,verbose_output=False,
                 radiator_geometry="cube",**kwargs):
        
        single_particle_gras_Cherenkov_runner.__init__(self,
                 number_of_particles=number_of_particles,
                 particle_type=particle_type,
                 energy_in_MeV=energy_in_MeV,
                 incoming_particles_per_s=incoming_particles_per_s_per_cm2 * self.generation_surface_area_cm2,
                 verbose_output=verbose_output,
                 aluminium_thickness_in_mm=shielding_geometry.aluminium_thickness_in_mm, tantalum_thickness_in_mm=shielding_geometry.tantalum_thickness_in_mm,
                 radiator_geometry=radiator_geometry, 
                 **kwargs)
        
@mem.cache
def wrapper_single_particle_gras_Cherenkov_runner_from_objects(*args, **kwargs):
    return single_particle_gras_Cherenkov_runner_from_objects(*args, **kwargs)
    
@mem.cache
def wrapper_gras_Cherenkov_runner_from_objects(*args, **kwargs):
    return gras_Cherenkov_runner_from_objects(*args, **kwargs)

class GCR_Cherenkov_runner(gras_Cherenkov_runner_from_objects):

    def __init__(self, 
                 datetime_for_GCR:dt.datetime,
                 shielding_geometry=spacecraft_shielding_geometry(),
                 atomic_number_for_cosmic_rays = 1,
                 number_of_particles=10_000, **kwargs):

        particle_spectrum_to_use = self.get_particle_spectrum_and_generate_GRAS_file(datetime_for_GCR, atomic_number_for_cosmic_rays)

        super().__init__(particle_spectrum_to_use=particle_spectrum_to_use,
                 shielding_geometry=shielding_geometry,
                 number_of_particles=number_of_particles, **kwargs)

    def get_particle_spectrum_and_generate_GRAS_file(self, datetime_for_GCR, atomic_number_for_cosmic_rays):
        cosmic_ray_spectra = CosRayModifiedISO.getSpectrumUsingTimestamp(datetime_for_GCR,
                                                                       atomicNumber=atomic_number_for_cosmic_rays)

        incoming_particles_per_cm2_per_s = np.trapz(cosmic_ray_spectra["d_Flux / d_E (cm-2 s-1 sr-1 (MeV/n)-1)"],cosmic_ray_spectra["Energy (MeV/n)"]) * np.pi
        # default_generation_sphere_radius_cm = 2.0 #cm
        # default_generation_sphere_area_cm2 = 4 * np.pi * (default_generation_sphere_radius_cm**2)
        # incoming_particles_per_s = incoming_particles_per_cm2_per_s * default_generation_sphere_area_cm2

        cosmic_ray_spectrum_file_for_GRAS = "cosmic_ray_spectrum_file_for_GRAS.csv"

        if atomic_number_for_cosmic_rays == 2:
            cosmic_ray_spectra["Energy (MeV/n)"] = cosmic_ray_spectra["Energy (MeV/n)"] * 4
            cosmic_ray_spectra["d_Flux / d_E (cm-2 s-1 sr-1 (MeV/n)-1)"] = cosmic_ray_spectra["d_Flux / d_E (cm-2 s-1 sr-1 (MeV/n)-1)"] / 4

        cosmic_ray_spectra[["Energy (MeV/n)","d_Flux / d_E (cm-2 s-1 sr-1 (MeV/n)-1)"]].to_csv(cosmic_ray_spectrum_file_for_GRAS,index=False, header=False, sep=" ")

        particle_spectrum_to_use = particle_spectrum(particle_species=particle({1:"proton",2:"alpha"}[atomic_number_for_cosmic_rays]),
                                        spectrum_file_path=cosmic_ray_spectrum_file_for_GRAS,
                                        incoming_particles_per_s_per_cm2=incoming_particles_per_cm2_per_s)
                                        
        return particle_spectrum_to_use
    
@mem.cache
def wrapper_GCR_Cherenkov_runner(*args, **kwargs):
    return GCR_Cherenkov_runner(*args, **kwargs)

acquire_count_rate = np.vectorize(lambda output_run,threshold_photon_value=10.0:output_run.outputted_tuple.pulse_height_distribution_monte_carlo.get_total_photon_inducing_event_count_per_second(threshold_photon_value=threshold_photon_value))
get_nominal_values = np.vectorize(lambda value:value.nominal_value)
get_std_values = np.vectorize(lambda value:value.std_dev)
vectorised_isnan = np.vectorize(lambda value:isnan(value))    

def get_GCR_Cherenkov_run_rigidity_cut_off(rigidity_cut_off_GV, datetime_to_use:dt.datetime, atomic_number_for_cosmic_rays = 1, **kwargs):

    if atomic_number_for_cosmic_rays == 1:
        particleMassAU = 1
    elif atomic_number_for_cosmic_rays == 2:
        particleMassAU = 4

    specified_threshold_energy = PRCT.convertParticleRigidityToEnergy(rigidity_cut_off_GV, 
                                                                      particleMassAU = particleMassAU, 
                                                                      particleChargeAU = atomic_number_for_cosmic_rays).iloc[0]

    output_Cherenkov_run = wrapper_GCR_Cherenkov_runner(datetime_to_use,
                        shielding_geometry=default_shielding_geometry,
                        atomic_number_for_cosmic_rays = atomic_number_for_cosmic_rays,
                        #number_of_particles=number_of_particles,
                        threshold_primary_energy=specified_threshold_energy,
                        **kwargs)
    
    return output_Cherenkov_run

global no_shielding_geometry, default_shielding_geometry

no_shielding_geometry = spacecraft_shielding_geometry()
default_shielding_geometry = spacecraft_shielding_geometry(aluminium_thickness_in_mm=2.0,tantalum_thickness_in_mm=0.5)

def get_GLE_Cherenkov_run_rigidity_cut_off(rigidity_cut_off_GV, GLE_spec:particle_spectrum, number_of_particles=100_000, **kwargs):

    specified_threshold_energy = PRCT.convertParticleRigidityToEnergy(rigidity_cut_off_GV, particleMassAU = 1.0, particleChargeAU = 1.0).iloc[0]

    output_Cherenkov_run = wrapper_gras_Cherenkov_runner_from_objects(GLE_spec,
                                   default_shielding_geometry,
                                   number_of_particles=number_of_particles,
                                   threshold_primary_energy=specified_threshold_energy,
                                   **kwargs);
    
    return output_Cherenkov_run

def error_bar_from_array(error_bar_array, label=None):

    error_bar_array_to_plot = error_bar_array[~vectorised_isnan(get_nominal_values(error_bar_array[:,1]))]

    plt.errorbar(error_bar_array_to_plot[:,0],
                 get_nominal_values(error_bar_array_to_plot[:,1]),
                 get_std_values(error_bar_array_to_plot[:,1]),marker="o", label=label)
    
    plt.grid(True)

def get_integration_time_for_sigma(required_sigma,signal_flux,background_flux):

    try:
        return (required_sigma**2) * (signal_flux + background_flux) / (signal_flux**2)
    except ZeroDivisionError:
        signal_std_limit = (signal_flux.std_dev**2)
        return ufloat(np.nan,((required_sigma**2) * (signal_flux + background_flux) / signal_std_limit).std_dev)
    
#GLE21_integral_count_rate = 1.50668474e+001 * 4*np.pi # per cm2 per s
GLE21_integral_count_rate = 1.50668474e+001 * np.pi # per cm2 per s
#GLE05_integral_count_rate = 1328.16 * 4*np.pi # per cm2 per s
GLE05_integral_count_rate = 1328.16 * np.pi # per cm2 per s

datetime_for_GCR_solar_min = dt.datetime(
  year = 2020,
  month = 1,
  day = 1,
  hour = 0,
  minute = 0,
  second = 0
  )

datetime_for_GCR_solar_max = dt.datetime(
  year = 2000,
  month = 1,
  day = 1,
  hour = 0,
  minute = 0,
  second = 0
  )