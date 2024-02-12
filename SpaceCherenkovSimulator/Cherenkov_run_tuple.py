import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import subprocess

import scipy

import uproot

from scipy.interpolate import interp1d

from uncertainties import ufloat, unumpy
import pyarrow
import copy

import pkg_resources

def read_root_file_photon_generation(root_file_relative_path:str):

    name_of_bash_script = create_root_bash_script(root_file_relative_path)
    name_of_output_file = "run_output_file.out"

    subprocess.run(f"chmod u+x ./{name_of_bash_script}",shell=True)
    subprocess.run(f"./{name_of_bash_script} | tr -d '[:blank:]' | head -n-2 > {name_of_output_file}",shell=True,capture_output=True)

    DF_of_output_hits = read_subprocess_text_to_DF(name_of_output_file)

    os.remove(name_of_bash_script)
    os.remove(name_of_output_file)

    return DF_of_output_hits

def read_subprocess_text_to_DF(name_of_output_file:str):

    columns_for_data = [
        "primary_or_secondary",
        "atomic_number",
        "atomic_mass",
        "particle_id",
        "event_id",
        "reaction_order_rnb",
        "primary_kinetic_energy",
        "kinetic_energy",
        "total_energy",
        "x_momentum",
        "y_momentum",
        "z_momentum",
        "time",
        "event_weight",
        "cross_section"
        ]
    
    try:
        DF_of_output_hits = pd.read_csv(name_of_output_file,skiprows=14, header=None,delimiter="*",engine="pyarrow").drop([0,1,17],axis=1)
    except pyarrow.lib.ArrowInvalid:
        DF_of_output_hits = pd.DataFrame(columns=columns_for_data)
            
    DF_of_output_hits.columns = columns_for_data
        
    return DF_of_output_hits

def create_root_bash_script(root_file_relative_path):

    name_of_bash_script = "root_running_bash_script.sh"

    with open(name_of_bash_script,"w") as root_script:
        root_script.write('#!/bin/bash\n')
        root_script.write('source /home/chrisswdavis/ROOT/root/bin/thisroot.sh\n')
        root_script.write('root << EOF\n')
        root_script.write('generalrun = TFile("' + root_file_relative_path + '")\n')
        root_script.write('TTree* photoncounts\n')
        root_script.write('generalrun.GetObject("RA_tuple_interaction;1",photoncounts)\n')
        root_script.write('photoncounts->Scan("*")\n')
        root_script.write('EOF')#& (DF_of_output_hits["kinetic_energy"] > 0.0)]

    return name_of_bash_script

photodetector_efficiency_BC = np.loadtxt(pkg_resources.resource_filename(__name__,'photon_detection_efficiencies/BC_PDE.csv'), 
                                      delimiter=',', dtype='float' )
photodetector_efficiency_interp_BC = interp1d(photodetector_efficiency_BC[:,0],photodetector_efficiency_BC[:,1]/100.,bounds_error=False,fill_value=0.0)

photodetector_efficiency_SiPM = np.loadtxt(pkg_resources.resource_filename(__name__,'photon_detection_efficiencies/SiPM_PDE.csv'), 
                                      delimiter=',', dtype='float' )
photodetector_efficiency_interp_SiPM = interp1d(photodetector_efficiency_BC[:,0],photodetector_efficiency_BC[:,1]/100.,bounds_error=False,fill_value=0.0)

# the PDE of SiPM
# def get_pde(lamb):
#   # lamb: 
#   de = pde_interp(lamb)
#   return de if de > 0. else 0.

# def get_photodetector_efficiency(lamb):
#   # lamb: 
#   detector_efficiency = photodetector_efficiency_interp(lamb)

#   return detector_efficiency * (detector_efficiency > 0)
#   #return de if de > 0. else 0.

# def random_reject_photodetector_efficiency(wavelength_nm):
#     pde_for_wavelength = photodetector_efficiency_interp(wavelength_nm)
#     accept_or_reject_choice = np.random.choice([0,1],
#                                                 p=[1-pde_for_wavelength,pde_for_wavelength])
#     return accept_or_reject_choice

def random_reject_photodetector_efficiency(accept_probabilities_list):

    uniform_values_array = np.random.rand(len(accept_probabilities_list))
    accept_or_reject_choices = accept_probabilities_list >= uniform_values_array

    return accept_or_reject_choices

class pulse_height_distribution():

    def __init__(self, generated_photons_DF,incoming_particles_per_second=1,label=None,number_of_events_simulated = None, save_generated_photons_DF=False, thresh=0.):

        self.label = label
        self.thresh = thresh

        random_reject_choices = random_reject_photodetector_efficiency(generated_photons_DF["Si_detection_probability"])
        generated_photons_DF_randomly_rejected = generated_photons_DF[random_reject_choices]

        self.calculate_pulse_height_values(generated_photons_DF_randomly_rejected)

        self.incoming_particles_per_second = incoming_particles_per_second
        if number_of_events_simulated is None:
            self.estimated_number_of_incoming_particles = generated_photons_DF["event_id"].max()
        else:
            self.estimated_number_of_incoming_particles = number_of_events_simulated

        self.estimated_observation_time_in_seconds = self.estimated_number_of_incoming_particles / self.incoming_particles_per_second

        self.kinetic_energies_series = generated_photons_DF_randomly_rejected["primary_kinetic_energy"]

        self.calculate_total_photon_inducing_event_count(generated_photons_DF_randomly_rejected)

        if save_generated_photons_DF == True:
            self.generated_photons_DF = generated_photons_DF
            self.generated_photons_DF_randomly_rejected = generated_photons_DF_randomly_rejected

    def calculate_total_photon_inducing_event_count(self, generated_photons_DF_randomly_rejected):
        self.total_photon_inducing_event_count = generated_photons_DF_randomly_rejected["event_id"].nunique()
        if self.total_photon_inducing_event_count == 0:
            self.total_photon_inducing_event_count_error = np.sqrt(1)
        else:
            self.total_photon_inducing_event_count_error = np.sqrt(self.total_photon_inducing_event_count)

    def parent_particle_quantile(self, quantile_number:float):

        return self.kinetic_energies_series.quantile(quantile_number)
    
    def parent_particle_quantile_raw(self, quantile_number:float):

        return self.kinetic_energies_series.quantile(quantile_number)

    def calculate_pulse_height_values(self, generated_photons_DF_randomly_rejected):
        
        self.pulse_height_values = generated_photons_DF_randomly_rejected.groupby("event_id",sort=False).agg({"Si_detection_probability":"count"})
        self.pulse_height_values =  self.pulse_height_values[self.pulse_height_values["Si_detection_probability"] >= self.thresh]

        ones_at_zero_values = self.pulse_height_values.applymap(lambda x: 1 if x==0 else 0)
        self.pulse_height_errors = np.sqrt(self.pulse_height_values + ones_at_zero_values) #add one to all values of 0, to better estimate errors

        # self.pulse_height_interp = interp1d(self.pulse_height_values)
        # self.pulse_height_errors_interp = interp1d(self.pulse_height_errors)

    def get_total_photon_count_per_second(self, threshold_photon_value = 0.0):

        pulse_height_values_above_threshold = self.pulse_height_values[self.pulse_height_values["Si_detection_probability"] >= threshold_photon_value]
        pulse_height_values_above_threshold_errors = self.pulse_height_errors[self.pulse_height_values["Si_detection_probability"] >= threshold_photon_value]

        total_photon_count_per_second = sum(unumpy.uarray(pulse_height_values_above_threshold, pulse_height_values_above_threshold_errors)) / self.estimated_observation_time_in_seconds

        return total_photon_count_per_second[0]
    
    def get_total_photon_inducing_event_count_per_second(self, threshold_photon_value = 0.0):

        pulse_height_values_above_threshold = self.pulse_height_values[self.pulse_height_values["Si_detection_probability"] >= threshold_photon_value]

        photon_inducing_event_count = len(pulse_height_values_above_threshold)
        if photon_inducing_event_count == 0:
            photon_inducing_event_count_error = np.sqrt(1)
        else:
            photon_inducing_event_count_error = np.sqrt(photon_inducing_event_count)

        total_photon_inducing_event_count_per_second = ufloat(photon_inducing_event_count,photon_inducing_event_count_error) / self.estimated_observation_time_in_seconds

        return total_photon_inducing_event_count_per_second

    def plot(self,return_values=False,**xargs):

        # self.pulse_height_values["Si_detection_probability"].hist(bins=100,**xargs)
        
        hist_values = np.histogram(self.pulse_height_values,bins=100)

        normalised_counts = hist_values[0] / self.estimated_observation_time_in_seconds
        plt.stairs(normalised_counts / np.diff(hist_values[1]),hist_values[1],label=self.label,**xargs)
        plt.grid(True)
        plt.xlabel("photon count")
        plt.ylabel("events / photon count / second")
        plt.title("pulse height distribution")
        plt.legend(loc="center left",bbox_to_anchor=(1.1,0.5))

        if return_values == True:
            return (normalised_counts / np.diff(hist_values[1]),
                                 hist_values[1])

    def plot_differential_parents(self,plot_quantiles=False,**xargs):

        # self.pulse_height_values["Si_detection_probability"].hist(bins=100,**xargs)
        
        hist_values = np.histogram(self.kinetic_energies_series,
                                   bins=np.geomspace(0.01,100_000,100))
        normalised_counts = hist_values[0] / self.estimated_observation_time_in_seconds
        plt.stairs(normalised_counts / np.diff(hist_values[1]),hist_values[1],label=self.label,**xargs)
        plt.grid(True)
        plt.xscale("log")
        plt.xlabel("kinetic energy (MeV)")
        plt.ylabel("particles / MeV / second")
        plt.title("parent particle kinetic energy differential spectrum")
        plt.legend(loc="center left",bbox_to_anchor=(1.1,0.5))

        self.plot_parent_quantiles(plot_quantiles)

    def plot_integral_parents(self,plot_quantiles=False,**xargs):

        # self.pulse_height_values["Si_detection_probability"].hist(bins=100,**xargs)
        
        hist_values = np.histogram(self.kinetic_energies_series,
                                   bins=np.geomspace(0.01,100_000,100))

        integral_values = np.flip(np.flip(hist_values[0]).cumsum())
        
        plt.stairs(integral_values / self.estimated_observation_time_in_seconds,
                   hist_values[1],label=self.label,**xargs)
        plt.grid(True)
        plt.xscale("log")
        plt.xlabel("kinetic energy (MeV)")
        plt.ylabel("particles / second")
        plt.title("parent particle kinetic energy integral spectrum")
        plt.legend(loc="center left",bbox_to_anchor=(1.1,0.5))

        self.plot_parent_quantiles(plot_quantiles)

    def plot_parent_quantiles(self, plot_quantiles):
        if plot_quantiles is True:
            print(f"quantile for 0.25 is: {self.parent_particle_quantile(0.25)}")
            plt.axvline(self.parent_particle_quantile(0.25),ls="--",label="25% quantile")
            print(f"quantile for 0.50 is: {self.parent_particle_quantile(0.50)}")
            plt.axvline(self.parent_particle_quantile(0.50),ls="--",label="50% quantile")
            print(f"quantile for 0.75 is: {self.parent_particle_quantile(0.75)}")
            plt.axvline(self.parent_particle_quantile(0.75),ls="--",label="75% quantile")

    # def __call__(self, n_photons_value):

    #     return(ufloat(self.pulse_height_interp(n_photons_value),
    #                   self.pulse_height_errors_interp(n_photons_value)))

    # def __add__(self, second_pulse_height_distribution:pulse_height_distribution):
    #     new_pulse_height_distribution = copy.deepcopy(self)

    #     new_pulse_height_distribution.pulse_height_values = 


class pulse_height_distribution_averaged_method(pulse_height_distribution):

    def calculate_pulse_height_values(self, generated_photons_DF):
        self.pulse_height_values = generated_photons_DF.groupby("event_id",sort=False).agg({"Si_detection_probability":sum})


class Cherenkov_run_tuple():

    def __init__(self, 
                 file_path=None,
                 DF_of_output_hits=None, 
                 Cherenkov_run_label = None,
                 incoming_particles_per_second=1,
                 number_of_events_simulated = None, 
                 save_DF_of_output_hits = False,
                 threshold_primary_energy=None,
                 use_experiment_PDE=False):

        self.Cherenkov_run_label = Cherenkov_run_label
        self.incoming_particles_per_second = incoming_particles_per_second
        self.number_of_events_simulated = number_of_events_simulated
        self.threshold_primary_energy = threshold_primary_energy

        if (file_path is None) and (DF_of_output_hits is None):
            raise Exception("error: neither file_path or DF_of_output_hits were given as an input!")

        if DF_of_output_hits is None:
            DF_of_output_hits = read_root_file_photon_generation(file_path)
    
        generated_photons_DF = DF_of_output_hits.query("`particle_id` == -22.0") # [(DF_of_output_hits["particle_id"] == -22.0)] 
        if not (self.threshold_primary_energy is None):
            generated_photons_DF = generated_photons_DF.query(f"`primary_kinetic_energy` > {self.threshold_primary_energy}")

        print("successfully read in data...")

        generated_photons_DF = self.add_more_columns_to_generated_photons_DF(generated_photons_DF,
                                                                             use_experiment_PDE=use_experiment_PDE)
        print("assigned wavelengths and detector detection probabilities...")

        self.determine_pulse_height_distributions(generated_photons_DF)
        print("calculated pulse height values.")

        self.primary_particle_Cherenkov_tuple = split_Cherenkov_run_tuple(generated_photons_DF[generated_photons_DF["primary_kinetic_energy"] >= 200.0],
                                                                          incoming_particles_per_second=incoming_particles_per_second,
                                                                          number_of_events_simulated = number_of_events_simulated,
                                                                          Cherenkov_run_label=f"{self.Cherenkov_run_label}, primaries")
        self.secondary_particle_Cherenkov_tuple = split_Cherenkov_run_tuple(generated_photons_DF[generated_photons_DF["primary_kinetic_energy"] < 200.0],
                                                                            incoming_particles_per_second=incoming_particles_per_second,
                                                                            number_of_events_simulated = number_of_events_simulated,
                                                                            Cherenkov_run_label=f"{self.Cherenkov_run_label}, secondaries")
        
        if save_DF_of_output_hits == True:
            self.DF_of_output_hits = DF_of_output_hits
            self.generated_photons_DF = generated_photons_DF

    def determine_pulse_height_distributions(self,generated_photons_DF,thresh=0.):
        self.pulse_height_distribution_monte_carlo = pulse_height_distribution(generated_photons_DF,
                                                                               incoming_particles_per_second=self.incoming_particles_per_second,
                                                                               label=self.Cherenkov_run_label,
                                                                               number_of_events_simulated = self.number_of_events_simulated,
                                                                               thresh=thresh)
        self.pulse_height_distribution_averaged_method = pulse_height_distribution_averaged_method(generated_photons_DF,
                                                                                                   incoming_particles_per_second=self.incoming_particles_per_second,
                                                                                                   label=self.Cherenkov_run_label,
                                                                                                   number_of_events_simulated = self.number_of_events_simulated,
                                                                                                   thresh=thresh)

    def get_fractions_of_particle_types(self,generated_photons_DF):

        return {"primaries":len(self.primary_particle_Cherenkov_tuple.generated_photons_DF)/len(generated_photons_DF),
                "secondaries":len(self.secondary_particle_Cherenkov_tuple.generated_photons_DF)/len(generated_photons_DF)}

    def add_more_columns_to_generated_photons_DF(self,generated_photons_DF, use_experiment_PDE=False):

        if use_experiment_PDE is True:
            photodetector_efficiency_interp = photodetector_efficiency_interp_SiPM
        else:
            photodetector_efficiency_interp = photodetector_efficiency_interp_BC

        generated_photons_DF["wavelength_nm"] = self.get_photon_wavelength_nm(generated_photons_DF["total_energy"]*1e6)
        if len(generated_photons_DF) > 0:
            generated_photons_DF["Si_detection_probability"] = photodetector_efficiency_interp(generated_photons_DF["wavelength_nm"])
        else:
            generated_photons_DF["Si_detection_probability"] = generated_photons_DF["wavelength_nm"] #set to arbitrary empty column just to make the new column empty
        return generated_photons_DF

    def get_photon_wavelength_nm(self, energy_in_eV):

        energy_in_J = energy_in_eV * 1.6e-19

        lamb_in_m = scipy.constants.h * scipy.constants.c / energy_in_J

        lamb_in_nm = lamb_in_m / 1e-9

        return lamb_in_nm
    
    def random_reject_pde(wavelength_nm, use_experiment_PDE=False):
        if use_experiment_PDE is True:
            pde_for_wavelength = photodetector_efficiency_interp_SiPM(wavelength_nm)
        else:
            pde_for_wavelength = photodetector_efficiency_interp_BC(wavelength_nm)
        accept_or_reject_choice = np.random.choice([0,1],
                                                    p=[1-pde_for_wavelength,pde_for_wavelength])
        return accept_or_reject_choice

    def plot_pulse_height_distribution(self, **xargs):
        self.pulse_height_distribution_monte_carlo.plot(**xargs)

    def plot_smoother_pulse_height_distribution(self, **xargs):
        self.pulse_height_distribution_averaged_method.plot(**xargs)

    def plot_integral_primary_spectra(self, **xargs):
        self.pulse_height_distribution_monte_carlo.plot_integral_parents(**xargs)

    def plot_differential_primary_spectra(self, **xargs):
        self.pulse_height_distribution_monte_carlo.plot_differential_parents(**xargs)

class multi_Cherenkov_run_tuple(Cherenkov_run_tuple):

    def __init__(self, 
                 file_path:str, 
                 Cherenkov_run_label = None,
                 incoming_particles_per_second=1,
                 number_of_events_simulated = None,
                 threshold_primary_energy=None, 
                 save_DF_of_output_hits=False,
                 use_experiment_PDE=False):
        
        self.Cherenkov_run_label = Cherenkov_run_label
        self.incoming_particles_per_second = incoming_particles_per_second
        self.number_of_events_simulated = number_of_events_simulated
        self.threshold_primary_energy = threshold_primary_energy
        self.save_DF_of_output_hits = save_DF_of_output_hits
        self.use_experiment_PDE = use_experiment_PDE
    
        self.hits, self.output_tuples, self.coincidence_tuple_dictionary = self.construct_hits_dictionary(file_path)

    def construct_hits_dictionary(self, file_path):
        rfile = uproot.open(file_path)
        #self.rfile = rfile
        full_hit_DFs = {}
        hits = {}
        output_tuples = {}
        for key in rfile.keys():
            if '_tuple_interaction;1' in key :
                df = rfile[key].arrays(["evt","id","ekin","etot","ekin_prim"],library='pd')
                df.rename(columns={'evt':'event_id','id':'particle_id','ekin':'kinetic_energy','etot':'total_energy','ekin_prim':'primary_kinetic_energy'},inplace=True)
                self.process_cherenkov_photon_DF(df)
                if not (self.threshold_primary_energy is None):
                        self.generated_photons_DF = self.generated_photons_DF.query(f"`primary_kinetic_energy` > {self.threshold_primary_energy}")
                hits[str(key)] =  self.generated_photons_DF
                full_hit_DFs[str(key)] = df
                output_tuples[str(key)] =  Cherenkov_run_tuple(DF_of_output_hits=self.generated_photons_DF,
                                                               incoming_particles_per_second = self.incoming_particles_per_second, 
                                                               Cherenkov_run_label=self.Cherenkov_run_label, 
                                                               number_of_events_simulated = self.number_of_events_simulated,
                                                               save_DF_of_output_hits = self.save_DF_of_output_hits,
                                                               use_experiment_PDE=self.use_experiment_PDE)

            if '_tuple_fluence;1' in key :
                df = rfile[key].arrays(["event","pdg","primarykine","primarymomx","primarymomy","primarymomz"],library='pd')
                df.rename(columns={'event':'event_id','pdg':'particle_id','primarykine':'primary_kinetic_energy'},inplace=True)
                primary_particle_id = df.iloc[0]["particle_id"]
                self.generated_primary_DF = df.query(f"`particle_id` == {primary_particle_id}").drop_duplicates(subset=["event_id"])
                
                self.process_fluence_DF(df)
                hits[str(key)+'_PR'] = self.generated_primary_DF
                full_hit_DFs[str(key)+'_PR'] = self.generated_primary_DF
                
                df = rfile[key].arrays(["event","pdg","kine","primarykine"],library='pd')
                df.rename(columns={'event':'event_id','pdg':'particle_id','kine':'kinetic_energy','primarykine':'primary_kinetic_energy'},inplace=True)
                #df.assign("total_energy" = df ['kinetic_energy'])
                df.eval('total_energy=kinetic_energy',inplace=True)
                full_hit_DFs[str(key)+'_CK'] = df
                self.process_cherenkov_photon_DF(df)
                if not (self.threshold_primary_energy is None):
                        self.generated_photons_DF = self.generated_photons_DF.query(f"`primary_kinetic_energy` > {self.threshold_primary_energy}")
                hits[str(key)+'_CK'] = self.generated_photons_DF
                output_tuples[str(key) + '_CK'] =  Cherenkov_run_tuple(DF_of_output_hits=self.generated_photons_DF,
                                                               incoming_particles_per_second = self.incoming_particles_per_second,
                                                                Cherenkov_run_label=self.Cherenkov_run_label, 
                                                                number_of_events_simulated = self.number_of_events_simulated,
                                                                save_DF_of_output_hits = self.save_DF_of_output_hits)

        coincidence_tuple_dictionary = {}
        list_of_radiator_ids = [1,2]
        for first_radiator_id in [1]: #list_of_radiator_ids:
            coincidence_tuple_dictionary[first_radiator_id] = {}
            for second_radiator_id in [2]: #list_of_radiator_ids:
                coincidence_tuple_dictionary[first_radiator_id][second_radiator_id] = {}
                for FL_or_RA in ["FL","RA"]:
                    coincidence_DF = self.get_coincidence_DF(full_hit_DFs, id1=first_radiator_id, id2=second_radiator_id, FL_or_RA=FL_or_RA)
                    #coincidence_DF["Si_detection_probability"] = coincidence_DF["Si_detection_probability_x"]
                    #coincidence_DF["total_energy"] = coincidence_DF["total_energy_x"]
                    #coincidence_DF["particle_id"] = coincidence_DF["particle_id_x"]
                    coincidence_tuple = Cherenkov_run_tuple(DF_of_output_hits=coincidence_DF,
                                                            incoming_particles_per_second = self.incoming_particles_per_second, 
                                                            Cherenkov_run_label=f"{self.Cherenkov_run_label}, radiators = [{first_radiator_id},{second_radiator_id}], {FL_or_RA} coincidence", 
                                                            number_of_events_simulated = self.number_of_events_simulated,
                                                            save_DF_of_output_hits = self.save_DF_of_output_hits)
                    coincidence_tuple_dictionary[first_radiator_id][second_radiator_id][FL_or_RA] = coincidence_tuple

        return hits, output_tuples, coincidence_tuple_dictionary
    
    def process_fluence_DF(self,df:pd.DataFrame):
        self.convert_to_theta_phi_DF()
        # add the angles
        
    def set_default_DF(self,df:pd.DataFrame, thresh = 0.):
        self.generated_photons_DF = df
        self.determine_pulse_height_distributions(self.generated_photons_DF, thresh=thresh)
        #print("calculated pulse height values.")

        self.primary_particle_Cherenkov_tuple = split_Cherenkov_run_tuple(self.generated_photons_DF[self.generated_photons_DF["primary_kinetic_energy"] >= 200.0],
                                                                            Cherenkov_run_label=f"{self.Cherenkov_run_label}, primaries")
        self.secondary_particle_Cherenkov_tuple = split_Cherenkov_run_tuple(self.generated_photons_DF[self.generated_photons_DF["primary_kinetic_energy"] < 200.0],
                                                                            Cherenkov_run_label=f"{self.Cherenkov_run_label}, secondaries")

    def process_cherenkov_photon_DF(self,df:pd.DataFrame):
        self.generated_photons_DF = df.query("`particle_id` == -22.0")
        if not (self.threshold_primary_energy is None):
            self.generated_photons_DF = self.generated_photons_DF.query(f"`primary_kinetic_energy` > {self.threshold_primary_energy}")
        
        self.add_more_columns_to_generated_photons_DF(self.generated_photons_DF)
        #print("assigned wavelengths and detector detection probabilities...")

        return self.generated_photons_DF

    def convert_to_theta_phi_DF(self):
        # φ=arctan(y/x) and θ=arccos(z/r)
        self.generated_primary_DF["phi"] = np.degrees(np.arctan2(self.generated_primary_DF["primarymomy"],self.generated_primary_DF["primarymomx"]))
        r = np.sqrt(np.square(self.generated_primary_DF["primarymomx"]) + np.square(self.generated_primary_DF["primarymomy"]) + np.square(self.generated_primary_DF["primarymomz"]))
        self.generated_primary_DF["theta"] = np.degrees(np.arccos(self.generated_primary_DF["primarymomz"]/r))
        self.generated_primary_DF.drop(["primarymomx","primarymomy","primarymomz"],axis=1)
    
    def select_coincidence_events(self, df1, df2, dfp):
        self.generated_coincidence_DF = pd.merge(df1,df2, on='event_id')
        self.generated_coincidence_DF = pd.merge(self.generated_coincidence_DF,dfp, on='event_id')
        
    def get_coincidence_events(self, df1, df2, dfp, x_or_y_detector="x"):
        ## generated_coincidence_DF = pd.merge(df1,df2, on='event_id')
        left_coincidence_DF = df1[df1["event_id"].isin(df2["event_id"])]
        # right_coincidence_DF = df2[df2["event_id"].isin(df1["event_id"])]
        # generated_coincidence_DF = pd.concat([left_coincidence_DF,right_coincidence_DF],ignore_index=True).sort_values(by="event_id",ignore_index=True)
        # generated_coincidence_DF = pd.merge(generated_coincidence_DF,dfp, on='event_id',how="inner")
        generated_coincidence_DF = pd.merge(left_coincidence_DF,dfp, on='event_id',how="inner")
        ##generated_coincidence_DF["Si_detection_probability"] = generated_coincidence_DF[f"Si_detection_probability_{x_or_y_detector}"]
        generated_coincidence_DF["particle_id"] = generated_coincidence_DF["particle_id_x"]
        generated_coincidence_DF["primary_kinetic_energy"] = generated_coincidence_DF["primary_kinetic_energy_x"]

        return generated_coincidence_DF
    
    def select_anticoincidence_events(self, df1, df2, dp1):
        aevents = df2['event_id']
        self.generated_anticoincidence_DF = df1[~df1['event_id'].isin(aevents)]
        self.generated_anticoincidence_DF = pd.merge(self.generated_anticoincidence_DF,dp1, on='event_id')

    def get_relevant_hit_key(self, FL_or_RA:str,radiator_id:str):

        if FL_or_RA=="FL":
            suffix = '_tuple_fluence;1_CK'
        elif FL_or_RA=="RA":
            suffix = '_tuple_interaction;1'
        else:
            raise Exception("ERROR: FL_or_RA must be either 'FL' or 'RA'")
        
        return f"{FL_or_RA}{radiator_id}{suffix}"

    def plot_pulse_height_distribution_for_single_radiator(self, radiator_id:str, FL_or_RA="FL",thresh=0., **xargs):

        hit_key = self.get_relevant_hit_key(FL_or_RA, radiator_id)
        self.set_default_DF(self.hits[hit_key], thresh=thresh)
        self.plot_pulse_height_distribution()

    def get_coincidence_tuple(self, id1=1, id2=2, FL_or_RA="FL", thresh=0.):

        coincidence_df = self.get_coincidence_DF(id1, id2, FL_or_RA)

        coincidence_tuple = Cherenkov_run_tuple(DF_of_output_hits=coincidence_df,
                                                Cherenkov_run_label=f"{self.Cherenkov_run_label}, {FL_or_RA} coincidence",
                                                use_experiment_PDE=self.use_experiment_PDE)

        return coincidence_tuple
    
    def get_coincidence_DF(self, full_dict_of_hit_DFs, id1=1, id2=2, FL_or_RA="FL"):
        phs1 = full_dict_of_hit_DFs[self.get_relevant_hit_key(FL_or_RA, id1)] #.pulse_height_distribution_monte_carlo.pulse_height_values.reset_index()
        pri = full_dict_of_hit_DFs[f'FL{id1}_tuple_fluence;1_PR']

        phs2 = full_dict_of_hit_DFs[self.get_relevant_hit_key(FL_or_RA, id2)] #.pulse_height_distribution_monte_carlo.pulse_height_values.reset_index()
        coincidence_df = self.get_coincidence_events(self.process_cherenkov_photon_DF(phs1), self.process_cherenkov_photon_DF(phs2), pri)
        return coincidence_df

    # def get_coincidence_DF(self, id1=1, id2=2, FL_or_RA="FL"):
    #     phs1 = self.hits[self.get_relevant_hit_key(FL_or_RA, id1)] #.pulse_height_distribution_monte_carlo.pulse_height_values.reset_index()
    #     pri = self.hits[f'FL{id1}_tuple_fluence;1_PR']

    #     phs2 = self.hits[self.get_relevant_hit_key(FL_or_RA, id2)] #.pulse_height_distribution_monte_carlo.pulse_height_values.reset_index()
    #     coincidence_df = self.get_coincidence_events(phs1, phs2, pri)
    #     return coincidence_df

    def plot_coincidence_pulse_height_distribution(self, id1=1, id2=2, FL_or_RA="FL", thresh=0.):

        self.set_default_DF(self.hits[self.get_relevant_hit_key(FL_or_RA, id1)], thresh=thresh)
        phs1 = self.pulse_height_distribution_monte_carlo.pulse_height_values.reset_index()
        pri = self.hits[f'FL{id1}_tuple_fluence;1_PR']

        self.set_default_DF(self.hits[self.get_relevant_hit_key(FL_or_RA, id2)], thresh=thresh)
        phs2 = self.pulse_height_distribution_monte_carlo.pulse_height_values.reset_index()
        self.select_coincidence_events(phs1, phs2, pri)
        df = self.generated_coincidence_DF

        self.pulse_height_distribution_monte_carlo.pulse_height_values = df["Si_detection_probability_x"]
        self.pulse_height_distribution_monte_carlo.plot()

class split_Cherenkov_run_tuple(Cherenkov_run_tuple):

    def __init__(self, generated_photons_DF:pd.DataFrame, Cherenkov_run_label = None,incoming_particles_per_second=1, number_of_events_simulated = None,
                 save_DF_of_output_hits=False):

        self.Cherenkov_run_label = Cherenkov_run_label
        self.incoming_particles_per_second = incoming_particles_per_second
        self.number_of_events_simulated = number_of_events_simulated

        self.determine_pulse_height_distributions(generated_photons_DF)

        if save_DF_of_output_hits == True:
            self.generated_photons_DF = generated_photons_DF