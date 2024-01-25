import os
import shutil

import numpy as np
from .Cherenkov_run_tuple import Cherenkov_run_tuple, \
                                multi_Cherenkov_run_tuple, \
                                read_root_file_photon_generation
import matplotlib.pyplot as plt
import joblib

import pkg_resources

mem = joblib.Memory('./Cherenkov_run_cache', verbose=1)

def run_gras_simulation():
    with open("gras_bash_script.sh","w") as gras_bash_script:
        gras_bash_script.write("#!/bin/bash\n")
        gras_bash_script.write("source /home/chrisswdavis/gras-06-00-beta/config/gras-env.sh\n")
        gras_bash_script.write("gras template_gras_input_macro.in\n")

    os.system("chmod u+x gras_bash_script.sh")
    os.system("./gras_bash_script.sh")
    os.remove("gras_bash_script.sh")

class gras_input_direction_macro():

    def __init__(self):
        pass

    def get_direction_command(self):
        return self.direction_command
    
class gras_isotropic_macro(gras_input_direction_macro):

    def __init__(self, sphere_radius_in_mm=20):

        self.direction_command = "/gps/pos/type Surface\n" + \
                                 "/gps/pos/shape Sphere\n" + \
                                 f"/gps/pos/radius {sphere_radius_in_mm} mm\n" + \
                                 "/gps/ang/type cos\n" + \
                                 "/gps/ang/maxtheta 90 deg\n"
        
class gras_single_direction_macro(gras_input_direction_macro):

    def __init__(self, beam_location_in_mm: list, beam_momentum_direction: list):

        self.direction_command = "/gps/pos/type Point\n" + \
                                 f"/gps/pos/centre {beam_location_in_mm[0]} {beam_location_in_mm[1]} {beam_location_in_mm[2]} mm\n" + \
                                 f"/gps/direction {beam_momentum_direction[0]} {beam_momentum_direction[1]} {beam_momentum_direction[2]}\n" + \
                                 "/gps/ang/type planar\n"
        
class gras_input_energy_macro():

    def __init__(self):
        pass

    def get_energy_command(self):
        return self.energy_command
    
class gras_single_energy_macro(gras_input_energy_macro):

    def __init__(self, energy_in_MeV):

        self.energy_command = f"/gps/energy {energy_in_MeV} MeV"

class gras_spectrum_macro(gras_input_energy_macro):

    def __init__(self, path_to_spectrum_file):

        # self.energy_command = "/gps/ene/type Arb\n" + \
        #                       "/gps/ene/diffspec true\n" + \
        #                       "/gps/hist/type arb\n" + \
        #                       f"/gps/hist/file {path_to_spectrum_file}\n" + \
        #                       "/gps/hist/inter Lin\n"

        self.energy_command = "/gps/ene/type Arb\n" + \
                              "/gps/hist/type energy\n" + \
                              f"/gps/hist/file {os.path.basename(path_to_spectrum_file)}\n" + \
                              "/gps/hist/inter Lin\n"
        
class gras_input_macro_file_generator():

    template_macro_file_path = pkg_resources.resource_filename(__name__,"input_macro_templates/template_gras_input_macro.in")
    energy_replacement_string = "REPLACE_WITH_ENERGY_COMMANDS"
    direction_replacement_string = "REPLACE_WITH_DIRECTION_COMMANDS"
    
    output_macro_file_path = "gras_input_macro.in"

    def __init__(self, number_of_particles:int, particle_type:str, 
                 gras_energy_macro:gras_input_energy_macro, gras_direction_macro:gras_input_direction_macro):

        with open(self.template_macro_file_path,"r") as template_macro_file:
            template_file_string = template_macro_file.read()

        output_file_string = template_file_string.replace(self.energy_replacement_string, gras_energy_macro.get_energy_command()) \
                                                .replace(self.direction_replacement_string, gras_direction_macro.get_direction_command()) \
                                                .replace("REPLACE_WITH_PARTICLE", particle_type) \
                                                .replace("REPLACE_WITH_PNUMBER", str(number_of_particles))

        
        with open(self.output_macro_file_path,"w") as output_macro_file:
            output_macro_file.write(output_file_string)

class gras_input_macro_file_custom_file_selector(gras_input_macro_file_generator):

    def __init__(self, path_to_input_macro_file:str):

        with open("../" + path_to_input_macro_file,"r") as macro_file_to_use:
            file_string = macro_file_to_use.read()

        with open(self.output_macro_file_path,"w") as output_macro_file:
            output_macro_file.write(file_string)

class Cherenkov_gdml_geometry_generator():

    template_gdml_file_path = "template_gdml_file_to_use.gdml"

    output_gdml_file_path = "gdml_file_to_use.gdml"

    cubic_radiator_geometry_string = '<box lunit="mm" name="radiator_solid" x="10" y="10" z="10"/>'
    spherical_radiator_geometry_string = '<sphere lunit="mm" aunit="degree" name="radiator_solid"  rmin="0" rmax="5.0" startphi="0" deltaphi="360" starttheta="0" deltatheta="180"/>'
    dictionary_of_radiator_geometries = {"cube":cubic_radiator_geometry_string,"sphere":spherical_radiator_geometry_string}

    RMIN_Ta = 14.0

    def __init__(self, aluminium_thickness_in_mm=0.0, tantalum_thickness_in_mm=0.0,radiator_geometry="cube",number_of_radiators=1):

        with open(self.template_gdml_file_path,"r") as template_gdml_file:
            template_file_string = template_gdml_file.read()

        if tantalum_thickness_in_mm != 0.0:
            self.tantalum_placement_string = '<physvol name="tantalum_phys">\n' + \
                                            '<volumeref ref="tantalum_shell_log"/>\n' + \
                                            '<position name="tantalum_pos" unit="mm" x="0" y="0" z="0"/>\n' + \
                                            '</physvol>\n'
        else:
            self.tantalum_placement_string = ''
            tantalum_thickness_in_mm = 0.001 #set to arbitrary thickness for the purpose of making Geant4 still work

        if aluminium_thickness_in_mm != 0.0:
            self.aluminium_placement_string = '<physvol name="aluminium_phys">\n' + \
                                            '<volumeref ref="spacecraft_shell_log"/>\n' + \
                                            '<position name="aluminium_pos" unit="mm" x="0" y="0" z="0"/>\n' + \
                                            '</physvol>\n'
        else:
            self.aluminium_placement_string = ''
            aluminium_thickness_in_mm = 0.001 #set to arbitrary thickness for the purpose of making Geant4 still work

        self.RMAX_Ta = self.RMIN_Ta + tantalum_thickness_in_mm

        self.RMIN_Al = self.RMAX_Ta
        self.RMAX_Al = self.RMIN_Al + aluminium_thickness_in_mm

        output_file_string = template_file_string.replace("REPLACE_WITH_Ta_RMIN",f"{self.RMIN_Ta}") \
                                                 .replace("REPLACE_WITH_Ta_RMAX",f"{self.RMAX_Ta}") \
                                                 .replace("REPLACE_WITH_Al_RMIN",f"{self.RMIN_Al}") \
                                                 .replace("REPLACE_WITH_Al_RMAX",f"{self.RMAX_Al}") \
                                                 .replace("REPLACE_WITH_Ta_PLACEMENT_STRING",self.tantalum_placement_string) \
                                                 .replace("REPLACE_WITH_Al_PLACEMENT_STRING",self.aluminium_placement_string) \
                                                 .replace("REPLACE_WITH_RADIATOR_GEOMETRY",self.dictionary_of_radiator_geometries[radiator_geometry])
                                                 
        
        with open(self.output_gdml_file_path,"w") as output_gdml_file:
            output_gdml_file.write(output_file_string)

class Cherenkov_gdml_geometry_selector(Cherenkov_gdml_geometry_generator):

    def __init__(self, path_to_input_gdml_file:str):

        with open("../" + path_to_input_gdml_file,"r") as gdml_file_to_use:
            file_string = gdml_file_to_use.read()

        with open(self.output_gdml_file_path,"w") as output_gdml_file:
            output_gdml_file.write(file_string)

class gras_Cherenkov_runner():

    directory_to_run_in = "run_directory/"

    def __init__(self, 
                 number_of_particles=10_000,
                 particle_type="proton",
                 spectrum_file_to_use="GCR_spectrum.csv",
                 incoming_particles_per_s=None,
                 verbose_output=False,
                 aluminium_thickness_in_mm=5, tantalum_thickness_in_mm=0.5,
                 radiator_geometry="cube",
                 number_of_radiators=1,
                 custom_macro_file=None,
                 custom_gdml_file=None,
                 **kwargs):

        self.initialise_running_directory_and_macro(number_of_particles, 
                                                    particle_type, 
                                                    spectrum_file_to_use, 
                                                    number_of_radiators, 
                                                    custom_macro_file=custom_macro_file)

        try:
            self.setup_input_variables(incoming_particles_per_s, 
                                       verbose_output, 
                                       aluminium_thickness_in_mm, 
                                       tantalum_thickness_in_mm, 
                                       radiator_geometry, 
                                       number_of_radiators,
                                       custom_gdml_file=custom_gdml_file)
            
            self.outputted_tuple = self.calculate_Cherenkov_tuple(number_of_particles,number_of_radiators,**kwargs)

        except Exception as failed_exception:
            raise failed_exception
        finally:
            os.chdir("..")

    def setup_input_variables(self, incoming_particles_per_s, verbose_output, aluminium_thickness_in_mm, tantalum_thickness_in_mm, radiator_geometry, number_of_radiators=1, custom_gdml_file=None):
        self.incoming_particles_per_s = incoming_particles_per_s
        self.verbose_output = verbose_output
        if custom_gdml_file is None:
            self.generated_geometry = Cherenkov_gdml_geometry_generator(aluminium_thickness_in_mm,
                                                                        tantalum_thickness_in_mm,
                                                                        radiator_geometry=radiator_geometry,
                                                                        number_of_radiators=number_of_radiators)
        else:
            self.generated_geometry = Cherenkov_gdml_geometry_selector(custom_gdml_file)

    def initialise_running_directory_and_macro(self, 
                                               number_of_particles, 
                                               particle_type, 
                                               spectrum_file_to_use, 
                                               number_of_radiators=1,
                                               custom_macro_file=None):
        self.initialise_output_gras_directory(spectrum_file_to_use, number_of_radiators)
        os.chdir(self.directory_to_run_in)

        try:
            if custom_macro_file is None:
                self.generated_macro = gras_input_macro_file_generator(number_of_particles, 
                                                                    particle_type,
                                                                    gras_spectrum_macro(spectrum_file_to_use),
                                                                    gras_isotropic_macro())
            else:
                self.generated_macro = gras_input_macro_file_custom_file_selector(custom_macro_file)

        except Exception as failed_exception:
            os.chdir("..")
            raise failed_exception

    def initialise_output_gras_directory(self, spectrum_file_to_use, number_of_radiators=1):
        self.initialise_dir_and_copy_geom_and_mac_files(number_of_radiators)
        try:
            shutil.copyfile(pkg_resources.resource_filename(__name__,spectrum_file_to_use),self.directory_to_run_in + os.path.basename(spectrum_file_to_use))
        except FileNotFoundError:
            shutil.copyfile(spectrum_file_to_use,self.directory_to_run_in + os.path.basename(spectrum_file_to_use))

    def initialise_dir_and_copy_geom_and_mac_files(self, number_of_radiators):
        try:
            os.mkdir(self.directory_to_run_in)
        except FileExistsError:
            shutil.rmtree(self.directory_to_run_in)
            os.mkdir(self.directory_to_run_in)

        if number_of_radiators == 1:
            template_gdml_file = pkg_resources.resource_filename(__name__,"gdml_templates/template_gdml_file_to_use_1_radiators.gdml")

            Analysis_file_path = pkg_resources.resource_filename(__name__,"Analysis_macros/Analysis.g4mac")
            shutil.copyfile(Analysis_file_path,self.directory_to_run_in + "Analysis.g4mac")
        elif number_of_radiators == 2:
            template_gdml_file = pkg_resources.resource_filename(__name__,"gdml_templates/template_gdml_file_to_use_2_radiators.gdml")

            shutil.copyfile(pkg_resources.resource_filename(__name__,"Analysis_macros/Analysis_2_radiators.g4mac"),
                            self.directory_to_run_in + "Analysis.g4mac")
            shutil.copyfile(pkg_resources.resource_filename(__name__,"Analysis_macros/Analysis_reaction_m.g4mac"),
                            self.directory_to_run_in + "Analysis_reaction_m.g4mac")
            shutil.copyfile(pkg_resources.resource_filename(__name__,"Analysis_macros/Analysis_fluence_m.g4mac"),
                            self.directory_to_run_in + "Analysis_fluence_m.g4mac")

        shutil.copyfile(template_gdml_file,self.directory_to_run_in + "template_gdml_file_to_use.gdml")
        shutil.copyfile(pkg_resources.resource_filename(__name__,"input_macro_templates/template_gras_input_macro.in"),
                        self.directory_to_run_in + "template_gras_input_macro.in")

    def calculate_Cherenkov_tuple(self, number_of_particles, number_of_radiators,**kwargs):
        self.run_gras_simulation()

        if number_of_radiators == 1:

            DF_of_output_hits = read_root_file_photon_generation("output_Cherenkov_data.root")

            outputted_tuple = Cherenkov_run_tuple(DF_of_output_hits=DF_of_output_hits,
                                incoming_particles_per_second=self.incoming_particles_per_s,
                                number_of_events_simulated=number_of_particles,
                                **kwargs)

        else:
            outputted_tuple = multi_Cherenkov_run_tuple(file_path="output_Cherenkov_data.root", 
                                                        incoming_particles_per_second=self.incoming_particles_per_s,
                                                        number_of_events_simulated=number_of_particles,
                                                        **kwargs)
                            
        return outputted_tuple

    def run_gras_simulation(self):
        with open("gras_bash_script.sh","w") as gras_bash_script:
            gras_bash_script.write("#!/bin/bash\n")
            gras_bash_script.write("source /home/chrisswdavis/Geant4Installations/geant4-v10.7.4/install/bin/geant4.sh\n")
            gras_bash_script.write("source /home/chrisswdavis/gras-06-00-beta/config/gras-env.sh\n")
            gras_bash_script.write(f"gras {self.generated_macro.output_macro_file_path}\n")

        os.system("chmod u+x gras_bash_script.sh")
        if self.verbose_output == True:
            os.system("./gras_bash_script.sh")
        else:
            os.system("./gras_bash_script.sh >/dev/null 2>&1")

class single_particle_gras_Cherenkov_runner(gras_Cherenkov_runner):

    def __init__(self, 
                 number_of_particles=10_000,
                 particle_type="proton",
                 energy_in_MeV=2_000.0,
                 incoming_particles_per_s=None,
                 verbose_output=False,
                 aluminium_thickness_in_mm=5, tantalum_thickness_in_mm=0.5,
                 radiator_geometry="cube",
                 number_of_radiators=1,
                 custom_macro_file=None,
                 custom_gdml_file=None,
                 **kwargs):
        
        self.initialise_running_directory_and_macro(number_of_particles, particle_type, energy_in_MeV, number_of_radiators,custom_macro_file=custom_macro_file)

        self.setup_input_variables(incoming_particles_per_s, verbose_output, aluminium_thickness_in_mm, tantalum_thickness_in_mm, radiator_geometry, number_of_radiators,custom_gdml_file=custom_gdml_file)
        
        self.outputted_tuple = self.calculate_Cherenkov_tuple(number_of_particles, number_of_radiators,**kwargs)

        os.chdir("..")

    def initialise_running_directory_and_macro(self, number_of_particles, particle_type, energy_in_MeV, number_of_radiators=1,custom_macro_file=None):
        self.initialise_dir_and_copy_geom_and_mac_files(number_of_radiators)
        os.chdir(self.directory_to_run_in)
        if not custom_macro_file is None:
            self.generated_macro = gras_input_macro_file_generator(number_of_particles, particle_type,gras_single_energy_macro(energy_in_MeV),gras_isotropic_macro())
        else:
            self.generated_macro = gras_input_macro_file_custom_file_selector(custom_macro_file)

@mem.cache
def wrapper_gras_Cherenkov_runner(**xargs):

    outputted_gras_Cherenkov_run = gras_Cherenkov_runner(**xargs) 

    return outputted_gras_Cherenkov_run

def errorbar_plot_variations(output_array,**xargs):

    plt.errorbar(output_array[:,0],
             np.vectorize(lambda x:x.nominal_value)(output_array[:,1]),
             np.vectorize(lambda x:x.std_dev)(output_array[:,1]),
    marker="o",
    **xargs)

    plt.ylabel("fractional reduction in\nphoton-inducing events")

    plt.yscale('log')
    plt.axhline(1e-4,ls="--")
    plt.ylim([1e-5,2e0])
    plt.grid(True)