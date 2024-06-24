from setuptools import find_packages, setup
import os
import glob as gb

# get requirements for installation
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='SpaceCherenkovSimulator',
    packages=find_packages(exclude='tests'),
    package_data={"SpaceCherenkovSimulator":[
                                            "gdml_templates/*.gdml",
                                            "IRENE8_input_data/*.txt",
                                            "Analysis_macros/*.g4mac",
                                            "photon_detection_efficiencies/*.csv",
                                            "input_macro_templates/*.in",
                                            "interplanetary_spectra/*.csv",
                                         ]},
    version='0.2.8',
    description='Python library containing tools for simulating Cherenkov detector count rates in space.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Space Environment and Protection Group, University of Surrey',
    url='https://github.com/ssc-maire/SpaceCherenkovSimulator',
    keywords = 'space physics earth asymptotic trajectory Cherenkov detector Geant4 GRAS geomagnetic rigidity magnetocosmics trapped particles AE8 AP8',
    license='GNU General Public License v3.0',
    install_requires=['AsympDirsCalculator>=1.0.8',
                        'CosRayModifiedISO>=1.2.3',
                        'geopandas>=0.11.1',
                        'joblib>=1.2.0',
                        'matplotlib>=3.5.2',
                        'numpy>=1.23.1',
                        'pandas>=1.4.3',
                        'ParticleRigidityCalculationTools>=1.5.4',
                        'pyarrow>=12.0.0',
                        'scipy>=1.10.1',
                        'seaborn>=0.13.0',
                        'setuptools>=45.2.0',
                        'sketch>=0.4.2',
                        'spacepy>=0.4.1',
                        'tqdm>=4.30.0',
                        'uncertainties>=3.1.7',
                        'uproot>=5.1.2'],
    #install_requires,
    setup_requires=['pytest-runner','wheel'],
    tests_require=['pytest'],
)
