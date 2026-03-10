# Comparing IRENE Models — Standalone Reproducible Run

This directory contains everything needed to run the HEPI Jupyter notebooks from scratch. You must have **GRAS** (Geant4 Radiation Analysis for Space), **Geant4**, and **ROOT** installed; this code has been run with **Geant4 v10.7.4**, **GRAS 06-00-beta**, and **ROOT**.

## Before you run (required dependencies)

If you will run cells that execute GRAS simulations or read ROOT output, **set these environment variables first** (in the same shell/terminal you use to start Jupyter, or in your environment):

```bash
export GEANT4_SH=/path/to/your/geant4/install/bin/geant4.sh
export GRAS_ENV_SH=/path/to/your/gras/config/gras-env.sh
export ROOT_THISROOT_SH=/path/to/your/root/bin/thisroot.sh
```

Replace the paths with your actual Geant4, GRAS, and ROOT install locations. If any is unset when needed, the code will raise an error. See **How to run** below for full setup.

## Contents

- **Notebooks:** `comparing_IRENE_models.ipynb`, `comparing_IRENE_models_2.ipynb`, `comparing_IRENE_models_2_multi_rad.ipynb`, `comparing_IRENE_models_3.ipynb`
- **Python modules:** `spectra_running_tools.py`, `tools_for_Cherenkov_runs.py`, `Cherenkov_run_tuple.py`
- **Dependencies:** `requirements.txt` (all Python packages required by the notebooks and scripts)
- **Input data:** IRENE orbit and spectrum files, `BC_PDE.csv`, and (for notebook 2_multi_rad) GLE spectra
- **GRAS templates:** macro and geometry files used when running GRAS simulations

## How to run

1. **Use this directory as the working directory**  
   Start Jupyter (or your IDE) so that the kernel’s current working directory is this folder. All paths in the notebooks and code are relative to here.

2. **Create a virtual environment and install Python dependencies**  
   Create and activate a venv, then install from `requirements.txt`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   `requirements.txt` lists every package required by the notebooks and Python scripts (including CosRayModifiedISO, ParticleRigidityCalculationTools, AsympDirsCalculator, cartopy, sketch, etc.).

3. **GRAS, Geant4, and ROOT (required for running simulations)**  
   The notebooks need [GRAS](https://spacecraft.esa.int/projects/geant4-radiation-analysis-for-space-gras) (and its Geant4 dependency) and [ROOT](https://root.cern/). **There are no hardcoded paths:** you must set these environment variables before running cells that execute GRAS or read ROOT output:

   ```bash
   export GEANT4_SH=/path/to/your/geant4/install/bin/geant4.sh
   export GRAS_ENV_SH=/path/to/your/gras/config/gras-env.sh
   export ROOT_THISROOT_SH=/path/to/your/root/bin/thisroot.sh
   ```
   If any is unset when needed, the code will raise a clear error asking you to set it.

4. **Run the notebooks**  
   The notebooks do not depend on each other: each reads only the shared input data in this directory. You can run them in any order (or run any subset).

## Other paths

- **BC_PDE.csv** is looked up next to `Cherenkov_run_tuple.py` by default. To override: `export BC_PDE_CSV=/path/to/BC_PDE.csv`.
- **GDML schema**: The template GDML files reference `gdml.xsd` in the same directory. If your GRAS/Geant4 setup validates GDML and complains, obtain the schema from CERN and place it here:

  **Option A — CERN (recommended)**  
  Download the official GDML schema release from CERN, extract it, and copy the schema files into this directory:
  ```bash
  wget http://cern.ch/service-spi/app/releases/GDML/downloads/GDML_3_1_7.tar.gz
  tar -xzf GDML_3_1_7.tar.gz
  cp GDML_3_1_7/schema/*.xsd .
  ```
  The `gdml.xsd` file includes other `.xsd` modules (`gdml_core.xsd`, `gdml_define.xsd`, etc.), so all schema files must be in this folder.

  **Option B — From Geant4**  
  If Geant4 is installed, the schema is bundled in its source tree. Copy the schema directory:
  ```bash
  cp /path/to/geant4/source/persistency/gdml/schema/*.xsd .
  ```
  Adjust the path to match your Geant4 installation (the schema is typically under `geant4/source/persistency/gdml/schema`).

  **Option C — Jefferson Lab mirror**  
  The schema is also available from [JeffersonLab/gdml](https://github.com/JeffersonLab/gdml); clone the repo and copy the `.xsd` files into this directory.

  You can also ignore validation if the run succeeds without it.

## Caching

Completed GRAS runs are cached under `./Cherenkov_run_cache/` (joblib). Delete that directory to force all simulations to re-run from scratch.
