# SpaceCherenkovSimulator

A Python-based tool to simulate the effect of interplanetary charged particles and the van Allen belts on count rates and pulse height distributions in simple Cherenkov detectors in space.

If you use this software for scientific research, please reference SpaceCherenkovSimulator as **C. S. W. Davis and F. Lei (2023). SpaceCherenkovSimulator version {version number}. https://github.com/ssc-maire/SpaceCherenkovSimulator , https://pypi.org/project/SpaceCherenkovSimulator/ . Surrey Space Center, University of Surrey.**

**N.B. Currently this tool only runs on Linux-based machines (if you are a Windows user, you must use [Windows Subsystem for Linux](https://ubuntu.com/wsl) or a virtual machine with Linux to install and run this tool)**

**To use this tool, you must first ensure you have MAGNETOCOSMICS and GRAS installed, as this software acts as a tool to run both of these Geant4-based tools**

This software is relatively complete from a programmatic perspective (although currently only containing relatively simple fused silica Cherenkov radiator geometries), but a relatively easy-to-use user interface and documentation has not yet been created. These should be added over time. For now, I have provided several Jupyter notebooks, `testing_simulator.ipynb` and the more complex notebooks in `rough_example_notebooks`, which provide some examples of how this software can be used. Note that the examples in `rough_example_notebooks` are for an older version of the software, and therefore do not directly work with this version - although the general logic for the calculations remains intact.

Feel free to use and modify/contribute to this software if you wish, and feel free to contact me if you have any questions or run into issues. 

## Installation

To install SpaceCherenkovSimulator, simply run

```
sudo pip install SpaceCherenkovSimulator
```

