TagNN
======

[![Build Status](https://img.shields.io/badge/release-0.1-orange)](https://github.com/arcunique/TagNN)
[![Python 2.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-371/)

TagNN is a code developed to calculate the presssure-temperature profile of the atmosphere of a T-dwarf (Methane brown
dwarf) from the grid of pressure-temperature profiles of the T-dwarfs, calculated for different values of effective 
temperature (Teff), surface gravity (g) and metallicity ([Fe/H]). These grids have been developed by
[Marley 2000](http://articles.adsabs.harvard.edu/pdf/2000ASPC..212..152M) and used by
[Sengupta & Marley 2009](http://dx.doi.org/10.1088/0004-637X/707/1/716), etc. 

The relation between the pressure (P) and temperature (T) at different layers of the atmosphere is non-linear in nature 
and hence, interpolation from the available grids at the required values of Teff, g and [Fe/H] is prone to give 
inaccurate profiles. For this purpose, I have used a Neural Network based approach to train the relation. I do not train 
the relation between P and T directly, as the model could be unreliable. Instead, I have performed an amalgamation
between the scientific formalisms of the radiative transfer of EM waves through an atmopshere 
([Chandrasekhar 1960](https://ui.adsabs.harvard.edu/abs/1960ratr.book.....C)) and the deep learning of the non-linear 
relationships. I have used the non-linear relationship between the optical depth (\tau) and the temperature, for which I
have used the NN-based training and the resulting NN layer and node information map the optical depth to corresponding
temperature. From the optical depth, the pressure at each layer can be found out by using the hydrostatic equation and by 
adopting an average (wavelength-integrated) representation of the extinction coefficient, which is performed by this
code itself. Users have to select whether to consider the 'Rosseland' opacity or the 'Planck' opacity.


Author
------
* Aritra Chakrabarty (Indian Institute of Astrophysics, Bangalore)

Requirements
------------
* python>3.6
* numpy
* sklearn

Instructions on installation and use
------------------------------------
Presently, the code is only available on [Github](https://github.com/arcunique/TagNN). Either download the code or
use the following line on terminal to install using pip:\
pip install git+https://github.com/arcunique/TagNN   #installs from the current master on this repo.

You can import the modules and classes by:\
from TagNN import *

Documentation of this package is underway. An example program has been shown in the file run2learn.py.






