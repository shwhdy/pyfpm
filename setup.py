from setuptools import setup, find_packages
import os, sys
import subprocess

# Define version
__version__ = 0.01

setup( name             = 'pyfpm'
     , version          = __version__
     , description      = 'Python implementation of Fourier Ptychography algorithm'
     , license          = 'BSD'
     , packages         = find_packages()
     , include_package_data = True
     , install_requires = ['llops', 'numpy', 'matplotlib']
     )
