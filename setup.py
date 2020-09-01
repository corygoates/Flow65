"""Flow65: A potential flow library"""

from setuptools import setup
import os
import sys

setup(name = 'Flow65',
    version = '1.0.0',
    description = "Flow65: a potential flow library",
    url = 'https://github.com/corygoates/Flow65',
    author = 'usuaero',
    author_email = 'cory.goates@aggiemail.usu.edu',
    install_requires = ['numpy>=1.18', 'scipy>=1.4', 'pytest', 'matplotlib'],
    python_requires ='>=3.6.0',
    license = 'MIT',
    packages = ['flow65'],
    zip_safe = False)
