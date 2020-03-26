#!/usr/bin/env python
"""Setup file."""

from setuptools import setup, find_packages


with open("requirements.txt", 'rt') as f:
    dependencies = f.readlines()
    dependencies = [dep.strip("\n") for dep in dependencies]
setup(
    name='database_creation',
    version='0.1.0',
    description='Create a dataframe for audio classification',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=dependencies
    )
