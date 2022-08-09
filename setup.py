#!/usr/bin/env python
from pathlib import Path

from setuptools import setup
from setuptools import find_namespace_packages

import toml

# Major version number.
# We can bump this to indicate big breaking changes to an application
major_version=0

# Main requirements
requires = [
    'pandas',
    'scikit-learn',
]

test_requires = [
    'pytest >= 5.3.5',
    'pylint',
    'mypy'
]

# All additional requirements needed for development
dev_requires = list(set(test_requires + [
    'tox',
    'tox-auto-recreate >= 1.0.1'
]))

project_dir = Path(__file__).absolute().parent

# with project_dir.joinpath('pyproject.toml').open('r') as handle:
#     pyproject_toml = toml.load(handle)

setup(
    name='cocktail-playlist',
    version='0.1',
    # description='Seasonal Cocktail Playlist',
    author='Jevon Beckles',
    author_email='jevonbeck@yahoo.com',
    python_requires='>=3.8',
    install_requires=requires,
    extras_require={
        'dev': dev_requires,
        'test': test_requires,
    },
    # packages=find_namespace_packages(where='src'),
    # package_dir={'': 'src'},
    data_files=[('', ['data/mr-boston-flattened.csv'])],
)
