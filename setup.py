"""Installation script for the 'wbc_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.5.2",
    # rsl-rl-lib (v5.4.1+) with AMP support must be installed separately:
    #   pip install git+https://github.com/chifongip/rsl_rl.git@main
]

# Installation operation
setup(
    name="wbc_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
