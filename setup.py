from setuptools import setup

setup(
   name='sbiwrapper',
   version='0.1',
   description='The is a wrapper for the sbi library allowing for simulation based estimation of DSGE models',
   author='Cameron Fen',
   author_email='cameronfen@gmail.com',
   packages=['sbiwrapper'], 
   install_requires=['sbi', 'torch', 'pyknos', 'nflows', 'numpy'], #external packages as dependencies
)
