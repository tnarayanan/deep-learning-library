from setuptools import setup, find_packages

setup(name='dll',
      version='0.0.1',
      author='Tejas Narayanan',
      description='A basic deep learning library written from scratch using Numpy',
      url='https://github.com/tnarayanan/deep-learning-library',
      install_requires=['numpy'],
      packages=find_packages())
