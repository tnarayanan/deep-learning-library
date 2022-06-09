from setuptools import setup, find_packages

setup(name='dll',
      version='0.0.1',
      install_requires=['numpy'],
      packages=['dll', 'dll.data', 'dll.layers', 'dll.optimizers'],
      include_package_data=True)
