from setuptools import setup
import TagNN

setup(name='TagNN',
      version=TagNN.__version__,
      description='T-dwarf Atmospheric Grid using Neural Network: A code to calculate the pressure-temperature grid of the atmospheres of the T-type brown'
                  ' dwarfs using Neural Networks.',
      long_description=open('README.md').read(),
      classifiers=['Development Status :: 0 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.7'],
      url='https://github.com/arcunique/tagNN',
      author='Aritra Chakrabarty',
      author_email='aritra@iiap.res.in',
      install_requires=['numpy', 'sklearn'],
      zip_safe=False)
