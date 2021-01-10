from setuptools import setup

setup(
    name='som_anomaly_detector',
    version='0.1',
    description=('This package provides a simple and (not so) efficient implementation of '
                 'Self Organizing Maps for anomaly detection purposes.'),
    url='https://github.com/FlorisHoogenboom/som-anomaly-detector',
    author='Floris Hoogenboom',
    author_email='floris@digitaldreamworks.nl',
    license='MIT',
    packages=['som_anomaly_detector'],
    install_requires=[
      'scipy',
      'numpy',
      'scikit-learn'
    ],
    zip_safe=False
)