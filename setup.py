import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='tsfin',
      version='0.0.1',
      authors=["Vinícius Calasans", "Pedro Coelho"],
      authors_emails=["calasans.vinicius@gmail.com", "coelhos1989@gmail.com"],
      description="A library for quantitative finance using Time Series I/O (tsio) and QuantLib (QuantLib-Python).",
      license="LGPL",
      keywords="quantitative finance MongoDB tsio QuantLib time series",
      url="https://github.com/LANXDEV/tsfin",
      packages=['tsfin'],
      python_requires='>3.7',
      install_requires=['numpy>=1.17.0',
                        'pandas>=0.25.0',
                        'scipy>=1.3.0',
                        'QuantLib>=1.19',
                        'tsio',
                        ],
      dependency_links=['https://github.com/LANXDEV/tsio'],
      long_description=read('README.md'),
      )
