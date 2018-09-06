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
      author="Lanx Capital & Vinícius Calasans",
      author_email="calasans.vinicius@gmail.com",
      description="A library for quantitative finance using Time Series I/O (tsio) and QuantLib (QuantLib-Python).",
      license="GPL",
      keywords="quantitative finance MongoDB tsio QuantLib time series",
      url="",
      packages=['tsfin'],
      python_requires='>3.0',
      install_requires=['numpy>=1.14.2',
                        'pandas>=0.22.0',
                        'QuantLib-Python>=1.12',
                        'tsio',
                        ],
      long_description=read('README'),
      )
