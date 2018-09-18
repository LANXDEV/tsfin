
Time Series Finance Documentation
=================================

Description
-----------

This is the documentation page of the Time Series Finance project. For installation instructions, see the `project's
GitHub page <https://github.com/LANXDEV/tsfin>`_.

This project is a Python library implementing models for financial instruments and curves. It is built on top of
`Time Series I/O <https://github.com/LANXDEV/tsio>`_ and `QuantLib (for Python) <https://www.quantlib.org/>`_.

Instruments
-----------
.. toctree::
   :maxdepth: 1

   tsfin/tsfin.base.instrument
   tsfin/tsfin.instruments.bonds.fixedratebond
   tsfin/tsfin.instruments.bonds.callablefixedratebond
   tsfin/tsfin.instruments.bonds.floatingratebond
   tsfin/tsfin.instruments.currencyfuture
   tsfin/tsfin.instruments.depositrate
   tsfin/tsfin.instruments.fraddi
   tsfin/tsfin.instruments.cupomcambial
   tsfin/tsfin.instruments.ois
   tsfin/tsfin.instruments.swaprate


Curves
------
.. toctree::
   :maxdepth: 1

   tsfin/tsfin.curves.yieldcurve
   tsfin/tsfin.curves.currencycurve
   tsfin/tsfin.curves.hybridyieldcurve


Indices and Tables:
-------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
