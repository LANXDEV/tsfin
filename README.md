# Time Series Finance

This project implements classes for financial instruments and curves. It is built on top of
[Time Series I/O](https://github.com/LANXDEV/tsio) and [QuantLib-Python](https://www.quantlib.org/). The objects in this
project are all based on the Time Series model, in the sense of the [Time Series I/O project](https://github.com/LANXDEV/tsio),
and provide similar features in addition to standard financial calculations. The classes and tools defined here might be
of great help in building financial applications.

Check the project's [Documentation](https://lanxdev.github.io/tsfin/index).


## Requirements
- A running MongoDB instance

Also see the Dependencies section below.

## Installation

```sh
pip install git+https://github.com/LANXDEV/tsfin/
```

## Dependencies

- [NumPy](https://www.numpy.org): 1.16.0 or higher
- [pandas](https://pandas.pydata.org/): 0.24.2 or higher
- [QuantLib-Python](https://www.quantlib.org/install/windows-python.shtml): 1.16 or higher
- [Time Series I/O](https://github.com/LANXDEV/tsio)


## Authors

* **Vinícius Calasans** - [vcalasans](https://github.com/vcalasans)


## Acknowledgements

Time Series Finance has been developed at [Lanx Capital Investimentos](https://www.lanxcapital.com/) since 2016, serving
as base for our internal frameworks. It wouldn't be possible without the support and insight provided by our team of
analysts:

- Tulio Ribeiro
- Eduardo Thiele
- Pedro Coelho
- Humberto Nardiello


## License

This project is licensed under the GNU LGPL v3. See [Copying](COPYING) and [Copying.Lesser](COPYING.LESSER) for details.

