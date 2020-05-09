## PyRossGeo: Spatially resolved infectious disease models in Python 

[About](#about) | [Installation](#installation) | [Documentation](#documentation)  | [Examples](#examples) | [Publications](#publications) | [License](#license) |  [Wiki](https://github.com/lukastk/PyRossGeo/wiki) |  [Contact](#contact)

![Imagel](docs/banner.jpg)

## About

[PyRossGeo](https://github.com/lukastk/PyRossGeo) is a numerical library for spatially resolved mathematical modelling of infectious disease in Python.

Note that PyRossGeo has a Python interface, but runs at C-speeds using Cython.

Add a few lines about link to PyRoss

The authors are part of the [Rapid Assistance in Modelling the Pandemic (RAMP)](https://royalsociety.org/news/2020/03/urgent-call-epidemic-modelling/) taskforce at the **University of Cambridge**. In alphabetical order, we are:
[Jakub Dolezal](https://github.com/JakubJDolezal),
[Tim Ekeh](https://github.com/tekeh),
[Lukas Kikuchi](https://github.com/lukastk),
[Hideki Kobayashi](https://github.com/hidekb),
[Paul Rohrbach](https://github.com/prohrbach),
[Rajesh Singh](https://github.com/rajeshrinet) and
[Fernando Pedrero](https://github.com/Ferfer93).

## Installation
Clone (or download) the repository and use a terminal to install using

```bash
>> git clone https://github.com/lukastk/PyRossGeo.git
>> cd pyross
>> python setup.py install
```

PyRossGeo requires the following software

- Python 3.4+
- [Cython 0.25.x+](http://docs.cython.org/en/latest/index.html)
| [NumPy 1.x+](http://www.numpy.org)
| [Pandas](https://pandas.pydata.org/)
| [SciPy 1.1.x+](https://www.scipy.org/)
- Optional dependencies:
| [Zarr](https://zarr.readthedocs.io/) (Saving simulations results)
| [Matplotlib 2.0.x+](https://matplotlib.org) (Notebooks)
| [Jupyter](https://jupyter.org/) (Notebooks)
| [PyTest](https://docs.pytest.org/) (Testing)
| [GeoPandas](https://geopandas.org/) (Visualisations)

## Documentation

## Examples

## Publications

## License

We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598).

## Contact

For inquiries about the code or the model, please contact [Lukas Kikuchi](ltk26@cam.ac.uk).