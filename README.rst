********
Overview
********

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |gh_actions|
        | |codecov|
    * - package
      - | |version| |wheel|
        | |supported-versions|
        | |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/cbytesparse/badge/?style=flat
    :target: https://readthedocs.org/projects/cbytesparse
    :alt: Documentation Status

.. |gh_actions| image:: https://github.com/TexZK/cbytesparse/workflows/CI/badge.svg
    :alt: GitHub Actions Status
    :target: https://github.com/TexZK/cbytesparse

.. |codecov| image:: https://codecov.io/gh/TexZK/cbytesparse/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/TexZK/cbytesparse

.. |version| image:: https://img.shields.io/pypi/v/cbytesparse.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/cbytesparse/

.. |wheel| image:: https://img.shields.io/pypi/wheel/cbytesparse.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/cbytesparse/

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/cbytesparse.svg
    :alt: Supported versions
    :target: https://pypi.org/project/cbytesparse/

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/cbytesparse.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/cbytesparse/


.. end-badges

Library to handle sparse bytes within a virtual memory space.

* Free software: BSD 2-Clause License


Objectives
==========

This library aims to provide utilities to work with a `virtual memory`, which
constsis in a virtual addressing space where sparse `chunks` of data can be
stored.

In order to be easy to use, its interface should be close to that of a
``bytearray``, which is the closest pythonic way to store dynamic data.
The main downside of a ``bytearray`` is that it requires a contiguous data
allocation starting from address 0. This is not good when sparse data have to
be stored, such as when emulating the addressing space of a generic
microcontroller.

The main idea is to provide a ``bytearray``-like class with the possibility to
internally hold the sparse `blocks` of data.
A `block` is ideally a tuple ``(start, data)`` where `start` is the start
address and `data` is the container of data items (e.g. ``bytearray``).
The length of the block is ``len(data)``.
Those blocks are usually not overlapping nor contiguous, and sorted by start
address.


Python implementation
=====================

This library is the Cython complement to the Python implementation provided by
the ``bytesparse`` Python package.
Please refer to its own documentation for more details.


Cython implementation
=====================

The library provides an experimental `Cython` implementation. It tries to
mimic the same algorithms of the Python implementation, while exploiting the
speedup of compiled `C` code.

Beware that the Cython implementation is meant to be potentially faster than
the pure Python one, but there might be even faster `ad-hoc` implementations
of virtual memory highly optimized for the underlying hardware.

The addressing space is limited to that of an ``uint_fast64_t`` (typically
32-bit or 64-bit as per the hosting machine), so it is not possible to have
an infinite addressing space, nor negative addresses.
To keep the implementation code simple enough, the highest address (e.g.
``0xFFFFFFFF`` on a 32-bit machine) is reserved.

Block data chunks cannot be greater than the maximum ``ssize_t`` value
(typically half of the addressing space).

The Cython implementation is optional, and potentially useful only when the
Python implementation seems too slow for the user's algorithms, within the
limits stated above.

If in doubt about using the Cython implementation, just stick with the Python
one, which is much easier to integrate and debug.

More details can be found within ``cbytesparse.c``.


Documentation
=============

For the full documentation, please refer to:

https://cbytesparse.readthedocs.io/


Installation
============

From PyPI (might not be the latest version found on *github*):

.. code-block:: sh

    $ pip install cbytesparse

From the source code root directory:

.. code-block:: sh

    $ pip install .


Development
===========

To run the all the tests:

.. code-block:: sh

    $ pip install tox
    $ tox


To regenerate the Cython files manually, run the following commands:

.. code-block:: sh

    $ python scripts/cython_build_src.py
    $ python scripts/cython_build_tests.py

or alternatively:

.. code-block:: sh

	$ tox -e cythonize
