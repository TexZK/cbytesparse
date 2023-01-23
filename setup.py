#!/usr/bin/env python

if __name__ == '__main__':
    import os

    import setuptools

    ext_macros = []
    if os.environ.get('CYTHON_TRACE_NOGIL') == '1':
        ext_macros.append(('CYTHON_TRACE_NOGIL', 1))

    ext_modules = [
        setuptools.Extension('cbytesparse.c', ['src/cbytesparse/c.c'], define_macros=ext_macros),
    ]

    setuptools.setup(
        ext_modules=ext_modules,
    )
