#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import os
import re
from glob import glob

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

ext_macros = []
if os.environ.get('CYTHON_TRACE_NOGIL') == '1':
    ext_macros.append(('CYTHON_TRACE_NOGIL', 1))


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


def read_version():
    path = os.path.join('src', 'cbytesparse', '__init__.py')
    with open(path, 'rt') as file:
        for line in file:
            if line.startswith('__version__'):
                return eval(line.split('=')[1])
    raise ValueError(f'cannot find __version__ inside of {path}')


setup(
    name='cbytesparse',
    version=read_version(),
    license='BSD 2-Clause License',
    description='Library to handle sparse bytes within a virtual memory space',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S)
            .sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    long_description_content_type='text/x-rst',
    author='Andrea Zoppi',
    author_email='texzk@email.it',
    url='https://github.com/TexZK/cbytesparse',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob('src/*.py')],
    ext_modules=[Extension('cbytesparse.c', ['src/cbytesparse/c.c'], define_macros=ext_macros)],
    include_dirs=['.'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Topic :: Software Development',
        'Topic :: Software Development :: Embedded Systems',
        'Topic :: Utilities',
    ],
    keywords=[
    ],
    install_requires=[
        'bytesparse>=0.0.5',  # comment out and use the one below for development instead
        # 'bytesparse @ git+https://github.com/TexZK/bytesparse.git',  # only for development
    ],
    extras_require={
        'testing': [
            'Cython',
            'pytest',
        ],
        'scripts': [
            'Cython',
        ],
    },
)
