#!/usr/bin/python3
import os

from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults

compiler_directives = get_directive_defaults()

if os.environ.get('CYTHON_TRACE_NOGIL') == '1':
	compiler_directives['linetrace'] = True
	compiler_directives['binding'] = True

pyx_path = os.path.join('tests', '_test_c.pyx')

cythonize(
    pyx_path,
    include_path=['src'],
    force=True,
    annotate=True,
    compiler_directives=compiler_directives,
)
