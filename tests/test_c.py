# Copyright (c) 2020-2023, Andrea Zoppi.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import importlib
import inspect
import sys
from typing import Type

import pytest

from _common import *

# noinspection PyUnresolvedReferences
from cbytesparse.c import InplaceView  # isort:skip
# noinspection PyUnresolvedReferences
from cbytesparse.c import Memory as _Memory  # isort:skip
# noinspection PyUnresolvedReferences
from cbytesparse.c import bytesparse as _bytesparse  # isort:skip


# Patch inspect.isfunction() to allow Cython functions to be discovered
def _patch_inspect_isfunction():  # pragma: no cover
    isfunction_ = inspect.isfunction

    def isfunction(obj):
        return (isfunction_(obj)
                or type(obj).__name__ == 'cython_function_or_method')

    isfunction.isfunction_ = isfunction_
    inspect.isfunction = isfunction


_patch_inspect_isfunction()


def _load_cython_tests():  # pragma: no cover
    # List of Cython modules containing tests
    cython_test_modules = ['_test_c']

    for mod in cython_test_modules:
        try:
            # For each callable in `mod` with name `test_*`,
            # set the result as an attribute of this module.
            mod = importlib.import_module(mod)

            for name in dir(mod):
                item = getattr(mod, name)

                if callable(item) and name.startswith('test_'):
                    setattr(sys.modules[__name__], name, item)

        except ImportError:
            pass


_load_cython_tests()


@pytest.fixture
def hexstr():
    return bytearray(b'0123456789ABCDEF')


@pytest.fixture
def hexview(hexstr):
    return memoryview(hexstr)


class TestInplaceView:

    def test___init__(self, hexview):
        instance = InplaceView(hexview)
        assert instance.wrapped is hexview

        InplaceView(None)  # FIXME TODO

    def test___sizeof__(self, hexview):
        instance = InplaceView(hexview)
        assert instance.__sizeof__() > 0

    def test_count(self, hexview, hexstr):
        instance = InplaceView(hexview)
        for i in range(len(hexstr)):
            assert instance.count(hexstr[i:(i + 1)]) == 1, i

        view = memoryview(bytearray(10))
        instance = InplaceView(view)
        assert instance.count(b'\0') == 10
        assert instance.count(b'\0' * 5) == 2
        assert instance.count(b'\0' * 3) == 3
        assert instance.count(b'\0' * 10) == 1
        for start in range(10):
            for endex in range(start, 10):
                assert instance.count(b'\0', start=start, endex=endex) == endex - start

        view = memoryview(bytearray(b'Hello, World!'))
        instance = InplaceView(view)
        assert instance.count(b'l') == 3
        assert instance.count(b'll') == 1
        assert instance.count(b'o') == 2
        assert instance.count(b'World') == 1

        with pytest.raises(TypeError, match='must not be None'):
            instance.count(None)

    def test_release(self, hexview):
        instance = InplaceView(hexview)
        assert instance.wrapped is hexview
        instance.release()
        assert instance.wrapped is None
        instance.release()
        assert instance.wrapped is None

    def test_startswith(self, hexview, hexstr):
        instance = InplaceView(hexview)
        assert instance.startswith(hexstr) is True
        assert instance.startswith(hexstr + b'\0') is False

        for endex in range(1, len(hexview)):
            assert instance.startswith(hexview[:endex]) is True

        for start in range(1, len(hexview)):
            assert instance.startswith(hexview[start:]) is False

        for start in range(1, len(hexview)):
            for endex in range(start, len(hexview)):
                assert instance.startswith(hexview[start:endex]) is False

        zeros = bytes(len(hexview) * 2)
        zeroview = memoryview(zeros)
        for i in range(len(zeroview)):
            assert instance.startswith(zeroview[:i]) is False

    def test_endswith(self, hexview, hexstr):
        instance = InplaceView(hexview)
        assert instance.endswith(hexstr) is True
        assert instance.endswith(b'\0' + hexstr) is False

        for endex in range(len(hexview) - 1):
            assert instance.endswith(hexview[:endex]) is False

        for start in range(len(hexview) - 1):
            assert instance.endswith(hexview[start:]) is True

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert instance.endswith(hexview[start:endex]) is False

        zeros = bytes(len(hexview) * 2)
        zeroview = memoryview(zeros)
        for i in range(len(zeroview)):
            assert instance.endswith(zeroview[:i]) is False


class TestMemory(BaseMemorySuite):
    Memory: Type['_Memory'] = _Memory
    ADDR_NEG: bool = False

    def test___sizeof__(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
        assert memory.__sizeof__() > 0
        assert memory.__sizeof__() > 6


class TestBytesparse(BaseBytearraySuite, BaseMemorySuite):
    bytesparse: Type['_bytesparse'] = _bytesparse

    # Reuse some of BaseMemorySuite methods
    Memory: Type['_Memory'] = _bytesparse
    ADDR_NEG: bool = False
