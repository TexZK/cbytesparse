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


@pytest.fixture()
def bytestr():
    return bytes(range(256))


@pytest.fixture
def loremstr():
    return (b'Lorem ipsum dolor sit amet, consectetur adipisici elit, sed '
            b'eiusmod tempor incidunt ut labore et dolore magna aliqua.')


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
            assert instance.count(hexstr[i:(i + 1)]) == 1

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

        with pytest.raises(TypeError, match='must not be None'):
            instance.startswith(None)

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

        with pytest.raises(TypeError, match='must not be None'):
            instance.endswith(None)

    def test___contains__(self, hexview, hexstr):
        instance = InplaceView(hexview)

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert hexview[start:endex] in instance

        assert hexstr in instance
        assert hexview in instance

        assert hexview[0:0] not in instance
        assert b'' not in instance
        assert (hexstr + b'\0') not in instance

        with pytest.raises(TypeError, match='must not be None'):
            assert None not in instance

    def test_contains(self, hexview, hexstr):
        instance = InplaceView(hexview)

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert instance.contains(hexview[start:endex]) is True
                assert instance.contains(hexview[start:endex], start=start, endex=endex) is True
                assert instance.contains(hexview[start:endex], start=start, endex=(endex + 1)) is True
                assert instance.contains(hexview[start:endex], start=(start + 1), endex=endex) is False
                if start:
                    assert instance.contains(hexview[start:endex], start=(start - 1), endex=endex) is True
                    assert instance.contains(hexview[start:endex], start=(start - 1), endex=(endex + 1)) is True
                if endex:
                    assert instance.contains(hexview[start:endex], start=start, endex=(endex - 1)) is False
                    assert instance.contains(hexview[start:endex], start=(start + 1), endex=(endex - 1)) is False

        assert instance.contains(hexstr) is True
        assert instance.contains(hexview) is True

        assert instance.contains(hexview[0:0]) is False
        assert instance.contains(b'') is False
        assert instance.contains(hexstr + b'\0') is False

        with pytest.raises(TypeError, match='must not be None'):
            instance.contains(None)

    def test_find(self, hexview, hexstr):
        instance = InplaceView(hexview)

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert instance.find(hexview[start:endex]) == start
                assert instance.find(hexview[start:endex], start=start, endex=endex) == start
                assert instance.find(hexview[start:endex], start=start, endex=(endex + 1)) == start
                assert instance.find(hexview[start:endex], start=(start + 1), endex=endex) < 0
                if start:
                    assert instance.find(hexview[start:endex], start=(start - 1), endex=endex) == start
                    assert instance.find(hexview[start:endex], start=(start - 1), endex=(endex + 1)) == start
                if endex:
                    assert instance.find(hexview[start:endex], start=start, endex=(endex - 1)) < 0
                    assert instance.find(hexview[start:endex], start=(start + 1), endex=(endex - 1)) < 0

        assert instance.find(hexstr) == 0
        assert instance.find(hexview) == 0

        assert instance.find(hexview[0:0]) < 0
        assert instance.find(b'') < 0
        assert instance.find(hexstr + b'\0') < 0

        with pytest.raises(TypeError, match='must not be None'):
            instance.find(None)

    def test_find_multi(self):
        buffer = b'Hello, World!'
        instance = InplaceView(memoryview(buffer))
        chars = list(sorted(bytes([c]) for c in set(buffer)))
        for c in chars:
            assert instance.find(c) == buffer.find(c)

    def test_rfind(self, hexview, hexstr):
        instance = InplaceView(hexview)

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert instance.rfind(hexview[start:endex]) == start
                assert instance.rfind(hexview[start:endex], start=start, endex=endex) == start
                assert instance.rfind(hexview[start:endex], start=start, endex=(endex + 1)) == start
                assert instance.rfind(hexview[start:endex], start=(start + 1), endex=endex) < 0
                if start:
                    assert instance.rfind(hexview[start:endex], start=(start - 1), endex=endex) == start
                    assert instance.rfind(hexview[start:endex], start=(start - 1), endex=(endex + 1)) == start
                if endex:
                    assert instance.rfind(hexview[start:endex], start=start, endex=(endex - 1)) < 0
                    assert instance.rfind(hexview[start:endex], start=(start + 1), endex=(endex - 1)) < 0

        assert instance.rfind(hexstr) == 0
        assert instance.rfind(hexview) == 0

        assert instance.rfind(hexview[0:0]) < 0
        assert instance.rfind(b'') < 0
        assert instance.rfind(hexstr + b'\0') < 0

        with pytest.raises(TypeError, match='must not be None'):
            instance.rfind(None)

    def test_rfind_multi(self):
        buffer = b'Hello, World!'
        instance = InplaceView(memoryview(buffer))
        chars = list(sorted(bytes([c]) for c in set(buffer)))
        for c in chars:
            assert instance.rfind(c) == buffer.rfind(c)

    def test_index(self, hexview, hexstr):
        instance = InplaceView(hexview)

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert instance.index(hexview[start:endex]) == start
                assert instance.index(hexview[start:endex], start=start, endex=endex) == start
                assert instance.index(hexview[start:endex], start=start, endex=(endex + 1)) == start
                with pytest.raises(ValueError, match='subsection not found'):
                    assert instance.index(hexview[start:endex], start=(start + 1), endex=endex)
                if start:
                    assert instance.index(hexview[start:endex], start=(start - 1), endex=endex) == start
                    assert instance.index(hexview[start:endex], start=(start - 1), endex=(endex + 1)) == start
                if endex:
                    with pytest.raises(ValueError, match='subsection not found'):
                        assert instance.index(hexview[start:endex], start=start, endex=(endex - 1))
                    with pytest.raises(ValueError, match='subsection not found'):
                        assert instance.index(hexview[start:endex], start=(start + 1), endex=(endex - 1))

        assert instance.index(hexstr) == 0
        assert instance.index(hexview) == 0

        with pytest.raises(ValueError, match='subsection not found'):
            assert instance.index(hexview[0:0])
        with pytest.raises(ValueError, match='subsection not found'):
            assert instance.index(b'')
        with pytest.raises(ValueError, match='subsection not found'):
            assert instance.index(hexstr + b'\0')

        with pytest.raises(TypeError, match='must not be None'):
            instance.index(None)

    def test_index_multi(self):
        buffer = b'Hello, World!'
        instance = InplaceView(memoryview(buffer))
        chars = list(sorted(bytes([c]) for c in set(buffer)))
        for c in chars:
            assert instance.index(c) == buffer.index(c)

    def test_rindex(self, hexview, hexstr):
        instance = InplaceView(hexview)

        for start in range(len(hexview) - 1):
            for endex in range(start + 1, len(hexview)):
                assert instance.rindex(hexview[start:endex]) == start
                assert instance.rindex(hexview[start:endex], start=start, endex=endex) == start
                assert instance.rindex(hexview[start:endex], start=start, endex=(endex + 1)) == start
                with pytest.raises(ValueError, match='subsection not found'):
                    assert instance.rindex(hexview[start:endex], start=(start + 1), endex=endex)
                if start:
                    assert instance.rindex(hexview[start:endex], start=(start - 1), endex=endex) == start
                    assert instance.rindex(hexview[start:endex], start=(start - 1), endex=(endex + 1)) == start
                if endex:
                    with pytest.raises(ValueError, match='subsection not found'):
                        assert instance.rindex(hexview[start:endex], start=start, endex=(endex - 1))
                    with pytest.raises(ValueError, match='subsection not found'):
                        assert instance.rindex(hexview[start:endex], start=(start + 1), endex=(endex - 1))

        assert instance.rindex(hexstr) == 0
        assert instance.rindex(hexview) == 0

        with pytest.raises(ValueError, match='subsection not found'):
            assert instance.rindex(hexview[0:0])
        with pytest.raises(ValueError, match='subsection not found'):
            assert instance.rindex(b'')
        with pytest.raises(ValueError, match='subsection not found'):
            assert instance.rindex(hexstr + b'\0')

        with pytest.raises(TypeError, match='must not be None'):
            instance.rindex(None)

    def test_rindex_multi(self):
        buffer = b'Hello, World!'
        instance = InplaceView(memoryview(buffer))
        chars = list(sorted(bytes([c]) for c in set(buffer)))
        for c in chars:
            assert instance.rindex(c) == buffer.rindex(c)

    def test_replace(self, hexstr):
        hexview = memoryview(hexstr)
        instance = InplaceView(hexview)
        negbytes = bytes(255 - c for c in hexstr)
        negview = memoryview(negbytes)
        for i in range(len(hexview)):
            instance.replace(hexview[i:(i + 1)], negview[i:(i + 1)])
        assert hexview == negbytes

    def test_isalnum(self):
        instance = InplaceView(b'H3ll0W0rld')
        assert instance.isalnum() is True

        instance = InplaceView(b'H3ll0W0rld!')
        assert instance.isalnum() is False

        instance = InplaceView(b'')
        assert instance.isalnum() is False

    def test_isalpha(self):
        instance = InplaceView(b'HelloWorld')
        assert instance.isalpha() is True

        instance = InplaceView(b'H3ll0W0rld')
        assert instance.isalpha() is False

        instance = InplaceView(b'')
        assert instance.isalpha() is False

    def test_isascii(self):
        instance = InplaceView(bytes(range(128)))
        assert instance.isascii() is True

        instance = InplaceView(bytes(range(129)))
        assert instance.isascii() is False

        instance = InplaceView(b'')
        assert instance.isascii() is False

    def test_isdigit(self, hexview):
        instance = InplaceView(hexview[:10])
        assert instance.isdigit() is True

        instance = InplaceView(hexview)
        assert instance.isdigit() is False

        instance = InplaceView(b'')
        assert instance.isdigit() is False

    def test_islower(self, hexstr):
        instance = InplaceView(hexstr.lower())
        assert instance.islower() is True

        instance = InplaceView(hexstr.upper())
        assert instance.islower() is False

        instance = InplaceView(b'')
        assert instance.islower() is False

    def test_isupper(self, hexstr):
        instance = InplaceView(hexstr.upper())
        assert instance.isupper() is True

        instance = InplaceView(hexstr.lower())
        assert instance.isupper() is False

        instance = InplaceView(b'')
        assert instance.isupper() is False

    def test_isspace(self):
        instance = InplaceView(b'\x09\x0A\x0B\x0C\x0D\x1C\x1D\x1E\x1F\x20')
        assert instance.isspace() is True

        instance = InplaceView(b'\x09\x0A\x0B\x0C\x0D\x1C\x1D\x1E\x1F\x20\x00')
        assert instance.isspace() is False

        instance = InplaceView(b'')
        assert instance.isspace() is False

    def test_istitle(self, loremstr):
        instance = InplaceView(loremstr.title())
        assert instance.istitle() is True

        instance = InplaceView(loremstr.capitalize())
        assert instance.istitle() is False

        instance = InplaceView(b'')
        assert instance.istitle() is False

    def test_lower(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.lower() == bytestr.lower()
        assert instance.islower() is True

        buffer = loremstr.upper()
        instance = InplaceView(bytearray(buffer))
        assert instance.lower() == buffer.lower()
        assert instance.islower() is True

        instance = InplaceView(bytearray())
        assert instance.lower() == b''.lower()
        assert instance.islower() is False

    def test_upper(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.upper() == bytestr.upper()
        assert instance.isupper() is True

        buffer = loremstr.lower()
        instance = InplaceView(bytearray(buffer))
        assert instance.upper() == buffer.upper()
        assert instance.isupper() is True

        instance = InplaceView(bytearray())
        assert instance.upper() == b''.upper()
        assert instance.isupper() is False

    def test_swapcase(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.swapcase() == bytestr.swapcase()

        buffer = loremstr.title()
        instance = InplaceView(bytearray(buffer))
        assert instance.swapcase() == buffer.swapcase()

    def test_capitalize(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.capitalize() == bytestr.capitalize()

        buffer = loremstr.lower()
        instance = InplaceView(bytearray(buffer))
        assert instance.capitalize() == buffer.capitalize()

    def test_title(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.title() == bytestr.title()
        assert instance.istitle() is True

        buffer = loremstr.title().swapcase()
        instance = InplaceView(bytearray(buffer))
        assert instance.title() == buffer.title()
        assert instance.istitle() is True

        instance = InplaceView(bytearray())
        assert instance.title() == b''.title()
        assert instance.istitle() is False

    def test_maketrans(self, bytestr):
        table = InplaceView.maketrans(b'', b'')
        assert table == bytestr

        table = InplaceView.maketrans(bytestr, bytestr)
        assert table == bytestr

        revbytes = bytes(reversed(bytestr))
        table = InplaceView.maketrans(bytestr, revbytes)
        assert table == revbytes

        table = InplaceView.maketrans(revbytes, bytestr)
        assert table == revbytes

        with pytest.raises(ValueError, match='different sizes'):
            InplaceView.maketrans(b'', bytestr)

        with pytest.raises(ValueError, match='different sizes'):
            InplaceView.maketrans(bytestr, b'')

    def test_translate(self, bytestr):
        table = InplaceView.maketrans(b'', b'')
        instance = InplaceView(bytearray(bytestr))
        instance.translate(table)
        assert instance.wrapped == bytestr

        table = InplaceView.maketrans(bytestr, bytestr)
        instance = InplaceView(bytearray(bytestr))
        instance.translate(table)
        assert instance.wrapped == bytestr

        revbytes = bytes(reversed(bytestr))
        table = InplaceView.maketrans(bytestr, revbytes)
        instance = InplaceView(bytearray(bytestr))
        instance.translate(table)
        assert instance.wrapped == revbytes

        table = InplaceView.maketrans(revbytes, bytestr)
        instance = InplaceView(bytearray(revbytes))
        instance.translate(table)
        assert instance.wrapped == bytestr

        with pytest.raises(ValueError, match='translation table must be 256 characters long'):
            instance = InplaceView(bytearray(bytestr))
            instance.translate(table[:-1])

    def test_readonly(self, bytestr, hexstr, hexview):
        instance = InplaceView(None)
        assert instance.readonly is True

        instance = InplaceView(bytestr)
        assert instance.readonly is True

        instance = InplaceView(hexstr)
        assert instance.readonly is False

        instance = InplaceView(hexview)
        assert instance.readonly is False

    def test_wrapped(self, bytestr, hexstr, hexview):
        instance = InplaceView(None)
        assert instance.wrapped is None
        instance.release()
        assert instance.wrapped is None

        instance = InplaceView(bytestr)
        assert instance.wrapped is bytestr
        instance.release()
        assert instance.wrapped is None

        instance = InplaceView(hexstr)
        assert instance.wrapped is hexstr
        instance.release()
        assert instance.wrapped is None

        instance = InplaceView(hexview)
        assert instance.wrapped is hexview
        instance.release()
        assert instance.wrapped is None


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
