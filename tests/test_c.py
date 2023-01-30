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

import array
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

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None


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

    def test___delitem__(self, hexview):
        with pytest.raises(IndexError, match='cannot resize view'):
            instance = InplaceView(hexview)
            del instance[0]

    def test___getitem__(self, hexview):
        instance = InplaceView(hexview)
        size = len(hexview)

        for i in range(size):
            assert instance[i] == hexview[i]

        for start in range(-size, size):
            for endex in range(-size, size):
                assert instance[start:endex] == hexview[start:endex]

    def test___init__(self, hexview):
        assert InplaceView(hexview).obj is hexview
        assert InplaceView(None).obj is None

        b = b'Hello, World!'
        assert InplaceView(b).obj is b

        a = array.array('B')
        assert InplaceView(a).obj is a

        with pytest.raises(ValueError, match='Buffer dtype mismatch'):
            InplaceView(array.array('H'))

        with pytest.raises(ValueError, match='Buffer dtype mismatch'):
            InplaceView(array.array('L'))

        if numpy is not None:  # pragma: no cover
            InplaceView(numpy.array([1, 2, 3], dtype=numpy.ubyte))

            with pytest.raises(ValueError, match='Buffer dtype mismatch'):
                InplaceView(numpy.array([1, 2, 3], dtype=numpy.ushort))

            with pytest.raises(ValueError, match='Buffer dtype mismatch'):
                InplaceView(numpy.array([1, 2, 3], dtype=numpy.uint))

    def test___iter__(self, hexview):
        for start in range(len(hexview)):
            for endex in range(len(hexview)):
                subview = hexview[start:endex]
                assert list(InplaceView(subview)) == list(subview)

    def test___len__(self, hexview):
        for start in range(len(hexview)):
            for endex in range(len(hexview)):
                subview = hexview[start:endex]
                assert len(InplaceView(subview)) == len(subview)

    def test___reversed__(self, hexview):
        for start in range(len(hexview)):
            for endex in range(len(hexview)):
                subview = hexview[start:endex]
                assert list(reversed(InplaceView(subview))) == list(reversed(subview))

    def test___richcmp__(self):
        pass  # TODO

    def test___setitem__(self, hexview):
        instance = InplaceView(hexview)
        size = len(hexview)

        for i in range(size):
            before = hexview[i]
            instance[i] = 255 - i
            after = hexview[i]
            assert before != after
            assert after == 255 - i

        instance = InplaceView(b'abc')
        with pytest.raises(TypeError, match='object does not support item assignment'):
            instance[0] = 0

    def test___sizeof__(self, hexview):
        instance = InplaceView(hexview)
        assert instance.__sizeof__() > 0

    def test_c_contiguous(self, hexview):
        assert InplaceView(None).c_contiguous is True
        assert InplaceView(hexview).c_contiguous is True

    def test_capitalize(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.capitalize() == bytestr.capitalize()

        buffer = loremstr.lower()
        instance = InplaceView(bytearray(buffer))
        assert instance.capitalize() == buffer.capitalize()

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

    def test_contiguous(self, hexview):
        assert InplaceView(None).contiguous is True
        assert InplaceView(hexview).contiguous is True

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

    def test_f_contiguous(self, hexview):
        assert InplaceView(None).f_contiguous is True
        assert InplaceView(hexview).f_contiguous is True

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

    def test_format(self, hexview):
        assert InplaceView(None).format == 'B'
        assert InplaceView(hexview).format == 'B'

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

    def test_isalnum(self):
        assert InplaceView(b'H3ll0W0rld').isalnum() is True
        assert InplaceView(b'H3ll0W0rld!').isalnum() is False
        assert InplaceView(b'').isalnum() is False

    def test_isalpha(self):
        assert InplaceView(b'HelloWorld').isalpha() is True
        assert InplaceView(b'H3ll0W0rld').isalpha() is False
        assert InplaceView(b'').isalpha() is False

    def test_isascii(self):
        assert InplaceView(bytes(range(128))).isascii() is True
        assert InplaceView(bytes(range(129))).isascii() is False
        assert InplaceView(b'').isascii() is False

    def test_isdecimal(self, hexview):
        assert InplaceView(hexview[:10]).isdecimal() is True
        assert InplaceView(hexview).isdecimal() is False
        assert InplaceView(b'').isdecimal() is False

    def test_isdigit(self, hexview):
        assert InplaceView(hexview[:10]).isdigit() is True
        assert InplaceView(hexview).isdigit() is False
        assert InplaceView(b'').isdigit() is False

    def test_isidentifier(self, hexstr, bytestr):
        assert InplaceView(hexstr[::-1]).isidentifier() is True
        assert InplaceView(hexstr).isidentifier() is False
        assert InplaceView(bytestr).isidentifier() is False

        assert InplaceView(b'a').isidentifier() is True
        assert InplaceView(b'_').isidentifier() is True
        assert InplaceView(b'a_').isidentifier() is True
        assert InplaceView(b'a0').isidentifier() is True
        assert InplaceView(b'_a').isidentifier() is True
        assert InplaceView(b'_0').isidentifier() is True

        assert InplaceView(b'').isidentifier() is False
        assert InplaceView(b'0').isidentifier() is False
        assert InplaceView(b'0a').isidentifier() is False
        assert InplaceView(b'0_').isidentifier() is False

        table = memoryview(bytestr)
        for i in range(128):
            assert InplaceView(table[i:(i + 1)]).isidentifier() is chr(i).isidentifier()
        for i in range(128, 256):
            assert InplaceView(table[i:(i + 1)]).isidentifier() is False

    def test_islower(self, hexstr):
        assert InplaceView(hexstr.lower()).islower() is True
        assert InplaceView(hexstr.upper()).islower() is False
        assert InplaceView(b'').islower() is False

    def test_isnumeric(self, hexview):
        assert InplaceView(hexview[:10]).isnumeric() is True
        assert InplaceView(hexview).isnumeric() is False
        assert InplaceView(b'').isnumeric() is False

    def test_isprintable(self):
        for i in range(0x20):
            assert InplaceView(bytes([i])).isprintable() is False

        for i in range(0x20, 0x7E + 1):
            assert InplaceView(bytes([i])).isprintable() is True

        for i in range(0x7E + 1, 256):
            assert InplaceView(bytes([i])).isprintable() is False

    def test_isspace(self):
        assert InplaceView(b'\x09\x0A\x0B\x0C\x0D\x1C\x1D\x1E\x1F\x20').isspace() is True
        assert InplaceView(b'\x09\x0A\x0B\x0C\x0D\x1C\x1D\x1E\x1F\x20\x00').isspace() is False
        assert InplaceView(b'').isspace() is False

    def test_istitle(self, loremstr):
        assert InplaceView(loremstr.title()).istitle() is True
        assert InplaceView(loremstr.capitalize()).istitle() is False
        assert InplaceView(b'').istitle() is False

    def test_isupper(self, hexstr):
        assert InplaceView(hexstr.upper()).isupper() is True
        assert InplaceView(hexstr.lower()).isupper() is False
        assert InplaceView(b'').isupper() is False

    def test_itemsize(self, hexview):
        assert InplaceView(None).itemsize == 1
        assert InplaceView(hexview).itemsize == 1

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

    def test_nbytes(self, hexview):
        assert InplaceView(b'').nbytes == 0
        assert InplaceView(hexview).nbytes == len(hexview)

    def test_ndim(self, hexview):
        assert InplaceView(None).ndim == 1
        assert InplaceView(hexview).ndim == 1

    def test_obj(self, bytestr, hexstr, hexview):
        instance = InplaceView(None)
        assert instance.obj is None
        instance.release()
        assert instance.obj is None

        instance = InplaceView(bytestr)
        assert instance.obj is bytestr
        instance.release()
        assert instance.obj is None

        instance = InplaceView(hexstr)
        assert instance.obj is hexstr
        instance.release()
        assert instance.obj is None

        instance = InplaceView(hexview)
        assert instance.obj is hexview
        instance.release()
        assert instance.obj is None

    def test_readonly(self, bytestr, hexstr, hexview):
        instance = InplaceView(None)
        assert instance.readonly is True

        instance = InplaceView(bytestr)
        assert instance.readonly is True
        with pytest.raises(TypeError, match='object does not support item assignment'):
            instance[0] = 0

        instance = InplaceView(hexstr)
        assert instance.readonly is False
        instance[0] = 0

        instance = InplaceView(hexview)
        assert instance.readonly is False
        instance[0] = 0

    def test_release(self, hexview):
        instance = InplaceView(hexview)
        assert instance.obj is hexview
        instance.release()
        assert instance.obj is None
        instance.release()
        assert instance.obj is None

    def test_replace(self, hexstr):
        hexview = memoryview(hexstr)
        instance = InplaceView(hexview)
        negbytes = bytes(255 - c for c in hexstr)
        negview = memoryview(negbytes)
        for i in range(len(hexview)):
            instance.replace(hexview[i:(i + 1)], negview[i:(i + 1)])
        assert hexview == negbytes

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

    def test_shape(self, hexview):
        assert InplaceView(b'').shape == (0,)
        assert InplaceView(hexview).shape == (len(hexview),)

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

    def test_strides(self, hexview):
        assert InplaceView(None).strides == (1,)
        assert InplaceView(hexview).strides == (1,)

    def test_suboffsets(self, hexview):
        assert InplaceView(None).suboffsets == ()
        assert InplaceView(hexview).suboffsets == ()

    def test_swapcase(self, bytestr, loremstr):
        instance = InplaceView(memoryview(bytearray(bytestr)))
        assert instance.swapcase() == bytestr.swapcase()

        buffer = loremstr.title()
        instance = InplaceView(bytearray(buffer))
        assert instance.swapcase() == buffer.swapcase()

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

    def test_tobytes(self, hexview, hexstr):
        exported = InplaceView(hexview).tobytes()
        assert isinstance(exported, bytes)
        assert exported == hexstr
        assert exported is not hexstr

    def test_tolist(self, hexview):
        exported = InplaceView(hexview).tolist()
        assert isinstance(exported, list)
        assert len(exported) == len(hexview)
        assert all(e == h for e, h in zip(exported, hexview))

    def test_toreadonly(self, hexview):
        instance = InplaceView(hexview)
        assert instance.readonly is False
        instance[0] = 0
        readonly = instance.toreadonly()
        assert readonly.readonly is True
        with pytest.raises(TypeError, match='object does not support item assignment'):
            readonly[1] = 1

    def test_translate(self, bytestr):
        table = InplaceView.maketrans(b'', b'')
        instance = InplaceView(bytearray(bytestr))
        instance.translate(table)
        assert instance.obj == bytestr

        table = InplaceView.maketrans(bytestr, bytestr)
        instance = InplaceView(bytearray(bytestr))
        instance.translate(table)
        assert instance.obj == bytestr

        revbytes = bytes(reversed(bytestr))
        table = InplaceView.maketrans(bytestr, revbytes)
        instance = InplaceView(bytearray(bytestr))
        instance.translate(table)
        assert instance.obj == revbytes

        table = InplaceView.maketrans(revbytes, bytestr)
        instance = InplaceView(bytearray(revbytes))
        instance.translate(table)
        assert instance.obj == bytestr

        with pytest.raises(ValueError, match='translation table must be 256 characters long'):
            instance = InplaceView(bytearray(bytestr))
            instance.translate(table[:-1])

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
