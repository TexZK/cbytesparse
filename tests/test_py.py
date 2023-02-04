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

from typing import Type

from _buffers import *
from _common import *

from cbytesparse.py import BytesMethods as _BytesMethods
from cbytesparse.py import InplaceView as _InplaceView
from cbytesparse.py import Memory as _Memory
from cbytesparse.py import bytesparse as _bytesparse


# Check tests against `bytes` for more consistency with Python
class TestBytesMethods_bytes(BytesMethodsSuite):
    BytesMethods: Type[bytes] = bytes
    SUPPORTS_NONE: bool = False

    # Delete incompatible tests
    test___delitem__ = None
    test_contains = None
    test_c_contiguous = None
    test_contiguous = None
    test_f_contiguous = None
    test_format = None
    test_isdecimal = None
    test_isidentifier = None
    test_isnumeric = None
    test_isprintable = None
    test_itemsize = None
    test_nbytes = None
    test_ndim = None
    test_obj = None
    test_readonly = None
    test_release = None
    test_shape = None
    test_strides = None
    test_suboffsets = None
    test_tobytes = None
    test_tolist = None


# Check tests against `bytearray` for more consistency with Python
class TestBytesMethods_bytearray(TestBytesMethods_bytes):
    BytesMethods: Type[bytearray] = bytearray

    # Delete incompatible tests
    test___setitem__ = None


# Check tests against `memoryview` for more consistency with Python
class TestBytesMethods_memoryview(BytesMethodsSuite):
    BytesMethods: Type[memoryview] = memoryview
    SUPPORTS_NONE: bool = False

    # Delete incompatible tests
    test___contains__ = None
    test___lt__ = None
    test___le__ = None
    test___ge__ = None
    test___gt__ = None
    test_capitalize = None
    test_contains = None
    test_count = None
    test_endswith = None
    test_find = None
    test_find_multi = None
    test_index = None
    test_index_multi = None
    test_isalnum = None
    test_isalpha = None
    test_isascii = None
    test_isdecimal = None
    test_isdigit = None
    test_isidentifier = None
    test_islower = None
    test_isnumeric = None
    test_isprintable = None
    test_isspace = None
    test_istitle = None
    test_isupper = None
    test_lower = None
    test_maketrans = None
    test_replace = None
    test_rfind = None
    test_rfind_multi = None
    test_rindex = None
    test_rindex_multi = None
    test_startswith = None
    test_swapcase = None
    test_title = None
    test_translate = None
    test_upper = None


class TestBytesMethods(BytesMethodsSuite):
    BytesMethods: Type['_BytesMethods'] = _BytesMethods


class TestInplaceView(InplaceViewSuite):
    BytesMethods: Type['_InplaceView'] = _InplaceView


class DONT_TestMemory(BaseMemorySuite):  # FIXME
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
