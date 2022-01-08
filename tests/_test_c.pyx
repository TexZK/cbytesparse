# cython: language_level = 3
# cython: embedsignature = True
# cython: binding = True

# Copyright (c) 2020-2022, Andrea Zoppi.
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

# NOTES
# Assertion "is True/False" is to ensure the answer is EXACTLY the expected one.

import pytest

from cbytesparse.c import collapse_blocks

from cbytesparse.c cimport *

cdef:
    bytes DATA1 = b'Hello, World!'
    size_t SIZE1 = <size_t>len(DATA1)
    tuple RACK1 = ((0x1234, DATA1),)

    bytes DATA2 = b'Foo/Bar'
    size_t SIZE2 = <size_t>len(DATA2)
    tuple RACK2 = ((0x4321, DATA2),)

    bytes DATA3 = b'#.#.##...##..###..##.#.#.'
    size_t SIZE3 = <size_t>len(DATA3)
    tuple TOKENS3 = (
        b'.#',
        b'#.',
        b'...',
        b'.#.',
        b'#.#',
        b'###',
        b'....',
        b'####',
    )


# =====================================================================================================================

def test_collapse_blocks___doctest__():
    blocks = [
        [0, b'0123456789'],
        [0, b'ABCD'],
        [3, b'EF'],
        [0, b'$'],
        [6, b'xyz'],
    ]
    ans_out = collapse_blocks(blocks)
    ans_ref = [[0, b'$BCEF5xyz9']]
    assert ans_out == ans_ref

    blocks = [
        [0, b'012'],
        [4, b'AB'],
        [6, b'xyz'],
        [1, b'$'],
    ]
    ans_out = collapse_blocks(blocks)
    ans_ref = [[0, b'0$2'], [4, b'ABxyz']]
    assert ans_out == ans_ref


# =====================================================================================================================

def test_addr_size_types():
    assert sizeof(size_t) == sizeof(ssize_t)
    assert SSIZE_MAX == +<ssize_t>SIZE_HMAX
    assert SSIZE_MIN == -<ssize_t>(SIZE_HMAX) - <ssize_t>1

    assert sizeof(addr_t) == sizeof(saddr_t)
    assert SADDR_MAX == +<saddr_t>(ADDR_MAX >> 1)
    assert SADDR_MIN == -<saddr_t>(ADDR_MAX >> 1) - <saddr_t>1

    assert sizeof(size_t) <= sizeof(addr_t)
    assert sizeof(ssize_t) <= sizeof(saddr_t)
    assert SIZE_MAX <= ADDR_MAX
    assert SSIZE_MAX <= SADDR_MAX
    assert SSIZE_MIN >= SADDR_MIN


# ---------------------------------------------------------------------------------------------------------------------

def test_calloc():
    cdef:
        byte_t* chunk = NULL
        size_t i

    chunk = <byte_t*>PyMem_Calloc(7, 5)
    try:
        assert chunk
        assert all(chunk[i] == 0 for i in range(7 * 5))

    finally:
        PyMem_Free(chunk)


# =====================================================================================================================

def test_downsize():
    cdef:
        size_t ans
        size_t ref
        size_t i

    ans = Downsize(0, 0)
    ref = MARGIN + MARGIN
    assert ans == ref, (ans, ref)

    for i in range(0x100 // 2):
        ans = Downsize(0x100, i)
        ref = i
        ref += (2 * MARGIN) - (ref % MARGIN)
        assert ans == ref, (ans, ref, i)

    ans = Downsize(0x100, 0x80)
    ref = 0x100
    assert ans == ref, (ans, ref)

    ans = Downsize(0x100, 0x7F)
    ref = 0x80 + MARGIN
    assert ans == ref, (ans, ref)


def test_upsize():
    cdef:
        size_t ans
        size_t ref
        size_t i

    ans = Upsize(0, 0)
    ref = MARGIN + MARGIN
    assert ans == ref, (ans, ref)

    ans = Upsize(0, 0x100)
    ref = MARGIN + 0x100 + MARGIN
    assert ans == ref, (ans, ref)

    for i in range((0x100 >> 3) + 1):
        ans = Upsize(0x100, 0x100 + i)
        ref = 0x100 + i
        ref += ref >> 3
        ref += (2 * MARGIN) - (ref % MARGIN)
        assert ans == ref, (ans, ref, i)


# =====================================================================================================================

def test_AddSizeU():
    assert AddSizeU(0, 0) == 0
    assert AddSizeU(0, 1) == 1
    assert AddSizeU(1, 0) == 1
    assert AddSizeU(1, 1) == 2
    assert AddSizeU(SIZE_MAX, 0) == SIZE_MAX
    assert AddSizeU(0, SIZE_MAX) == SIZE_MAX
    assert AddSizeU(SIZE_HMAX, SIZE_HMAX) == SIZE_MAX - 1
    assert AddSizeU(SIZE_HMAX + 1, SIZE_HMAX) == SIZE_MAX
    assert AddSizeU(SIZE_HMAX, SIZE_HMAX + 1) == SIZE_MAX

    with pytest.raises(OverflowError): AddSizeU(SIZE_MAX, 1)
    with pytest.raises(OverflowError): AddSizeU(1, SIZE_MAX)
    with pytest.raises(OverflowError): AddSizeU(SIZE_MAX, SIZE_MAX)
    with pytest.raises(OverflowError): AddSizeU(SIZE_HMAX + 1, SIZE_HMAX + 1)


def test_SubSizeU():
    assert SubSizeU(0, 0) == 0
    assert SubSizeU(1, 0) == 1
    assert SubSizeU(1, 1) == 0
    assert SubSizeU(SIZE_MAX, 0) == SIZE_MAX
    assert SubSizeU(SIZE_MAX, SIZE_MAX) == 0

    with pytest.raises(OverflowError): SubSizeU(0, 1)
    with pytest.raises(OverflowError): SubSizeU(0, SIZE_MAX)
    with pytest.raises(OverflowError): SubSizeU(SIZE_HMAX, SIZE_MAX)
    with pytest.raises(OverflowError): SubSizeU(SIZE_MAX - 1, SIZE_MAX)


def test_MulSizeU():
    cdef size_t SIZE_MAX_SQRT = (<size_t>1 << ((8 * sizeof(size_t)) // 2)) - 1
    assert 0 < SIZE_MAX_SQRT < SIZE_MAX

    assert MulSizeU(0, 0) == 0
    assert MulSizeU(0, 1) == 0
    assert MulSizeU(1, 0) == 0
    assert MulSizeU(1, 1) == 1
    assert MulSizeU(0, SIZE_MAX) == 0
    assert MulSizeU(1, SIZE_MAX) == SIZE_MAX
    assert MulSizeU(SIZE_MAX, 0) == 0
    assert MulSizeU(SIZE_MAX, 1) == SIZE_MAX
    assert MulSizeU(SIZE_HMAX, 2) == SIZE_MAX - 1
    assert MulSizeU(2, SIZE_HMAX) == SIZE_MAX - 1
    assert MulSizeU(SIZE_MAX_SQRT, SIZE_MAX_SQRT) == SIZE_MAX - SIZE_MAX_SQRT - SIZE_MAX_SQRT
    assert MulSizeU(SIZE_MAX_SQRT + 1, SIZE_MAX_SQRT) == SIZE_MAX - SIZE_MAX_SQRT
    assert MulSizeU(SIZE_MAX_SQRT, SIZE_MAX_SQRT + 1) == SIZE_MAX - SIZE_MAX_SQRT

    with pytest.raises(OverflowError): MulSizeU(SIZE_MAX, 2)
    with pytest.raises(OverflowError): MulSizeU(2, SIZE_MAX)
    with pytest.raises(OverflowError): MulSizeU(SIZE_MAX, SIZE_MAX)
    with pytest.raises(OverflowError): MulSizeU(SIZE_HMAX, SIZE_HMAX)
    with pytest.raises(OverflowError): MulSizeU(SIZE_MAX_SQRT + 1, SIZE_MAX_SQRT + 1)


# ---------------------------------------------------------------------------------------------------------------------

def test_AddSizeS():
    assert AddSizeS( 0,  0) == 0
    assert AddSizeS( 0, +1) == +1
    assert AddSizeS(+1,  0) == +1
    assert AddSizeS(+1, +1) == +2
    assert AddSizeS( 0, -1) == -1
    assert AddSizeS(-1,  0) == -1
    assert AddSizeS(-1, -1) == -2
    assert AddSizeS(SSIZE_MAX,  0) == SSIZE_MAX
    assert AddSizeS(SSIZE_MAX, -1) == SSIZE_MAX - 1
    assert AddSizeS(SSIZE_MIN,  0) == SSIZE_MIN
    assert AddSizeS(SSIZE_MIN, +1) == SSIZE_MIN + 1
    assert AddSizeS( 0, SSIZE_MAX) == SSIZE_MAX
    assert AddSizeS( 0, SSIZE_MIN) == SSIZE_MIN
    assert AddSizeS(-1, SSIZE_MAX) == SSIZE_MAX - 1
    assert AddSizeS(+1, SSIZE_MIN) == SSIZE_MIN + 1
    assert AddSizeS(SSIZE_MAX, SSIZE_MIN) == -1
    assert AddSizeS(SSIZE_MIN, SSIZE_MAX) == -1

    with pytest.raises(OverflowError): AddSizeS(SSIZE_MAX, +1)
    with pytest.raises(OverflowError): AddSizeS(+1, SSIZE_MAX)
    with pytest.raises(OverflowError): AddSizeS(SSIZE_MAX, SSIZE_MAX)
    with pytest.raises(OverflowError): AddSizeS(SSIZE_MIN, -1)
    with pytest.raises(OverflowError): AddSizeS(-1, SSIZE_MIN)
    with pytest.raises(OverflowError): AddSizeS(SSIZE_MIN, SSIZE_MIN)


def test_CheckSubSizeS():
    assert SubSizeS( 0,  0) == 0
    assert SubSizeS( 0, +1) == -1
    assert SubSizeS(+1,  0) == +1
    assert SubSizeS(+1, +1) == 0
    assert SubSizeS( 0, -1) == +1
    assert SubSizeS(-1,  0) == -1
    assert SubSizeS(-1, -1) == 0
    assert SubSizeS( 0, SSIZE_MAX) == SSIZE_MIN + 1
    assert SubSizeS(+1, SSIZE_MAX) == SSIZE_MIN + 2
    assert SubSizeS(-1, SSIZE_MAX) == SSIZE_MIN
    assert SubSizeS(-1, SSIZE_MIN) == SSIZE_MAX
    assert SubSizeS(SSIZE_MAX,  0) == SSIZE_MAX
    assert SubSizeS(SSIZE_MAX, +1) == SSIZE_MAX - 1
    assert SubSizeS(SSIZE_MIN,  0) == SSIZE_MIN
    assert SubSizeS(SSIZE_MIN, -1) == SSIZE_MIN + 1
    assert SubSizeS(SSIZE_MAX, SSIZE_MAX) == 0
    assert SubSizeS(SSIZE_MIN, SSIZE_MIN) == 0

    with pytest.raises(OverflowError): SubSizeS( 0, SSIZE_MIN)
    with pytest.raises(OverflowError): SubSizeS(+1, SSIZE_MIN)
    with pytest.raises(OverflowError): SubSizeS(SSIZE_MAX, -1)
    with pytest.raises(OverflowError): SubSizeS(SSIZE_MIN, +1)
    with pytest.raises(OverflowError): SubSizeS(SSIZE_MAX, SSIZE_MIN)
    with pytest.raises(OverflowError): SubSizeS(SSIZE_MIN, SSIZE_MAX)


def test_CheckMulSizeS():
    assert MulSizeS( 0,  0) == 0
    assert MulSizeS( 0, +1) == 0
    assert MulSizeS(+1,  0) == 0
    assert MulSizeS(+1, +1) == +1
    assert MulSizeS( 0, -1) == 0
    assert MulSizeS(-1,  0) == 0
    assert MulSizeS(-1, -1) == +1
    assert MulSizeS( 0, SSIZE_MAX) == 0
    assert MulSizeS(+1, SSIZE_MAX) == SSIZE_MAX
    assert MulSizeS(-1, SSIZE_MAX) == SSIZE_MIN + 1
    assert MulSizeS( 0, SSIZE_MIN) == 0
    assert MulSizeS(+1, SSIZE_MIN) == SSIZE_MIN
    assert MulSizeS(SSIZE_MAX,  0) == 0
    assert MulSizeS(SSIZE_MAX, +1) == SSIZE_MAX
    assert MulSizeS(SSIZE_MAX, -1) == SSIZE_MIN + 1
    assert MulSizeS(SSIZE_MIN,  0) == 0
    assert MulSizeS(SSIZE_MIN, +1) == SSIZE_MIN

    with pytest.raises(OverflowError): MulSizeS(-1, SSIZE_MIN)
    with pytest.raises(OverflowError): MulSizeS(SSIZE_MAX, +2)
    with pytest.raises(OverflowError): MulSizeS(+2, SSIZE_MAX)
    with pytest.raises(OverflowError): MulSizeS(SSIZE_MAX, -2)
    with pytest.raises(OverflowError): MulSizeS(-2, SSIZE_MAX)
    with pytest.raises(OverflowError): MulSizeS(SSIZE_MAX, SSIZE_MAX)
    with pytest.raises(OverflowError): MulSizeS(SSIZE_MIN, -1)
    with pytest.raises(OverflowError): MulSizeS(SSIZE_MIN, +2)
    with pytest.raises(OverflowError): MulSizeS(+2, SSIZE_MIN)
    with pytest.raises(OverflowError): MulSizeS(SSIZE_MIN, -2)
    with pytest.raises(OverflowError): MulSizeS(-2, SSIZE_MIN)


# =====================================================================================================================

def test_AddAddrU():
    assert AddAddrU(0, 0) == 0
    assert AddAddrU(0, 1) == 1
    assert AddAddrU(1, 0) == 1
    assert AddAddrU(1, 1) == 2
    assert AddAddrU(ADDR_MAX, 0) == ADDR_MAX
    assert AddAddrU(0, ADDR_MAX) == ADDR_MAX

    with pytest.raises(OverflowError): AddAddrU(ADDR_MAX, 1)
    with pytest.raises(OverflowError): AddAddrU(1, ADDR_MAX)
    with pytest.raises(OverflowError): AddAddrU(ADDR_MAX, ADDR_MAX)


def test_SubAddrU():
    assert SubAddrU(0, 0) == 0
    assert SubAddrU(1, 0) == 1
    assert SubAddrU(1, 1) == 0
    assert SubAddrU(ADDR_MAX, 0) == ADDR_MAX
    assert SubAddrU(ADDR_MAX, ADDR_MAX) == 0

    with pytest.raises(OverflowError): SubAddrU(0, 1)
    with pytest.raises(OverflowError): SubAddrU(0, ADDR_MAX)
    with pytest.raises(OverflowError): SubAddrU(ADDR_MAX - 1, ADDR_MAX)


def test_MulAddrU():
    cdef addr_t ADDR_MAX_SQRT = (<addr_t>1 << ((8 * sizeof(addr_t)) // 2)) - 1
    assert ADDR_MAX_SQRT > 0, ADDR_MAX_SQRT

    assert MulAddrU(0, 0) == 0
    assert MulAddrU(0, 1) == 0
    assert MulAddrU(1, 0) == 0
    assert MulAddrU(1, 1) == 1
    assert MulAddrU(0, ADDR_MAX) == 0
    assert MulAddrU(1, ADDR_MAX) == ADDR_MAX
    assert MulAddrU(ADDR_MAX, 0) == 0
    assert MulAddrU(ADDR_MAX, 1) == ADDR_MAX
    assert MulAddrU(ADDR_MAX_SQRT, ADDR_MAX_SQRT) == ADDR_MAX - ADDR_MAX_SQRT - ADDR_MAX_SQRT
    assert MulAddrU(ADDR_MAX_SQRT + 1, ADDR_MAX_SQRT) == ADDR_MAX - ADDR_MAX_SQRT
    assert MulAddrU(ADDR_MAX_SQRT, ADDR_MAX_SQRT + 1) == ADDR_MAX - ADDR_MAX_SQRT

    with pytest.raises(OverflowError): MulAddrU(ADDR_MAX, 2)
    with pytest.raises(OverflowError): MulAddrU(2, ADDR_MAX)
    with pytest.raises(OverflowError): MulAddrU(ADDR_MAX, ADDR_MAX)
    with pytest.raises(OverflowError): MulAddrU(ADDR_MAX_SQRT + 1, ADDR_MAX_SQRT + 1)


def test_AddrToSizeU():
    assert AddrToSizeU(0) == 0
    assert AddrToSizeU(<addr_t>SIZE_MAX) == SIZE_MAX

    if SIZE_MAX < ADDR_MAX:
        with pytest.raises(OverflowError): AddrToSizeU(AddAddrU(SIZE_MAX, 1))
    else:
        assert AddrToSizeU(ADDR_MAX) == <size_t>ADDR_MAX


# ---------------------------------------------------------------------------------------------------------------------

def test_AddAddrS():
    assert AddAddrS( 0,  0) == 0
    assert AddAddrS( 0, +1) == +1
    assert AddAddrS(+1,  0) == +1
    assert AddAddrS(+1, +1) == +2
    assert AddAddrS( 0, -1) == -1
    assert AddAddrS(-1,  0) == -1
    assert AddAddrS(-1, -1) == -2
    assert AddAddrS(SADDR_MAX,  0) == SADDR_MAX
    assert AddAddrS(SADDR_MAX, -1) == SADDR_MAX - 1
    assert AddAddrS(SADDR_MIN,  0) == SADDR_MIN
    assert AddAddrS(SADDR_MIN, +1) == SADDR_MIN + 1
    assert AddAddrS( 0, SADDR_MAX) == SADDR_MAX
    assert AddAddrS( 0, SADDR_MIN) == SADDR_MIN
    assert AddAddrS(-1, SADDR_MAX) == SADDR_MAX - 1
    assert AddAddrS(+1, SADDR_MIN) == SADDR_MIN + 1
    assert AddAddrS(SADDR_MAX, SADDR_MIN) == -1
    assert AddAddrS(SADDR_MIN, SADDR_MAX) == -1

    with pytest.raises(OverflowError): AddAddrS(SADDR_MAX, +1)
    with pytest.raises(OverflowError): AddAddrS(+1, SADDR_MAX)
    with pytest.raises(OverflowError): AddAddrS(SADDR_MAX, SADDR_MAX)
    with pytest.raises(OverflowError): AddAddrS(SADDR_MIN, -1)
    with pytest.raises(OverflowError): AddAddrS(-1, SADDR_MIN)
    with pytest.raises(OverflowError): AddAddrS(SADDR_MIN, SADDR_MIN)


def test_SubAddrS():
    assert SubAddrS( 0,  0) == 0
    assert SubAddrS( 0, +1) == -1
    assert SubAddrS(+1,  0) == +1
    assert SubAddrS(+1, +1) == 0
    assert SubAddrS( 0, -1) == +1
    assert SubAddrS(-1,  0) == -1
    assert SubAddrS(-1, -1) == 0
    assert SubAddrS( 0, SADDR_MAX) == SADDR_MIN + 1
    assert SubAddrS(+1, SADDR_MAX) == SADDR_MIN + 2
    assert SubAddrS(-1, SADDR_MAX) == SADDR_MIN
    assert SubAddrS(-1, SADDR_MIN) == SADDR_MAX
    assert SubAddrS(SADDR_MAX,  0) == SADDR_MAX
    assert SubAddrS(SADDR_MAX, +1) == SADDR_MAX - 1
    assert SubAddrS(SADDR_MIN,  0) == SADDR_MIN
    assert SubAddrS(SADDR_MIN, -1) == SADDR_MIN + 1
    assert SubAddrS(SADDR_MAX, SADDR_MAX) == 0
    assert SubAddrS(SADDR_MIN, SADDR_MIN) == 0

    with pytest.raises(OverflowError): SubAddrS( 0, SADDR_MIN)
    with pytest.raises(OverflowError): SubAddrS(+1, SADDR_MIN)
    with pytest.raises(OverflowError): SubAddrS(SADDR_MAX, -1)
    with pytest.raises(OverflowError): SubAddrS(SADDR_MIN, +1)
    with pytest.raises(OverflowError): SubAddrS(SADDR_MAX, SADDR_MIN)
    with pytest.raises(OverflowError): SubAddrS(SADDR_MIN, SADDR_MAX)


def test_MulAddrS():
    assert MulAddrS( 0,  0) == 0
    assert MulAddrS( 0, +1) == 0
    assert MulAddrS(+1,  0) == 0
    assert MulAddrS(+1, +1) == +1
    assert MulAddrS( 0, -1) == 0
    assert MulAddrS(-1,  0) == 0
    assert MulAddrS(-1, -1) == +1
    assert MulAddrS( 0, SADDR_MAX) == 0
    assert MulAddrS(+1, SADDR_MAX) == SADDR_MAX
    assert MulAddrS(-1, SADDR_MAX) == SADDR_MIN + 1
    assert MulAddrS( 0, SADDR_MIN) == 0
    assert MulAddrS(+1, SADDR_MIN) == SADDR_MIN
    assert MulAddrS(SADDR_MAX,  0) == 0
    assert MulAddrS(SADDR_MAX, +1) == SADDR_MAX
    assert MulAddrS(SADDR_MAX, -1) == SADDR_MIN + 1
    assert MulAddrS(SADDR_MIN,  0) == 0
    assert MulAddrS(SADDR_MIN, +1) == SADDR_MIN

    with pytest.raises(OverflowError): MulAddrS(-1, SADDR_MIN)
    with pytest.raises(OverflowError): MulAddrS(SADDR_MAX, +2)
    with pytest.raises(OverflowError): MulAddrS(+2, SADDR_MAX)
    with pytest.raises(OverflowError): MulAddrS(SADDR_MAX, -2)
    with pytest.raises(OverflowError): MulAddrS(-2, SADDR_MAX)
    with pytest.raises(OverflowError): MulAddrS(SADDR_MAX, SADDR_MAX)
    with pytest.raises(OverflowError): MulAddrS(SADDR_MIN, -1)
    with pytest.raises(OverflowError): MulAddrS(SADDR_MIN, +2)
    with pytest.raises(OverflowError): MulAddrS(+2, SADDR_MIN)
    with pytest.raises(OverflowError): MulAddrS(SADDR_MIN, -2)
    with pytest.raises(OverflowError): MulAddrS(-2, SADDR_MIN)


def test_AddrToSizeS():
    assert AddrToSizeS(0) == 0
    assert AddrToSizeS(<saddr_t>SSIZE_MAX) == SSIZE_MAX

    if SSIZE_MAX < SADDR_MAX:
        with pytest.raises(OverflowError): AddrToSizeS(AddAddrS(SSIZE_MAX, +1))
    else:
        assert AddrToSizeS(SADDR_MAX) == <ssize_t>SADDR_MAX

    if SSIZE_MIN > SADDR_MIN:
        with pytest.raises(OverflowError): AddrToSizeS(AddAddrS(SSIZE_MIN, -1))
    else:
        assert AddrToSizeS(SADDR_MIN) == <ssize_t>SADDR_MIN


# =====================================================================================================================

def test_Block_Alloc_Free():
    cdef:
        Block_* block = NULL

    try:
        block = Block_Alloc(0x1234, 0, True)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        block = Block_Alloc(0x1234, 0x100, True)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + 0x100 + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN + 0x100
        assert Block_Length(block) == 0x100
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1334
        assert Block_Eq_(block, 0x100, b'\0' * 0x100) is True
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Create():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1

    try:
        block = Block_Create(0x1234, 0, NULL)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, data)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == size + ((2 * MARGIN) - (size % MARGIN))
        assert block.start == MARGIN
        assert block.endex == MARGIN + size
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert all(Block_Get__(block, i) == data[i] for i in range(size))
        block = Block_Free(block)

        with pytest.raises(ValueError, match='null pointer'):
            block = Block_Create(0x1234, size, NULL)

    finally:
        block = Block_Free(block)


def test_Block_Copy():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i

    try:
        block1 = Block_Create(0x1234, size, data)

        block2 = Block_Copy(block1)
        assert block2 != NULL
        assert block2 != block1
        assert block2.address == block1.address
        assert block2.allocated == block1.allocated
        assert block2.start == block1.start
        assert block2.endex == block1.endex
        assert Block_Length(block2) == Block_Length(block1)
        assert Block_Start(block2) == Block_Start(block1)
        assert Block_Endex(block2) == Block_Endex(block1)
        assert all(Block_Get__(block2, i) == data[i] for i in range(size))
        block2 = Block_Free(block2)

        with pytest.raises(ValueError, match='null pointer'):
            block2 = Block_Copy(NULL)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_FromObject():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'invalid block data size'

    try:
        block = Block_FromObject(0x1234, 0x69, True)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated >= 1
        assert block.start == MARGIN
        assert block.endex == MARGIN + 1
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Get__(block, 0) == 0x69
        block = Block_Free(block)

        block = Block_FromObject(0x1234, data, True)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated >= size
        assert block.start == MARGIN
        assert block.endex == MARGIN + size
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert all(Block_Get__(block, i) == data[i] for i in range(size))
        block = Block_Free(block)

        block = Block_FromObject(0x1234, memoryview(data), True)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated >= size
        assert block.start == MARGIN
        assert block.endex == MARGIN + size
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert all(Block_Get__(block, i) == data[i] for i in range(size))
        block = Block_Free(block)

        block = Block_FromObject(0x1234, bytearray(data), True)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated >= size
        assert block.start == MARGIN
        assert block.endex == MARGIN + size
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert all(Block_Get__(block, i) == data[i] for i in range(size))
        block = Block_Free(block)

        block = Block_FromObject(0x1234, b'', False)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        block = Block_FromObject(0x1234, memoryview(b''), False)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        block = Block_FromObject(0x1234, bytearray(b''), False)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        with pytest.raises(ValueError, match=match):
            block = Block_FromObject(0x1234, b'', True)

        with pytest.raises(ValueError, match=match):
            block = Block_FromObject(0x1234, memoryview(b''), True)

        with pytest.raises(ValueError, match=match):
            block = Block_FromObject(0x1234, bytearray(b''), True)

    finally:
        block = Block_Free(block)


def test_Block_Acquire_Release_():
    cdef:
        Block_* block = NULL
        Block_* block_ = NULL

    try:
        block = Block_Create(0x1234, 0, NULL)
        block_ = block
        assert block != NULL
        assert block.references == 1

        block = Block_Acquire(block)
        assert block == block_
        assert block.references == 2

        block = Block_Release_(block)
        assert block == block_
        assert block.references == 1

        block = Block_Release_(block)
        assert block == NULL

    finally:
        block = Block_Free(block)


def test_Block_Acquire_Release():
    cdef:
        Block_* block = NULL
        Block_* block_ = NULL

    try:
        block = Block_Create(0x1234, 0, NULL)
        block_ = block
        assert block != NULL
        assert block.references == 1

        block = Block_Acquire(block)
        assert block == block_
        assert block.references == 2

        block_ = Block_Release(block)
        assert block_ == NULL
        assert block.references == 1

        block = Block_Release(block)
        assert block == NULL

    finally:
        block = Block_Free(block)


# TODO: test_Block_BoundAddress()


# TODO: test_Block_BoundAddressToOffset()


# TODO: test_Block_BoundOffset()


# TODO: test_Block_BoundAddressSlice()


# TODO: test_Block_BoundAddressSliceToOffset()


# TODO: test_Block_BoundOffsetSlice()


def test_Block_CheckMutable():
    cdef:
        Block_* block = NULL

    try:
        block = Block_Create(0x1234, 0, NULL)
        assert block.references == 1
        Block_CheckMutable(block)

        block = Block_Acquire(block)
        assert block.references == 2
        with pytest.raises(RuntimeError, match='Existing exports of data: object cannot be re-sized'):
            Block_CheckMutable(block)

        block = Block_Release_(block)
        assert block.references == 1
        Block_CheckMutable(block)

        block = Block_Release_(block)

    finally:
        block = Block_Free(block)


def test_Block_Eq():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset

    try:
        block2 = Block_Create(0x1234, size, data)

        block1 = Block_Copy(block2)
        assert Block_Eq(block1, block2) is True
        block1 = Block_Free(block1)

        for offset in range(size):
            block1 = Block_Copy(block2)
            Block_Set__(block1, offset, Block_Get__(block1, offset) ^ 0xFF)
            assert Block_Eq(block1, block2) is False
            block1 = Block_Free(block1)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_Cmp():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset

    try:
        block2 = Block_Create(0x1234, size, data)

        block1 = Block_Copy(block2)
        assert Block_Eq(block1, block2) is True
        assert Block_Cmp(block1, block2) is 0
        block1 = Block_Free(block1)

        for offset in range(size):
            block1 = Block_Copy(block2)
            Block_Set__(block1, offset, Block_Get__(block1, offset) - 1)
            assert Block_Eq(block1, block2) is False
            assert Block_Cmp(block1, block2) is -1
            block1 = Block_Free(block1)

        for offset in range(size):
            block1 = Block_Copy(block2)
            Block_Set__(block1, offset, Block_Get__(block1, offset) + 1)
            assert Block_Eq(block1, block2) is False
            assert Block_Cmp(block1, block2) is +1
            block1 = Block_Free(block1)

        block1 = Block_Copy(block2)
        block1 = Block_Pop__(block1, NULL)
        assert Block_Eq(block1, block2) is False
        assert Block_Cmp(block1, block2) is -1
        block1 = Block_Free(block1)

        block1 = Block_Copy(block2)
        block1 = Block_Append(block1, b'#'[0])
        assert Block_Eq(block1, block2) is False
        assert Block_Cmp(block1, block2) is +1
        block1 = Block_Free(block1)

        for offset in range(size):
            block1 = Block_Copy(block2)
            Block_Set__(block1, offset, Block_Get__(block1, offset) - 1)
            block1 = Block_Append(block1, b'#'[0])
            assert Block_Eq(block1, block2) is False
            assert Block_Cmp(block1, block2) is -1
            block1 = Block_Free(block1)

        for offset in range(size - 1):  # skip last pop
            block1 = Block_Copy(block2)
            Block_Set__(block1, offset, Block_Get__(block1, offset) + 1)
            block1 = Block_Pop__(block1, NULL)
            assert Block_Eq(block1, block2) is False
            assert Block_Cmp(block1, block2) is +1
            block1 = Block_Free(block1)

        offset = size - 1  # check last pop
        block1 = Block_Copy(block2)
        Block_Set__(block1, offset, Block_Get__(block1, offset) + 1)
        block1 = Block_Pop__(block1, NULL)
        assert Block_Eq(block1, block2) is False
        assert Block_Cmp(block1, block2) is -1
        block1 = Block_Free(block1)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_Find__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        byte_t value
        ssize_t ans
        ssize_t ref

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_Find__(block, 0, SIZE_MAX, value) == -1
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for start in range(size + 1):
                for endex in range(size + 1):
                    ans = Block_Find__(block, start, endex, value)
                    ref = data.find(value, start, endex)
                    assert ans == ref
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Find_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        byte_t value
        tuple tokens
        bytes token
        ssize_t ans
        ssize_t ref

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_Find_(block, 0, SIZE_MAX, 1, &value) == -1
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for start in range(size + 1):
                for endex in range(size + 1):
                    ans = Block_Find_(block, start, endex, 1, &value)
                    ref = data.find(value, start, endex)
                    assert ans == ref
        block = Block_Free(block)

        data = DATA3
        size = SIZE3
        tokens = TOKENS3
        block = Block_Create(0x1234, size, data)
        for token in tokens:
            for start in range(size + 1):
                for endex in range(size + 1):
                    ans = Block_Find_(block, start, endex, <size_t>len(token), token)
                    ref = data.find(token, start, endex)
                    assert ans == ref
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Find():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        byte_t value
        tuple tokens
        bytes token
        ssize_t ans
        ssize_t ref

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_Find(block, 0, SSIZE_MAX, 1, &value) == -1
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for start in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                for endex in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                    ans = Block_Find(block, start, endex, 1, &value)
                    ref = data.find(value, start, endex)
                    assert ans == ref
        block = Block_Free(block)

        data = DATA3
        size = SIZE3
        tokens = TOKENS3
        block = Block_Create(0x1234, size, data)
        for token in tokens:
            for start in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                for endex in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                    ans = Block_Find(block, start, endex, <size_t>len(token), token)
                    ref = data.find(token, start, endex)
                    assert ans == ref
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ReverseFind__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        byte_t value
        ssize_t ans
        ssize_t ref

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_ReverseFind__(block, 0, SIZE_MAX, value) == -1
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for start in range(size + 1):
                for endex in range(size + 1):
                    ans = Block_ReverseFind__(block, start, endex, value)
                    ref = data.rfind(value, start, endex)
                    assert ans == ref
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ReverseFind_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        byte_t value
        tuple tokens
        bytes token
        ssize_t ans
        ssize_t ref

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_ReverseFind_(block, 0, SIZE_MAX, 1, &value) == -1
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for start in range(size + 1):
                for endex in range(size + 1):
                    ans = Block_ReverseFind_(block, start, endex, 1, &value)
                    ref = data.rfind(value, start, endex)
                    assert ans == ref
        block = Block_Free(block)

        data = DATA3
        size = SIZE3
        tokens = TOKENS3
        block = Block_Create(0x1234, size, data)
        for token in tokens:
            for start in range(size + 1):
                for endex in range(size + 1):
                    ans = Block_ReverseFind_(block, start, endex, <size_t>len(token), token)
                    ref = data.rfind(token, start, endex)
                    assert ans == ref
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ReverseFind():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        byte_t value
        tuple tokens
        bytes token
        ssize_t ans
        ssize_t ref

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_ReverseFind(block, 0, SSIZE_MAX, 1, &value) == -1
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for start in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                for endex in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                    ans = Block_ReverseFind(block, start, endex, 1, &value)
                    ref = data.rfind(value, start, endex)
                    assert ans == ref
        block = Block_Free(block)

        data = DATA3
        size = SIZE3
        tokens = TOKENS3
        block = Block_Create(0x1234, size, data)
        for token in tokens:
            for start in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                for endex in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                    ans = Block_ReverseFind(block, start, endex, <size_t>len(token), token)
                    ref = data.rfind(token, start, endex)
                    assert ans == ref
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Count__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset
        byte_t value

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_Count__(block, 0, SIZE_MAX, value) == 0
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for offset in range(size + 1):
                assert Block_Count__(block, offset, SIZE_MAX, value) == data.count(value, offset)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Count_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset
        byte_t value
        tuple tokens
        bytes token

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_Count_(block, 0, SIZE_MAX, 1, &value) == 0
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for offset in range(size + 1):
                assert Block_Count_(block, offset, SIZE_MAX, 1, &value) == data.count(value, offset)
        block = Block_Free(block)

        data = DATA3
        size = SIZE3
        tokens = TOKENS3
        block = Block_Create(0x1234, size, data)
        for token in tokens:
            for offset in range(size + 1):
                assert Block_Count_(block, offset, SIZE_MAX, <size_t>len(token), token) == data.count(token, offset)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Count():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t offset
        byte_t value
        tuple tokens
        bytes token

    try:
        block = Block_Create(0x1234, 0, NULL)
        for value in set(data):
            assert Block_Count(block, SSIZE_MIN, SSIZE_MAX, 1, &value) == 0
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        for value in set(data):
            for offset in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                assert Block_Count(block, offset, SSIZE_MAX, 1, &value) == data.count(value, offset)
        block = Block_Free(block)

        data = DATA3
        size = SIZE3
        tokens = TOKENS3
        block = Block_Create(0x1234, size, data)
        for token in tokens:
            for offset in range(-<ssize_t>(size + 1), <ssize_t>(size + 1)):
                assert Block_Count(block, offset, SSIZE_MAX, <size_t>len(token), token) == data.count(token, offset)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Reserve_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Reserve_(block, 0, 0, True)  # same
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 0
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        block = Block_Reserve_(block, 0, 3, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 3 + 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 3 + 0
        assert Block_Eq_(block, 3 + 0, bytes(3)) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, 0, 0, True)  # same
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert Block_Eq_(block, size, data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, 0, 3, True)  # before
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 3 + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 3 + size
        assert Block_Eq_(block, 3 + size, bytes(3) + data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, size, 3, True)  # after
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 3
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 3
        assert Block_Eq_(block, size + 3, data + bytes(3)) is True
        block = Block_Free(block)

        offset = size >> 1  # half
        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, offset, 3, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 3
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 3
        assert Block_Eq_(block, size + 3, data[:offset] + bytes(3) + data[offset:]) is True
        block = Block_Free(block)

        offset = (size * 1) >> 2  # first quarter
        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, offset, 3, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 3
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 3
        assert Block_Eq_(block, size + 3, data[:offset] + bytes(3) + data[offset:]) is True
        block = Block_Free(block)

        offset = (size * 3) >> 2  # third quarter
        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, offset, 3, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 3
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 3
        assert Block_Eq_(block, size + 3, data[:offset] + bytes(3) + data[offset:]) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, 0, size, True)  # before
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + size
        assert Block_Eq_(block, size + size, bytes(size) + data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, size, size, True)  # after
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + size
        assert Block_Eq_(block, size + size, data + bytes(size)) is True
        block = Block_Free(block)

        offset = size >> 1  # half
        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, offset, size, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + size
        assert Block_Eq_(block, size + size, data[:offset] + bytes(size) + data[offset:]) is True
        block = Block_Free(block)

        offset = (size * 1) >> 2  # first quarter
        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, offset, size, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + size
        assert Block_Eq_(block, size + size, data[:offset] + bytes(size) + data[offset:]) is True
        block = Block_Free(block)

        offset = (size * 3) >> 2  # third quarter
        block = Block_Create(0x1234, size, data)
        block = Block_Reserve_(block, offset, size, True)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + size
        assert Block_Eq_(block, size + size, data[:offset] + bytes(size) + data[offset:]) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(OverflowError, match='size overflow'): Block_Reserve_(block, 0, SIZE_MAX, True)
        with pytest.raises(OverflowError, match='size overflow'): Block_Reserve_(block, 0, SIZE_HMAX, True)
        with pytest.raises(IndexError, match='index out of range'): Block_Reserve_(block, size + 1, 1, True)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Delete_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        bytearray buffer

    try:
        start = endex = 0
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Delete_(block, 0, 0)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        for start in range(size):
            for endex in range(start, size):
                buffer = bytearray(data)
                del buffer[start:endex]
                block = Block_Create(0x1234, size, data)
                block = Block_Delete_(block, start, endex - start)
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == <size_t>len(buffer)
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(OverflowError, match='size overflow'): Block_Delete_(block, 0, SIZE_MAX)
        with pytest.raises(IndexError, match='index out of range'): Block_Delete_(block, size + 1, 1)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Clear():
    cdef:
        Block_* block = NULL

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Clear(block)
        assert block != NULL
        assert block.address == 0x1234
        assert block.allocated == MARGIN + MARGIN
        assert block.start == MARGIN
        assert block.endex == MARGIN
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

        block = Block_FromObject(0x1234, DATA1, True)
        block = Block_Clear(block)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Get_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset
        str match = 'index out of range'

    try:
        block = Block_Create(0x1234, size, data)

        for offset in range(size):
            assert Block_Get_(block, offset) == data[offset]

        with pytest.raises(IndexError, match=match): Block_Get_(block,  13)
        with pytest.raises(IndexError, match=match): Block_Get_(block,  99)

    finally:
        block = Block_Free(block)


def test_Block_Get():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t offset
        str match = 'index out of range'

    try:
        block = Block_Create(0x1234, size, data)

        for offset in range(-<ssize_t>size, <ssize_t>size):
            assert Block_Get(block, offset) == data[offset]

        with pytest.raises(IndexError, match=match): Block_Get(block,  13)
        with pytest.raises(IndexError, match=match): Block_Get(block,  99)
        with pytest.raises(IndexError, match=match): Block_Get(block, -14)
        with pytest.raises(IndexError, match=match): Block_Get(block, -99)

    finally:
        block = Block_Free(block)


def test_Block_Set_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t offset
        byte_t value
        byte_t complement
        str match = 'index out of range'

    try:
        for offset in range(size):
            block = Block_Create(0x1234, size, data)
            value = Block_Get__(block, offset)
            complement = <byte_t>(value ^ <byte_t>0xFF)
            assert Block_Set_(block, offset, complement) == value
            assert Block_Get__(block, offset) == complement
            block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(IndexError, match=match): Block_Set_(block,  13, 0x69)
        with pytest.raises(IndexError, match=match): Block_Set_(block,  99, 0x69)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Set():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t offset
        byte_t value
        byte_t complement
        str match = 'index out of range'

    try:
        for offset in range(-<ssize_t>size, <ssize_t>size):
            block = Block_Create(0x1234, size, data)
            value = Block_Get(block, offset)
            complement = <byte_t>(value ^ <byte_t>0xFF)
            assert Block_Set(block, offset, complement) == value
            assert Block_Get(block, offset) == complement
            block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(IndexError, match=match): Block_Set(block,  13, 0x69)
        with pytest.raises(IndexError, match=match): Block_Set(block,  99, 0x69)
        with pytest.raises(IndexError, match=match): Block_Set(block, -14, 0x69)
        with pytest.raises(IndexError, match=match): Block_Set(block, -99, 0x69)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Pop__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'pop index out of range'
        byte_t backup

    try:
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop__(block, &backup)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:-1]) is True
        assert backup == b'!'[0]
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        with pytest.raises(IndexError, match=match): Block_Pop__(block, NULL)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Pop_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'pop index out of range'
        byte_t backup
        size_t offset

    try:
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop_(block, size - 1, &backup)  # final
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:-1]) is True
        assert backup == data[-1]
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop_(block, 0, &backup)  # initial
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[1:]) is True
        assert backup == data[0]
        block = Block_Free(block)

        offset = size >> 1  # half
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop_(block, offset, &backup)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:offset] + data[offset + 1:]) is True
        assert backup == data[offset]
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        with pytest.raises(IndexError, match=match): Block_Pop_(block, 0, NULL)
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(IndexError, match=match): Block_Pop_(block, size, NULL)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Pop():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'pop index out of range'
        byte_t backup
        ssize_t offset

    try:
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop(block, <ssize_t>size - 1, &backup)  # final
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:-1]) is True
        assert backup == data[-1]
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop(block, -1, &backup)  # final
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:-1]) is True
        assert backup == data[-1]
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop(block, 0, &backup)  # initial
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[1:]) is True
        assert backup == data[0]
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop(block, -<ssize_t>size, &backup)  # initial
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[1:]) is True
        assert backup == data[0]
        block = Block_Free(block)

        offset = <ssize_t>(size >> 1)  # half
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop(block, offset, &backup)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:offset] + data[offset + 1:]) is True
        assert backup == data[offset]
        block = Block_Free(block)

        offset = <ssize_t>(size >> 1)  # half
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_Pop(block, -offset, &backup)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[:-offset] + data[-offset + 1:]) is True
        assert backup == data[-offset]
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        with pytest.raises(IndexError, match=match): Block_Pop(block,  0, NULL)
        with pytest.raises(IndexError, match=match): Block_Pop(block, -1, NULL)
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(IndexError, match=match): Block_Pop(block, size, NULL)
        with pytest.raises(IndexError, match=match): Block_Pop(block, -<ssize_t>size - 1, NULL)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_PopLeft():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'pop index out of range'
        byte_t backup

    try:
        block = Block_Create(0x1234, size, data)
        backup = 0
        block = Block_PopLeft(block, &backup)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size - 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size - 1
        assert Block_Eq_(block, size - 1, data[1:]) is True
        assert backup == data[0]
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        with pytest.raises(IndexError, match=match): Block_PopLeft(block, NULL)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Insert_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'index out of range'
        size_t offset

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Insert_(block, 0, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Eq_(block, 1, b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert_(block, 0, b'@'[0])  # before
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1 + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1 + size
        assert Block_Eq_(block, 1 + size, b'@' + data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert_(block, size, b'@'[0])  # after
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data + b'@') is True
        block = Block_Free(block)

        offset = size >> 1  # half
        block = Block_Create(0x1234, size, data)
        block = Block_Insert_(block, offset, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data[:offset] + b'@' + data[offset:]) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        with pytest.raises(IndexError, match=match): Block_Insert_(block, 1, 0x69)
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        with pytest.raises(IndexError, match=match): Block_Insert_(block, size + 1, 0x69)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Insert():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        str match = 'index out of range'
        ssize_t offset

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Insert(block, 0, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Eq_(block, 1, b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        block = Block_Insert(block, -1, b'@'[0])  # before, over
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Eq_(block, 1, b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        block = Block_Insert(block, 1, b'@'[0])  # after, over
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Eq_(block, 1, b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, 0, b'@'[0])  # before
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1 + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1 + size
        assert Block_Eq_(block, 1 + size, b'@' + data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, -<ssize_t>size, b'@'[0])  # before
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1 + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1 + size
        assert Block_Eq_(block, 1 + size, b'@' + data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, -<ssize_t>size - 1, b'@'[0])  # before, over
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1 + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1 + size
        assert Block_Eq_(block, 1 + size, b'@' + data) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, <ssize_t>size, b'@'[0])  # after
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data + b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, <ssize_t>size + 1, b'@'[0])  # after, over
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data + b'@') is True
        block = Block_Free(block)

        offset = <ssize_t>(size >> 1)  # half
        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, offset, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data[:offset] + b'@' + data[offset:]) is True
        block = Block_Free(block)

        offset = -<ssize_t>(size >> 1)  # half
        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, offset, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data[:offset] + b'@' + data[offset:]) is True
        block = Block_Free(block)

        offset = -1  # before end
        block = Block_Create(0x1234, size, data)
        block = Block_Insert(block, offset, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data[:offset] + b'@' + data[offset:]) is True
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Append():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Append(block, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Eq_(block, 1, b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_Append(block, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size + 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size + 1
        assert Block_Eq_(block, size + 1, data + b'@') is True
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_AppendLeft():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_AppendLeft(block, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1
        assert Block_Eq_(block, 1, b'@') is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        block = Block_AppendLeft(block, b'@'[0])
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 1 + size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 1 + size
        assert Block_Eq_(block, 1 + size, b'@' + data) is True
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Extend_():
    cdef:
        Block_* block = NULL
        bytes data1 = DATA1
        size_t size1 = SIZE1
        bytes data2 = DATA2
        size_t size2 = SIZE2

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_Extend_(block, 0, NULL)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 0
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size1, data1)
        block = Block_Extend_(block, 0, NULL)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size1
        assert Block_Eq_(block, size1, data1) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        block = Block_Extend_(block, size2, data2)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size2
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size2
        assert Block_Eq_(block, size2, data2) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size1, data1)
        block = Block_Extend_(block, size2, data2)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size1 + size2
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size1 + size2
        assert Block_Eq_(block, size1 + size2, data1 + data2) is True
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Extend():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data1 = DATA1
        size_t size1 = SIZE1
        bytes data2 = DATA2
        size_t size2 = SIZE2

    try:
        block1 = Block_Create(0x1234, 0, NULL)

        block2 = Block_Create(0x1234, 0, NULL)
        block2 = Block_Extend(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == 0
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + 0
        assert Block_Eq_(block2, 0, NULL) is True
        block2 = Block_Free(block2)

        block2 = Block_Create(0x1234, size1, data1)
        block2 = Block_Extend(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == size1
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + size1
        assert Block_Eq_(block2, size1, data1) is True
        block2 = Block_Free(block2)

        block1 = Block_Free(block1)
        block1 = Block_Create(0x1234, size2, data2)

        block2 = Block_Create(0x1234, 0, NULL)
        block2 = Block_Extend(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == size2
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + size2
        assert Block_Eq_(block2, size2, data2) is True
        block2 = Block_Free(block2)

        block2 = Block_Create(0x1234, size1, data1)
        block2 = Block_Extend(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == size1 + size2
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + size1 + size2
        assert Block_Eq_(block2, size1 + size2, data1 + data2) is True
        block2 = Block_Free(block2)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_ExtendLeft():
    cdef:
        Block_* block = NULL
        bytes data1 = DATA1
        size_t size1 = SIZE1
        bytes data2 = DATA2
        size_t size2 = SIZE2

    try:
        block = Block_Create(0x1234, 0, NULL)
        block = Block_ExtendLeft_(block, 0, NULL)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + 0
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size1, data1)
        block = Block_ExtendLeft_(block, 0, NULL)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size1
        assert Block_Eq_(block, size1, data1) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        block = Block_ExtendLeft_(block, size2, data2)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size2
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size2
        assert Block_Eq_(block, size2, data2) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size1, data1)
        block = Block_ExtendLeft_(block, size2, data2)
        assert block != NULL
        assert block.address == 0x1234
        assert Block_Length(block) == size2 + size1
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size2 + size1
        assert Block_Eq_(block, size2 + size1, data2 + data1) is True
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ExtendLeft():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data1 = DATA1
        size_t size1 = SIZE1
        bytes data2 = DATA2
        size_t size2 = SIZE2

    try:
        block1 = Block_Create(0x1234, 0, NULL)

        block2 = Block_Create(0x1234, 0, NULL)
        block2 = Block_ExtendLeft(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == 0
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + 0
        assert Block_Eq_(block2, 0, NULL) is True
        block2 = Block_Free(block2)

        block2 = Block_Create(0x1234, size1, data1)
        block2 = Block_ExtendLeft(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == size1
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + size1
        assert Block_Eq_(block2, size1, data1) is True
        block2 = Block_Free(block2)

        block1 = Block_Free(block1)
        block1 = Block_Create(0x1234, size2, data2)

        block2 = Block_Create(0x1234, 0, NULL)
        block2 = Block_ExtendLeft(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == size2
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + size2
        assert Block_Eq_(block2, size2, data2) is True
        block2 = Block_Free(block2)

        block2 = Block_Create(0x1234, size1, data1)
        block2 = Block_ExtendLeft(block2, block1)
        assert block2 != NULL
        assert block2.address == 0x1234
        assert Block_Length(block2) == size2 + size1
        assert Block_Start(block2) == 0x1234
        assert Block_Endex(block2) == 0x1234 + size2 + size1
        assert Block_Eq_(block2, size2 + size1, data2 + data1) is True
        block2 = Block_Free(block2)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_RotateLeft__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i

    try:
        block = Block_Create(0x1234, 0, NULL)
        Block_RotateLeft__(block, 3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        Block_RotateLeft__(block, 0)
        assert block.address == 0x1234
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert Block_Eq_(block, size, data) is True
        block = Block_Free(block)

        for i in range(size):
            block = Block_Create(0x1234, size, data)
            Block_RotateLeft__(block, i)
            assert block.address == 0x1234
            assert Block_Length(block) == size
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + size
            assert Block_Eq_(block, size, data[i:] + data[:i]) is True
            block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_RotateLeft_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i
        size_t k

    try:
        block = Block_Create(0x1234, 0, NULL)
        Block_RotateLeft_(block, 3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        Block_RotateLeft_(block, 0)
        assert block.address == 0x1234
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert Block_Eq_(block, size, data) is True
        block = Block_Free(block)

        for i in range(size * 3):
            block = Block_Create(0x1234, size, data)
            Block_RotateLeft_(block, i)
            assert block.address == 0x1234
            assert Block_Length(block) == size
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + size
            k = i % size
            assert Block_Eq_(block, size, data[k:] + data[:k]) is True
            block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_RotateRight__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i
        ssize_t k

    try:
        block = Block_Create(0x1234, 0, NULL)
        Block_RotateRight__(block, 3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        Block_RotateRight__(block, 0)
        assert block.address == 0x1234
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert Block_Eq_(block, size, data) is True
        block = Block_Free(block)

        for i in range(size):
            block = Block_Create(0x1234, size, data)
            Block_RotateRight__(block, i)
            assert block.address == 0x1234
            assert Block_Length(block) == size
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + size
            k = -<ssize_t>i
            assert Block_Eq_(block, size, data[k:] + data[:k]) is True
            block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_RotateRight_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i
        ssize_t k

    try:
        block = Block_Create(0x1234, 0, NULL)
        Block_RotateRight_(block, 3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        Block_RotateRight_(block, 0)
        assert block.address == 0x1234
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert Block_Eq_(block, size, data) is True
        block = Block_Free(block)

        for i in range(size * 3):
            block = Block_Create(0x1234, size, data)
            Block_RotateRight_(block, i)
            assert block.address == 0x1234
            assert Block_Length(block) == size
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + size
            k = -<ssize_t>(i % size)
            assert Block_Eq_(block, size, data[k:] + data[:k]) is True
            block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Rotate():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t i
        ssize_t k

    try:
        block = Block_Create(0x1234, 0, NULL)
        Block_Rotate(block, 3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        Block_Rotate(block, -3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        Block_Rotate(block, 0)
        assert block.address == 0x1234
        assert Block_Length(block) == size
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234 + size
        assert Block_Eq_(block, size, data) is True
        block = Block_Free(block)

        for i in range(0, -<ssize_t>size * 3, -1):
            block = Block_Create(0x1234, size, data)
            Block_Rotate(block, i)
            assert block.address == 0x1234
            assert Block_Length(block) == size
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + size
            k = -i % <ssize_t>size
            assert Block_Eq_(block, size, data[k:] + data[:k]) is True
            block = Block_Free(block)

        for i in range(<ssize_t>size * 3):
            block = Block_Create(0x1234, size, data)
            Block_Rotate(block, i)
            assert block.address == 0x1234
            assert Block_Length(block) == size
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + size
            k = -(i % <ssize_t>size)
            assert Block_Eq_(block, size, data[k:] + data[:k]) is True
            block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Repeat():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i

    try:
        block = Block_Create(0x1234, 0, NULL)
        Block_Repeat(block, 3)
        assert block.address == 0x1234
        assert Block_Length(block) == 0
        assert Block_Start(block) == 0x1234
        assert Block_Endex(block) == 0x1234
        assert Block_Eq_(block, 0, NULL) is True
        block = Block_Free(block)

        for i in range(4):
            block = Block_Create(0x1234, size, data)
            block = Block_Repeat(block, i)
            assert block != NULL
            assert block.address == 0x1234
            assert Block_Length(block) == (size * i)
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + (size * i)
            assert Block_Eq_(block, (size * i), (data * i)) is True
            block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_RepeatToSize():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t i

    try:
        for i in range(size * 4):
            block = Block_Create(0x1234, size, data)
            block = Block_RepeatToSize(block, i)
            assert block != NULL
            assert block.address == 0x1234
            assert Block_Length(block) == i
            assert Block_Start(block) == 0x1234
            assert Block_Endex(block) == 0x1234 + i
            assert Block_Eq_(block, i, (data * ((i + size) // size))[:i]) is True
            block = Block_Free(block)

        block = Block_Create(0x1234, 0, NULL)
        for i in range(4):
            with pytest.raises(RuntimeError, match='empty'):
                block = Block_RepeatToSize(block, i)
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Read_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        bytearray buffer

    try:
        block = Block_Create(0x1234, size, data)

        for start in range(size):
            for endex in range(start, size):
                buffer = bytearray(endex - start)
                Block_Read_(block, start, endex - start, buffer)
                assert buffer == data[start:endex]

        with pytest.raises(OverflowError, match='size overflow'):
            Block_Read_(block, 0, SIZE_MAX, NULL)

        with pytest.raises(OverflowError):
            Block_Read_(block, SIZE_MAX, 1, NULL)

        with pytest.raises(IndexError, match='index out of range'):
            Block_Read_(block, SIZE_HMAX, 1, NULL)

        with pytest.raises(IndexError, match='index out of range'):
            Block_Read_(block, size, 1, NULL)

    finally:
        block = Block_Free(block)


def test_Block_Write_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        bytes buffer

    try:
        for start in range(size):
            for endex in range(start, size):
                block = Block_Create(0x1234, 0, NULL)
                block = Block_Write_(block, 0, endex - start, data)
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == endex - start
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + endex - start
                assert Block_Eq_(block, endex - start, data) is True
                block = Block_Free(block)

        for start in range(size):
            for endex in range(start, start + size):
                block = Block_Create(0x1234, size, data)
                block = Block_Write_(block, start, endex - start, data)
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == max(size, endex)
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + max(size, endex)
                buffer = data[:start] + data[:endex - start] + data[endex:]
                assert Block_Eq_(block, max(size, endex), buffer) is True
                block = Block_Free(block)

        block = Block_Create(0x1234, size, data)

        with pytest.raises(OverflowError):
            Block_Write_(block, SIZE_MAX, 1, NULL)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_Write_(block, SIZE_MAX - size, 1, NULL)

        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ReadSlice_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        bytearray buffer
        size_t num

    try:
        block = Block_Create(0x1234, size, data)

        for start in range(size):
            for endex in range(start + size):
                buffer = bytearray(endex - start if start < endex else 0)
                Block_ReadSlice_(block, start, endex, &num, buffer)
                assert num == <size_t>len(data[start:endex])
                assert buffer[:num] == data[start:endex]

        with pytest.raises(OverflowError, match='size overflow'):
            Block_ReadSlice_(block, SIZE_MAX, SIZE_MAX, &num, NULL)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_ReadSlice_(block, 0, SIZE_MAX, &num, NULL)

    finally:
        block = Block_Free(block)


def test_Block_ReadSlice():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        bytearray buffer
        size_t num

    try:
        block = Block_Create(0x1234, size, data)

        for start in range(-<ssize_t>size, <ssize_t>size):
            for endex in range(-<ssize_t>size, <ssize_t>(start + size)):
                buffer = bytearray(size)
                Block_ReadSlice(block, start, endex, &num, buffer)
                assert num == <size_t>len(data[start:endex])
                assert buffer[:num] == data[start:endex]

    finally:
        block = Block_Free(block)


def test_Block_GetSlice_():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        bytes buffer

    try:
        block1 = Block_Create(0x1234, size, data)

        for start in range(size):
            for endex in range(start + size):
                block2 = Block_GetSlice_(block1, start, endex)
                buffer = data[start:endex]
                assert block2 != NULL
                assert block2.address == 0x1234 + start
                assert Block_Length(block2) == <size_t>len(buffer)
                assert Block_Start(block2) == 0x1234 + start
                assert Block_Endex(block2) == 0x1234 + start + <size_t>len(buffer)
                assert Block_Eq_(block2, <size_t>len(buffer), buffer) is True
                block2 = Block_Free(block2)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_GetSlice_(block1, SIZE_MAX, SIZE_MAX)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_GetSlice_(block1, 0, SIZE_MAX)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_GetSlice():
    cdef:
        Block_* block1 = NULL
        Block_* block2 = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        bytes buffer

    try:
        block1 = Block_Create(0x1234, size, data)

        for start in range(-<ssize_t>size, <ssize_t>size):
            for endex in range(-<ssize_t>size, <ssize_t>(start + size)):
                start_, endex_ = start, endex
                block2 = Block_GetSlice(block1, start, endex)
                buffer = data[start:endex]
                start = start % <ssize_t>size
                assert block2 != NULL
                assert block2.address == 0x1234 + <size_t>start
                assert Block_Length(block2) == <size_t>len(buffer)
                assert Block_Start(block2) == 0x1234 + <size_t>start
                assert Block_Endex(block2) == 0x1234 + <size_t>start + <size_t>len(buffer)
                assert Block_Eq_(block2, <size_t>len(buffer), buffer) is True
                block2 = Block_Free(block2)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_WriteSlice_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        size_t num
        bytearray buffer

    try:
        for start in range(size):
            for endex in range(size):
                for num in range(size):
                    block = Block_Create(0x1234, 0, NULL)
                    block = Block_WriteSlice_(block, start, endex, num, data)
                    buffer = bytearray()
                    buffer[start:endex] = data[:num]
                    assert block != NULL
                    assert block.address == 0x1234
                    assert Block_Length(block) == <size_t>len(buffer)
                    assert Block_Start(block) == 0x1234
                    assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                    assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                    block = Block_Free(block)

        for start in range(size + size):
            for endex in range(size + size):
                for num in range(size):
                    block = Block_Create(0x1234, size, data)
                    block = Block_WriteSlice_(block, start, endex, num, data)
                    buffer = bytearray(data)
                    buffer[start:endex] = data[:num]
                    assert block != NULL
                    assert block.address == 0x1234
                    assert Block_Length(block) == <size_t>len(buffer)
                    assert Block_Start(block) == 0x1234
                    assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                    assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                    block = Block_Free(block)

        block = Block_Create(0x1234, size, data)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_WriteSlice_(block, SIZE_HMAX + 1, 0, 1, NULL)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_WriteSlice_(block, 0, SIZE_HMAX + 1, 1, NULL)

        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_WriteSlice():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        size_t num
        bytearray buffer

    try:
        for start in range(-<ssize_t>size, <ssize_t>size):
            for endex in range(-<ssize_t>size, <ssize_t>size):
                for num in range(size):
                    block = Block_Create(0x1234, 0, NULL)
                    block = Block_WriteSlice(block, start, endex, num, data)
                    buffer = bytearray()
                    buffer[start:endex] = data[:num]
                    assert block != NULL
                    assert block.address == 0x1234
                    assert Block_Length(block) == <size_t>len(buffer)
                    assert Block_Start(block) == 0x1234
                    assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                    assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                    block = Block_Free(block)

        for start in range(-<ssize_t>(size + size), <ssize_t>(size + size)):
            for endex in range(-<ssize_t>size, <ssize_t>size):
                for num in range(size):
                    block = Block_Create(0x1234, size, data)
                    block = Block_WriteSlice(block, start, endex, num, data)
                    buffer = bytearray(data)
                    buffer[start:endex] = data[:num]
                    assert block != NULL
                    assert block.address == 0x1234
                    assert Block_Length(block) == <size_t>len(buffer)
                    assert Block_Start(block) == 0x1234
                    assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                    assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                    block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_SetSlice_():
    cdef:
        Block_* block1 = NULL
        bytes data1 = DATA1
        size_t size1 = SIZE1
        size_t start1
        size_t endex1

        Block_* block2 = NULL
        bytes data2 = DATA2
        size_t size2 = SIZE2
        size_t start2
        size_t endex2

        bytearray buffer

    try:
        block2 = Block_Create(0, 0, NULL)

        for start1 in range(size1):
            for endex1 in range(size1):
                for start2 in range(size2):
                    for endex2 in range(size2):
                        block1 = Block_Create(0x1234, 0, NULL)
                        block1 = Block_SetSlice_(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray()
                        buffer[start1:endex1] = b''[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        for start1 in range(size1):
            for endex1 in range(size1 + size1):
                for start2 in range(size2):
                    for endex2 in range(size2 + size2):
                        block1 = Block_Create(0x1234, size1, data1)
                        block1 = Block_SetSlice_(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray(data1)
                        buffer[start1:endex1] = b''[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        block2 = Block_Free(block2)
        block2 = Block_Create(0, size2, data2)

        for start1 in range(size1):
            for endex1 in range(size1):
                for start2 in range(size2):
                    for endex2 in range(size2):
                        block1 = Block_Create(0x1234, 0, NULL)
                        block1 = Block_SetSlice_(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray()
                        buffer[start1:endex1] = data2[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        for start1 in range(size1):
            for endex1 in range(size1 + size1):
                for start2 in range(size2):
                    for endex2 in range(size2 + size2):
                        block1 = Block_Create(0x1234, size1, data1)
                        block1 = Block_SetSlice_(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray(data1)
                        buffer[start1:endex1] = data2[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        block1 = Block_Free(block1)
        block1 = Block_Create(0, size1, data1)
        block2 = Block_Free(block2)
        block2 = Block_Create(0, size2, data2)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_SetSlice_(block1, 0, 0, block2, SIZE_HMAX + 1, 0)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_SetSlice_(block1, 0, 0, block2, 0, SIZE_HMAX + 1)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_SetSlice_(block1, SIZE_HMAX + 1, 0, block2, 0, 0)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_SetSlice_(block1, 0, SIZE_HMAX + 1, block2, 0, 0)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_SetSlice():
    cdef:
        Block_* block1 = NULL
        bytes data1 = DATA1
        size_t size1 = SIZE1
        ssize_t start1
        ssize_t endex1

        Block_* block2 = NULL
        bytes data2 = DATA2
        size_t size2 = SIZE2
        ssize_t start2
        ssize_t endex2

        bytearray buffer

    try:
        block2 = Block_Create(0, 0, NULL)

        for start1 in range(-<ssize_t>size1, <ssize_t>size1):
            for endex1 in range(-<ssize_t>size1, <ssize_t>size1):
                for start2 in range(-<ssize_t>size2, <ssize_t>size2):
                    for endex2 in range(-<ssize_t>size2, <ssize_t>size2):
                        block1 = Block_Create(0x1234, 0, NULL)
                        block1 = Block_SetSlice(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray()
                        buffer[start1:endex1] = b''[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        for start1 in range(-<ssize_t>size1, <ssize_t>size1):
            for endex1 in range(-<ssize_t>size1, <ssize_t>(size1 + size1)):
                for start2 in range(-<ssize_t>size2, <ssize_t>size2):
                    for endex2 in range(-<ssize_t>size2, <ssize_t>(size2 + size2)):
                        block1 = Block_Create(0x1234, size1, data1)
                        block1 = Block_SetSlice(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray(data1)
                        buffer[start1:endex1] = b''[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        block2 = Block_Free(block2)
        block2 = Block_Create(0, size2, data2)

        for start1 in range(-<ssize_t>size1, <ssize_t>size1):
            for endex1 in range(-<ssize_t>size1, <ssize_t>size1):
                for start2 in range(-<ssize_t>size2, <ssize_t>size2):
                    for endex2 in range(-<ssize_t>size2, <ssize_t>size2):
                        block1 = Block_Create(0x1234, 0, NULL)
                        block1 = Block_SetSlice(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray()
                        buffer[start1:endex1] = data2[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

        for start1 in range(-<ssize_t>size1, <ssize_t>size1):
            for endex1 in range(-<ssize_t>size1, <ssize_t>(size1 + size1)):
                for start2 in range(-<ssize_t>size2, <ssize_t>size2):
                    for endex2 in range(-<ssize_t>size2, <ssize_t>(size2 + size2)):
                        block1 = Block_Create(0x1234, size1, data1)
                        block1 = Block_SetSlice(block1, start1, endex1, block2, start2, endex2)
                        buffer = bytearray(data1)
                        buffer[start1:endex1] = data2[start2:endex2]
                        assert block1 != NULL
                        assert block1.address == 0x1234
                        assert Block_Length(block1) == <size_t>len(buffer)
                        assert Block_Start(block1) == 0x1234
                        assert Block_Endex(block1) == 0x1234 + <size_t>len(buffer)
                        assert Block_Eq_(block1, <size_t>len(buffer), buffer) is True
                        block1 = Block_Free(block1)

    finally:
        block1 = Block_Free(block1)
        block2 = Block_Free(block2)


def test_Block_DelSlice_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        size_t start
        size_t endex
        bytearray buffer

    try:
        for start in range(size):
            for endex in range(size):
                block = Block_Create(0x1234, 0, NULL)
                block = Block_DelSlice_(block, start, endex)
                buffer = bytearray()
                del buffer[start:endex]
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == <size_t>len(buffer)
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                block = Block_Free(block)

        for start in range(size + size):
            for endex in range(size + size):
                block = Block_Create(0x1234, size, data)
                block = Block_DelSlice_(block, start, endex)
                buffer = bytearray(data)
                del buffer[start:endex]
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == <size_t>len(buffer)
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                block = Block_Free(block)

        block = Block_Create(0x1234, size, data)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_DelSlice_(block, SIZE_HMAX + 1, 0)

        with pytest.raises(OverflowError, match='size overflow'):
            Block_DelSlice_(block, 0, SIZE_HMAX + 1)

        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_DelSlice():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        bytearray buffer

    try:
        for start in range(-<ssize_t>size, <ssize_t>size):
            for endex in range(-<ssize_t>size, <ssize_t>size):
                block = Block_Create(0x1234, 0, NULL)
                block = Block_DelSlice(block, start, endex)
                buffer = bytearray()
                del buffer[start:endex]
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == <size_t>len(buffer)
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                block = Block_Free(block)

        for start in range(-<ssize_t>(size + size), <ssize_t>(size + size)):
            for endex in range(-<ssize_t>size, <ssize_t>size):
                block = Block_Create(0x1234, size, data)
                block = Block_DelSlice(block, start, endex)
                buffer = bytearray(data)
                del buffer[start:endex]
                assert block != NULL
                assert block.address == 0x1234
                assert Block_Length(block) == <size_t>len(buffer)
                assert Block_Start(block) == 0x1234
                assert Block_Endex(block) == 0x1234 + <size_t>len(buffer)
                assert Block_Eq_(block, <size_t>len(buffer), buffer) is True
                block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Bytes():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        bytes buffer

    try:
        block = Block_Create(0x1234, 0, NULL)
        buffer = Block_Bytes(block)
        assert buffer == b''
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        buffer = Block_Bytes(block)
        assert buffer == data
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_Bytearray():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        bytearray buffer

    try:
        block = Block_Create(0x1234, 0, NULL)
        buffer = Block_Bytearray(block)
        assert buffer == b''
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        buffer = Block_Bytearray(block)
        assert buffer == data
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_View():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        bytearray buffer
        BlockView view

    try:
        block = Block_Create(0x1234, 0, NULL)
        view = Block_View(block)
        assert bytes(view) == b''
        view.dispose()
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        assert bytes(view) == data
        view.dispose()
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ViewSlice_():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view
        size_t start
        size_t endex

    try:
        for start in range(size):
            for endex in range(size + size):
                block = Block_Create(0x1234, 0, NULL)
                view = Block_ViewSlice_(block, start, endex)
                assert bytes(view) == b''
                view.dispose()
                block = Block_Free(block)

        for start in range(size):
            for endex in range(size + size):
                block = Block_Create(0x1234, size, data)
                view = Block_ViewSlice_(block, start, endex)
                assert bytes(view) == data[start:endex]
                view.dispose()
                block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_Block_ViewSlice():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view
        ssize_t start
        ssize_t endex

    try:
        for start in range(-<ssize_t>size, <ssize_t>(size + size)):
            for endex in range(-<ssize_t>size, <ssize_t>(size + size)):
                block = Block_Create(0x1234, 0, NULL)
                view = Block_ViewSlice(block, start, endex)
                assert bytes(view) == b''
                view.dispose()
                block = Block_Free(block)

        for start in range(-<ssize_t>size, <ssize_t>(size + size)):
            for endex in range(-<ssize_t>size, <ssize_t>(size + size)):
                block = Block_Create(0x1234, size, data)
                view = Block_ViewSlice(block, start, endex)
                assert bytes(view) == data[start:endex]
                view.dispose()
                block = Block_Free(block)

    finally:
        block = Block_Free(block)


# =====================================================================================================================

def test_BlockView_basics():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view

    try:
        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        assert view is not None
        assert view._block != NULL
        assert view._start == block.start
        assert view._endex == block.endex
        assert view._memview is None
        assert bytes(view) == data
        assert view._memview is not None
        assert bytes(memoryview(view)) == data
        view.dispose()
        assert view._block == NULL
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_BlockView___bool__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view

    try:
        block = Block_Create(0x1234, 0, NULL)
        view = Block_View(block)
        assert not bool(view)
        view.dispose()
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        assert bool(view)
        view.dispose()
        with pytest.raises(RuntimeError, match='null internal data pointer'):
            bool(view)

    finally:
        block = Block_Free(block)


def test_BlockView___bytes__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view
        object memview

    try:
        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        memview = view.memview
        assert memview is not None
        assert bytes(memview) == data
        view.dispose()
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_BlockView_memview():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view
        object memview

    try:
        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        memview = view.memview
        assert memview is not None
        assert bytes(memview) == data
        view.dispose()
        with pytest.raises(RuntimeError, match='null internal data pointer'):
            bool(view.memview)

    finally:
        block = Block_Free(block)


def test_BlockView___len__():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view

    try:
        block = Block_Create(0x1234, 0, NULL)
        view = Block_View(block)
        assert <size_t>len(view) == 0
        view.dispose()
        block = Block_Free(block)

        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        assert <size_t>len(view) == size
        view.dispose()
        with pytest.raises(RuntimeError, match='null internal data pointer'):
            <size_t>len(view)

    finally:
        block = Block_Free(block)


def test_BlockView_bounds():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view

    try:
        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        assert view.start == Block_Start(block)
        assert view.endex == Block_Endex(block)
        assert view.endin == <object>Block_Endex(block) - 1
        view.dispose()
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_BlockView_check():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view

    try:
        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        view.check()
        view.dispose()
        with pytest.raises(RuntimeError, match='null internal data pointer'):
            view.check()
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


def test_BlockView_dispose():
    cdef:
        Block_* block = NULL
        bytes data = DATA1
        size_t size = SIZE1
        BlockView view

    try:
        block = Block_Create(0x1234, size, data)
        view = Block_View(block)
        assert view._block != NULL
        assert view.acquired == True
        view.dispose()
        assert view._block == NULL
        assert view.acquired == False
        view.dispose()
        assert view._block == NULL
        assert view.acquired == False
        block = Block_Free(block)

    finally:
        block = Block_Free(block)


# =====================================================================================================================

cdef tuple TEMPLATE_BLOCKS = (
    (2, b'234'),
    (8, b'89A'),
    (12, b'C'),
    (14, b'EF'),
    (18, b'I'),
)

cdef tuple HELLO_WORLD_BLOCKS = (
    (2, b'Hello'),
    (10, b'World!'),
)


cdef Rack_* create_rack(tuple template):
    cdef:
        Rack_* blocks = NULL
        Block_* block = NULL
        size_t block_count = <size_t>len(template)
        size_t block_index
        addr_t start
        bytes data

    try:
        blocks = Rack_Alloc(block_count)

        for block_index in range(block_count):
            start, data = template[block_index]
            block = Block_Create(start, <size_t>len(data), data)
            Rack_Set__(blocks, block_index, block)
            block = NULL

        return blocks

    except:
        block = Block_Free(block)
        blocks = Rack_Free(blocks)
        raise


cdef Rack_* create_template_rack():
    return create_rack(TEMPLATE_BLOCKS)


cdef Rack_* create_hello_world_rack():
    return create_rack(HELLO_WORLD_BLOCKS)


cdef bint check_null_rack(const Rack_* blocks) except -1:
    cdef:
        size_t offset

    for offset in range(Rack_Length(blocks)):
        assert Rack_Get__(blocks, offset) == NULL


# ---------------------------------------------------------------------------------------------------------------------

def test_Rack_Alloc_Free():
    cdef:
        Rack_* blocks = NULL
        size_t block_index

    try:
        blocks = Rack_Alloc(0)
        assert blocks != NULL
        assert blocks.allocated == MARGIN + MARGIN
        assert blocks.start == MARGIN
        assert blocks.endex == MARGIN
        assert Rack_Length(blocks) == 0
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0x100)
        assert blocks != NULL
        assert blocks.allocated == MARGIN + 0x100 + MARGIN
        assert blocks.start == MARGIN
        assert blocks.endex == MARGIN + 0x100
        assert Rack_Length(blocks) == 0x100
        for block_index in range(0x100):
            assert Rack_Get__(blocks, block_index) == NULL
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_ShallowCopy():
    cdef:
        Rack_* blocks = NULL
        Rack_* temp = NULL
        size_t size
        size_t offset

    try:
        temp = Rack_Alloc(0)
        size = Rack_Length(temp)

        blocks = Rack_ShallowCopy(temp)
        assert blocks != NULL
        assert blocks != temp
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        temp = Rack_Free(temp)
        temp = create_template_rack()
        size = Rack_Length(temp)

        blocks = Rack_ShallowCopy(temp)
        assert blocks != NULL
        assert blocks != temp
        assert all(Rack_Get__(blocks, offset) == Rack_Get__(temp, offset) for offset in range(size)) is True
        assert all(Rack_Get__(blocks, offset).references == 2 for offset in range(size)) is True
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)
        temp = Rack_Free(temp)


def test_Rack_Copy():
    cdef:
        Rack_* blocks = NULL
        Rack_* temp = NULL
        size_t size
        size_t offset

    try:
        temp = Rack_Alloc(0)
        size = Rack_Length(temp)

        blocks = Rack_Copy(temp)
        assert blocks != NULL
        assert blocks != temp
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        temp = Rack_Free(temp)
        temp = create_template_rack()
        size = Rack_Length(temp)

        blocks = Rack_Copy(temp)
        assert blocks != NULL
        assert blocks != temp
        assert all(Rack_Get__(blocks, offset) != Rack_Get__(temp, offset) for offset in range(size)) is True
        assert all(Rack_Get__(blocks, offset).references == 1 for offset in range(size)) is True
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)
        temp = Rack_Free(temp)


def test_Rack_FromObject():
    cdef:
        Rack_* blocks = NULL
        Rack_* temp = NULL
        size_t size
        size_t offset

    try:
        temp = Rack_Alloc(0)
        size = Rack_Length(temp)

        blocks = Rack_FromObject((), 0)
        assert blocks != NULL
        assert blocks != temp
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        blocks = Rack_FromObject([], 0)
        assert blocks != NULL
        assert blocks != temp
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        blocks = Rack_FromObject(iter(()), 0)
        assert blocks != NULL
        assert blocks != temp
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        temp = Rack_Free(temp)
        temp = create_template_rack()
        size = Rack_Length(temp)

        blocks = Rack_FromObject(tuple(TEMPLATE_BLOCKS), 0)
        assert blocks != temp
        assert all(Rack_Get__(blocks, offset) != Rack_Get__(temp, offset) for offset in range(size)) is True
        assert all(Rack_Get__(blocks, offset).references == 1 for offset in range(size)) is True
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        blocks = Rack_FromObject(list(TEMPLATE_BLOCKS), 0)
        assert blocks != temp
        assert all(Rack_Get__(blocks, offset) != Rack_Get__(temp, offset) for offset in range(size)) is True
        assert all(Rack_Get__(blocks, offset).references == 1 for offset in range(size)) is True
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        blocks = Rack_FromObject(iter(TEMPLATE_BLOCKS), 0)
        assert blocks != temp
        assert all(Rack_Get__(blocks, offset) != Rack_Get__(temp, offset) for offset in range(size)) is True
        assert all(Rack_Get__(blocks, offset).references == 1 for offset in range(size)) is True
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

        blocks = Rack_FromObject(iter(TEMPLATE_BLOCKS), 69)
        assert blocks != temp
        assert all(Rack_Get__(blocks, offset) != Rack_Get__(temp, offset) for offset in range(size)) is True
        assert all(Rack_Get__(blocks, offset).references == 1 for offset in range(size)) is True
        for offset in range(size): Rack_Get_(blocks, offset).address -= 69
        assert Rack_Eq(blocks, temp) is True
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)
        temp = Rack_Free(temp)


def test_Rack_BoundSlice():
    cdef:
        Rack_* blocks = NULL
        addr_t start
        addr_t endex

    try:
        blocks = Rack_Alloc(0)

        start, endex = Rack_BoundSlice(blocks, 0, 0)
        assert start == 0
        assert endex == 0

        start, endex = Rack_BoundSlice(blocks, 0, ADDR_MAX)
        assert start == 0
        assert endex == 0

        start, endex = Rack_BoundSlice(blocks, ADDR_MAX, ADDR_MAX)
        assert start == 0
        assert endex == 0

        start, endex = Rack_BoundSlice(blocks, ADDR_MAX, 0)
        assert start == 0
        assert endex == 0

        blocks = Rack_Free(blocks)
        blocks = create_template_rack()

        start, endex = Rack_BoundSlice(blocks, 0, 0)
        assert start == Block_Start(Rack_First_(blocks))
        assert endex == start

        start, endex = Rack_BoundSlice(blocks, 0, ADDR_MAX)
        assert start == Block_Start(Rack_First_(blocks))
        assert endex == Block_Endex(Rack_Last_(blocks))

        start, endex = Rack_BoundSlice(blocks, ADDR_MAX, ADDR_MAX)
        assert start == Block_Endex(Rack_Last_(blocks))
        assert endex == start

        start, endex = Rack_BoundSlice(blocks, ADDR_MAX, 0)
        assert start == Block_Endex(Rack_Last_(blocks))
        assert endex == start

    finally:
        blocks = Rack_Free(blocks)


# TODO: def test_Rack_Shift_()


# TODO: def test_Rack_Shift()


def test_Rack_Eq():
    cdef:
        Rack_* blocks1 = NULL
        Rack_* blocks2 = NULL
        size_t block_index

    try:
        blocks2 = create_template_rack()

        blocks1 = create_template_rack()
        assert Rack_Eq(blocks1, blocks2) is True
        blocks1 = Rack_Free(blocks1)

        for block_index in range(Rack_Length(blocks2)):
            blocks1 = create_template_rack()
            blocks1 = Rack_Pop_(blocks1, block_index, NULL)
            assert Rack_Eq(blocks1, blocks2) is False
            blocks1 = Rack_Free(blocks1)

    finally:
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)


def test_Rack_Reserve_():
    cdef:
        Rack_* blocks = NULL
        size_t size = 10
        list data
        size_t offset

    try:
        blocks = Rack_Alloc(0)
        blocks = Rack_Reserve_(blocks, 0, 0)  # same
        assert blocks != NULL
        assert Rack_Length(blocks) == 0
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        blocks = Rack_Reserve_(blocks, 0, 3)
        assert blocks != NULL
        assert Rack_Length(blocks) == 3 + 0
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, 0, 0)  # same
        assert blocks != NULL
        assert Rack_Length(blocks) == size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, 0, 3)  # before
        assert blocks != NULL
        assert Rack_Length(blocks) == 3 + size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, size, 3)  # after
        assert blocks != NULL
        assert Rack_Length(blocks) == size + 3
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        offset = size >> 1  # half
        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, offset, 3)
        assert blocks != NULL
        assert Rack_Length(blocks) == size + 3
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        offset = (size * 1) >> 2  # first quarter
        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, offset, 3)
        assert blocks != NULL
        assert Rack_Length(blocks) == size + 3
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        offset = (size * 3) >> 2  # third quarter
        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, offset, 3)
        assert blocks != NULL
        assert Rack_Length(blocks) == size + 3
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, 0, size)  # before
        assert blocks != NULL
        assert Rack_Length(blocks) == size + size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, size, size)  # after
        assert blocks != NULL
        assert Rack_Length(blocks) == size + size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        offset = size >> 1  # half
        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, offset, size)
        assert blocks != NULL
        assert Rack_Length(blocks) == size + size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        offset = (size * 1) >> 2  # first quarter
        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, offset, size)
        assert blocks != NULL
        assert Rack_Length(blocks) == size + size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        offset = (size * 3) >> 2  # third quarter
        blocks = Rack_Alloc(size)
        blocks = Rack_Reserve_(blocks, offset, size)
        assert blocks != NULL
        assert Rack_Length(blocks) == size + size
        check_null_rack(blocks)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        with pytest.raises(OverflowError, match='size overflow'): Rack_Reserve_(blocks, 0, SIZE_MAX)
        with pytest.raises(OverflowError, match='size overflow'): Rack_Reserve_(blocks, 0, SIZE_HMAX)
        with pytest.raises(IndexError, match='index out of range'): Rack_Reserve_(blocks, size + 1, 1)
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_Delete_():
    cdef:
        Rack_* blocks = NULL
        size_t size = 10
        size_t start
        size_t endex

    try:
        start = endex = 0
        blocks = Rack_Alloc(0)
        blocks = Rack_Delete_(blocks, 0, 0)
        assert blocks != NULL
        assert Rack_Length(blocks) == 0
        blocks = Rack_Free(blocks)

        for start in range(size):
            for endex in range(start, size):
                blocks = Rack_Alloc(size)
                blocks = Rack_Delete_(blocks, start, endex - start)
                assert blocks != NULL
                assert Rack_Length(blocks) == size - (endex - start)
                check_null_rack(blocks)
                blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(size)
        with pytest.raises(OverflowError, match='size overflow'): Rack_Delete_(blocks, 0, SIZE_MAX)
        with pytest.raises(IndexError, match='index out of range'): Rack_Delete_(blocks, size + 1, 1)
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_Clear():
    cdef:
        Rack_* blocks = NULL

    try:
        blocks = Rack_Alloc(0)
        blocks = Rack_Clear(blocks)
        assert blocks != NULL
        assert blocks.allocated == MARGIN + MARGIN
        assert blocks.start == MARGIN
        assert blocks.endex == MARGIN
        assert Rack_Length(blocks) == 0
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        blocks = Rack_Clear(blocks)
        assert blocks != NULL
        assert Rack_Length(blocks) == 0
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_At_First_Last():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        size_t offset

    try:
        blocks = create_template_rack()

        assert Rack_First__(blocks) == blocks.blocks[blocks.start]
        assert Rack_First_(blocks) == blocks.blocks[blocks.start]

        assert Rack_Last__(blocks) == blocks.blocks[blocks.endex - 1]
        assert Rack_Last_(blocks) == blocks.blocks[blocks.endex - 1]

        for offset in range(size):
            assert Rack_At__(blocks, offset) == <const Block_**>&blocks.blocks[blocks.start + offset]
            assert Rack_At_(blocks, offset) == &blocks.blocks[blocks.start + offset]

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_Get_():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        size_t offset
        str match = 'index out of range'

    try:
        blocks = create_template_rack()
        for offset in range(size):
            assert Rack_Get(blocks, offset) == blocks.blocks[blocks.start + offset]
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Get(blocks,  size + 1)
        with pytest.raises(IndexError, match=match): Rack_Get(blocks,  99)
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_Get():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        ssize_t offset
        str match = 'index out of range'

    try:
        blocks = create_template_rack()
        for offset in range(<ssize_t>size):
            assert Rack_Get(blocks, offset) == blocks.blocks[blocks.start + <size_t>offset]
        for offset in range(-<ssize_t>size, 0):
            assert Rack_Get(blocks, offset) == blocks.blocks[blocks.start + size - <size_t>-offset]
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Get(blocks,  <ssize_t>size + 1)
        with pytest.raises(IndexError, match=match): Rack_Get(blocks,  99)
        with pytest.raises(IndexError, match=match): Rack_Get(blocks, -<ssize_t>size - 1)
        with pytest.raises(IndexError, match=match): Rack_Get(blocks, -99)
        blocks = Rack_Free(blocks)

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_Set_():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        size_t offset
        size_t offset2
        Block_* backup = NULL
        Block_* backup2 = NULL
        Block_* block = NULL
        str match = 'index out of range'

    try:
        for offset in range(size):
            blocks = create_template_rack()
            backup2 = Rack_Get__(blocks, offset)
            block = Block_Copy(backup2)
            for offset2 in range(block.start, block.endex):
                block.data[offset2] ^= 0xFF
            Rack_Set_(blocks, offset, block, &backup)
            assert backup == backup2
            assert Rack_Get__(blocks, offset) == block
            block = NULL  # invalidate pointer
            backup = Block_Free(backup)
            blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Set_(blocks,  size + 1, NULL, NULL)
        with pytest.raises(IndexError, match=match): Rack_Set_(blocks,  99, NULL, NULL)
        blocks = Rack_Free(blocks)

    finally:
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)


def test_Rack_Set():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        ssize_t offset
        size_t offset2
        Block_* backup = NULL
        Block_* backup2 = NULL
        Block_* block = NULL
        str match = 'index out of range'

    try:
        for offset in range(-<ssize_t>size, <ssize_t>size):
            blocks = create_template_rack()
            backup2 = Rack_Get(blocks, offset)
            block = Block_Copy(backup2)
            for offset2 in range(block.start, block.endex):
                block.data[offset2] ^= 0xFF
            Rack_Set(blocks, offset, block, &backup)
            assert backup == backup2
            assert Rack_Get(blocks, offset) == block
            block = NULL  # invalidate pointer
            backup = Block_Free(backup)
            blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Set(blocks,  <ssize_t>size + 1, NULL, NULL)
        with pytest.raises(IndexError, match=match): Rack_Set(blocks,  99, NULL, NULL)
        with pytest.raises(IndexError, match=match): Rack_Set(blocks, -<ssize_t>size - 1, NULL, NULL)
        with pytest.raises(IndexError, match=match): Rack_Set(blocks, -99, NULL, NULL)
        blocks = Rack_Free(blocks)

    finally:
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)


def test_Rack_Pop__():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'pop index out of range'
        Rack_* temp = NULL
        Block_* backup = NULL
        Block_* backup2 = NULL

    try:
        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, size - 1)
        blocks = Rack_Pop__(blocks, &backup)
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:-1])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        with pytest.raises(IndexError, match=match): Rack_Pop__(blocks, NULL)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)


def test_Rack_Pop_():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'pop index out of range'
        Rack_* temp = NULL
        Block_* backup = NULL
        Block_* backup2 = NULL
        size_t offset

    try:
        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, size - 1)
        blocks = Rack_Pop_(blocks, size - 1, &backup)  # final
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:-1])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, 0)
        blocks = Rack_Pop_(blocks, 0, &backup)  # initial
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        offset = size >> 1  # half
        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, offset)
        blocks = Rack_Pop_(blocks, offset, &backup)
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:offset] + TEMPLATE_BLOCKS[offset + 1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        with pytest.raises(IndexError, match=match): Rack_Pop_(blocks, 0, NULL)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Pop_(blocks, size, NULL)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)


def test_Rack_Pop():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'pop index out of range'
        Rack_* temp = NULL
        Block_* backup = NULL
        Block_* backup2 = NULL
        ssize_t offset

    try:
        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, size - 1)
        blocks = Rack_Pop(blocks, <ssize_t>size - 1, &backup)  # final
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:-1])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, size - 1)
        blocks = Rack_Pop(blocks, -1, &backup)  # final
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:-1])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, 0)
        blocks = Rack_Pop(blocks, 0, &backup)  # initial
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, 0)
        blocks = Rack_Pop(blocks, -<ssize_t>size, &backup)  # initial
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        offset = <ssize_t>(size >> 1)  # half
        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, size >> 1)
        blocks = Rack_Pop(blocks, offset, &backup)
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:offset] + TEMPLATE_BLOCKS[offset + 1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        offset = <ssize_t>(size >> 1)  # half
        blocks =create_template_rack()
        backup2 = Rack_Get(blocks, size - (size >> 1))
        blocks = Rack_Pop(blocks, -offset, &backup)
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[:-offset] + TEMPLATE_BLOCKS[-offset + 1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        with pytest.raises(IndexError, match=match): Rack_Pop(blocks,  0, NULL)
        with pytest.raises(IndexError, match=match): Rack_Pop(blocks, -1, NULL)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Pop(blocks, size, NULL)
        with pytest.raises(IndexError, match=match): Rack_Pop(blocks, -<ssize_t>size - 1, NULL)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)


def test_Rack_PopLeft():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'pop index out of range'
        Rack_* temp = NULL
        Block_* backup = NULL
        Block_* backup2 = NULL

    try:
        blocks = create_template_rack()
        backup2 = Rack_Get(blocks, 0)
        blocks = Rack_PopLeft(blocks, &backup)
        assert Rack_Length(blocks) == size - 1
        temp = create_rack(TEMPLATE_BLOCKS[1:])
        assert Rack_Eq(blocks, temp) is True
        assert backup == backup2
        assert backup.references == 1
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        with pytest.raises(IndexError, match=match): Rack_PopLeft(blocks, NULL)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        backup = Block_Free(backup)
        blocks = Rack_Free(blocks)


def test_Rack_Insert_():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'index out of range'
        Rack_* temp = NULL
        Block_* block = NULL
        size_t offset

    try:
        blocks = Rack_Alloc(0)
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert_(blocks, 0, block)
        block = NULL
        assert Rack_Length(blocks) == 1
        temp = create_rack(RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert_(blocks, 0, block)  # before
        block = NULL
        assert Rack_Length(blocks) == 1 + size
        temp = create_rack(RACK1 + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert_(blocks, size, block)  # after
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS + RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        offset = size >> 1  # half
        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert_(blocks, offset, block)
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS[:offset] + RACK1 + TEMPLATE_BLOCKS[offset:])
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Alloc(0)
        with pytest.raises(IndexError, match=match): Rack_Insert_(blocks, 1, block)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        with pytest.raises(IndexError, match=match): Rack_Insert_(blocks, size + 1, block)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        block = Block_Free(block)
        blocks = Rack_Free(blocks)


def test_Rack_Insert():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'index out of range'
        Rack_* temp = NULL
        Block_* block = NULL
        ssize_t offset

    try:
        blocks = Rack_Alloc(0)
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, 0, block)
        block = NULL
        assert Rack_Length(blocks) == 1
        temp = create_rack(RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, -1, block)  # before, over
        block = NULL
        assert Rack_Length(blocks) == 1
        temp = create_rack(RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = Rack_Alloc(0)
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, 1, block)  # after, over
        block = NULL
        assert Rack_Length(blocks) == 1
        temp = create_rack(RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, 0, block)  # before
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(RACK1 + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, -<ssize_t>size, block)  # before
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(RACK1 + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, -<ssize_t>size - 1, block)  # before, over
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(RACK1 + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, <ssize_t>size, block)  # after
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS + RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, <ssize_t>size + 1, block)  # after, over
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS + RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        offset = <ssize_t>(size >> 1)  # half
        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, offset, block)
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS[:offset] + RACK1 + TEMPLATE_BLOCKS[offset:])
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        offset = -<ssize_t>(size >> 1)  # half
        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, offset, block)
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS[:offset] + RACK1 + TEMPLATE_BLOCKS[offset:])
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        offset = -1  # before end
        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Insert(blocks, offset, block)
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS[:offset] + RACK1 + TEMPLATE_BLOCKS[offset:])
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        block = Block_Free(block)
        blocks = Rack_Free(blocks)


def test_Rack_Append():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'index out of range'
        Rack_* temp = NULL
        Block_* block = NULL
        size_t offset

    try:
        blocks = Rack_Alloc(0)
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Append(blocks, block)
        block = NULL
        assert Rack_Length(blocks) == 1
        temp = create_rack(RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_Append(blocks, block)
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(TEMPLATE_BLOCKS + RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        block = Block_Free(block)
        blocks = Rack_Free(blocks)


def test_Rack_AppendLeft():
    cdef:
        Rack_* blocks = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        str match = 'index out of range'
        Rack_* temp = NULL
        Block_* block = NULL
        size_t offset

    try:
        blocks = Rack_Alloc(0)
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_AppendLeft(blocks, block)
        block = NULL
        assert Rack_Length(blocks) == 1
        temp = create_rack(RACK1)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

        blocks = create_template_rack()
        block = Block_Create(0x1234, SIZE1, DATA1)
        blocks = Rack_AppendLeft(blocks, block)
        block = NULL
        assert Rack_Length(blocks) == size + 1
        temp = create_rack(RACK1 + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks, temp) is True
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        block = Block_Free(block)
        blocks = Rack_Free(blocks)


def test_Rack_Extend_():
    cdef:
        Rack_* blocks1 = NULL
        Rack_* blocks2 = NULL
        size_t size1 = <size_t>len(TEMPLATE_BLOCKS)
        size_t size2 = <size_t>len(HELLO_WORLD_BLOCKS)
        Rack_* temp = NULL

    try:
        blocks1 = Rack_Alloc(0)
        blocks2 = create_rack(())
        blocks1 = Rack_Extend_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == 0
        temp = create_rack(())
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_rack(())
        blocks1 = Rack_Extend_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == size1
        temp = create_rack(TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = Rack_Alloc(0)
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_Extend_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == size2
        temp = create_rack(HELLO_WORLD_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_Extend_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == size1 + size2
        temp = create_rack(TEMPLATE_BLOCKS + HELLO_WORLD_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

    finally:
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)


def test_Rack_Extend():
    cdef:
        Rack_* blocks1 = NULL
        Rack_* blocks2 = NULL
        size_t size1 = <size_t>len(TEMPLATE_BLOCKS)
        size_t size2 = <size_t>len(HELLO_WORLD_BLOCKS)
        Rack_* temp = NULL

    try:
        blocks1 = Rack_Alloc(0)
        blocks2 = create_rack(())
        blocks1 = Rack_Extend(blocks1, blocks2)
        assert Rack_Length(blocks1) == 0
        temp = create_rack(())
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_rack(())
        blocks1 = Rack_Extend(blocks1, blocks2)
        assert Rack_Length(blocks1) == size1
        temp = create_rack(TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = Rack_Alloc(0)
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_Extend(blocks1, blocks2)
        assert Rack_Length(blocks1) == size2
        temp = create_rack(HELLO_WORLD_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_Extend(blocks1, blocks2)
        assert Rack_Length(blocks1) == size1 + size2
        temp = create_rack(TEMPLATE_BLOCKS + HELLO_WORLD_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

    finally:
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)


def test_Rack_ExtendLeft_():
    cdef:
        Rack_* blocks1 = NULL
        Rack_* blocks2 = NULL
        size_t size1 = <size_t>len(TEMPLATE_BLOCKS)
        size_t size2 = <size_t>len(HELLO_WORLD_BLOCKS)
        Rack_* temp = NULL

    try:
        blocks1 = Rack_Alloc(0)
        blocks2 = create_rack(())
        blocks1 = Rack_ExtendLeft_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == 0
        temp = create_rack(())
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_rack(())
        blocks1 = Rack_ExtendLeft_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == size1
        temp = create_rack(TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = Rack_Alloc(0)
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_ExtendLeft_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == size2
        temp = create_rack(HELLO_WORLD_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_ExtendLeft_(blocks1, Rack_Length(blocks2), Rack_At_(blocks2, 0), False)
        assert Rack_Length(blocks1) == size2 + size1
        temp = create_rack(HELLO_WORLD_BLOCKS + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

    finally:
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)


def test_Rack_ExtendLeft():
    cdef:
        Rack_* blocks1 = NULL
        Rack_* blocks2 = NULL
        size_t size1 = <size_t>len(TEMPLATE_BLOCKS)
        size_t size2 = <size_t>len(HELLO_WORLD_BLOCKS)
        Rack_* temp = NULL

    try:
        blocks1 = Rack_Alloc(0)
        blocks2 = create_rack(())
        blocks1 = Rack_ExtendLeft(blocks1, blocks2)
        assert Rack_Length(blocks1) == 0
        temp = create_rack(())
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_rack(())
        blocks1 = Rack_ExtendLeft(blocks1, blocks2)
        assert Rack_Length(blocks1) == size1
        temp = create_rack(TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = Rack_Alloc(0)
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_ExtendLeft(blocks1, blocks2)
        assert Rack_Length(blocks1) == size2
        temp = create_rack(HELLO_WORLD_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

        blocks1 = create_template_rack()
        blocks2 = create_hello_world_rack()
        blocks1 = Rack_ExtendLeft(blocks1, blocks2)
        assert Rack_Length(blocks1) == size2 + size1
        temp = create_rack(HELLO_WORLD_BLOCKS + TEMPLATE_BLOCKS)
        assert Rack_Eq(blocks1, temp) is True
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)

    finally:
        temp = Rack_Free(temp)
        blocks1 = Rack_Free(blocks1)
        blocks2 = Rack_Free(blocks2)


# TODO: Rack_Read_()


# TODO: Rack_Write_()


# TODO: Rack_ReadSlice_()


# TODO: Rack_ReadSlice()


# TODO: Rack_GetSlice_()


# TODO: Rack_GetSlice()


# TODO: Rack_WriteSlice_()


# TODO: Rack_WriteSlice()


# TODO: Rack_SetSlice_()


# TODO: Rack_SetSlice()


def test_Rack_DelSlice_():
    cdef:
        Rack_* blocks = NULL
        Rack_* temp = NULL
        size_t size = <size_t>len(TEMPLATE_BLOCKS)
        size_t start
        size_t endex
        list temp_

    try:
        for start in range(size):
            for endex in range(size):
                blocks = create_rack(())
                blocks = Rack_DelSlice_(blocks, start, endex)
                temp = create_rack(())
                assert blocks != NULL
                assert blocks != temp
                assert Rack_Length(blocks) == 0
                assert Rack_Eq(blocks, temp) is True
                temp = Rack_Free(temp)
                blocks = Rack_Free(blocks)

        for start in range(size + size):
            for endex in range(size + size):
                blocks = create_template_rack()
                blocks = Rack_DelSlice_(blocks, start, endex)
                temp_ = list(TEMPLATE_BLOCKS)
                del temp_[start:endex]
                temp = create_rack(tuple(temp_))
                assert blocks != NULL
                assert blocks != temp
                assert Rack_Length(blocks) == <size_t>len(temp_)
                assert Rack_Eq(blocks, temp) is True
                temp = Rack_Free(temp)
                blocks = Rack_Free(blocks)

        blocks = create_template_rack()

        with pytest.raises(OverflowError, match='size overflow'):
            Rack_DelSlice_(blocks, SIZE_HMAX + 1, 0)

        with pytest.raises(OverflowError, match='size overflow'):
            Rack_DelSlice_(blocks, 0, SIZE_HMAX + 1)

        blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)


def test_Rack_DelSlice():
    cdef:
        Rack_* blocks = NULL
        Rack_* temp = NULL
        size_t size = SIZE1
        ssize_t start
        ssize_t endex
        list temp_

    try:
        for start in range(-<ssize_t>size, <ssize_t>size):
            for endex in range(-<ssize_t>size, <ssize_t>size):
                blocks = create_rack(())
                blocks = Rack_DelSlice(blocks, start, endex)
                temp = create_rack(())
                assert blocks != NULL
                assert blocks != temp
                assert Rack_Length(blocks) == 0
                assert Rack_Eq(blocks, temp) is True
                temp = Rack_Free(temp)
                blocks = Rack_Free(blocks)

        for start in range(-<ssize_t>(size + size), <ssize_t>(size + size)):
            for endex in range(-<ssize_t>size, <ssize_t>size):
                blocks = create_template_rack()
                blocks = Rack_DelSlice(blocks, start, endex)
                temp_ = list(TEMPLATE_BLOCKS)
                del temp_[start:endex]
                temp = create_rack(tuple(temp_))
                assert blocks != NULL
                assert blocks != temp
                assert Rack_Length(blocks) == <size_t>len(temp_)
                assert Rack_Eq(blocks, temp) is True
                temp = Rack_Free(temp)
                blocks = Rack_Free(blocks)

    finally:
        temp = Rack_Free(temp)
        blocks = Rack_Free(blocks)


def test_Rack_IndexAt():
    cdef:
        Rack_* blocks = NULL
        size_t block_count
        size_t block_index
        const Block_* block
        addr_t block_start
        addr_t block_endex
        addr_t address

    try:
        blocks = Rack_Alloc(0)
        assert Rack_IndexAt(blocks, 0) == -1
        assert Rack_IndexAt(blocks, ADDR_MAX) == -1

        blocks = Rack_Free(blocks)
        blocks = create_template_rack()
        block_count = Rack_Length(blocks)

        for block_index in range(block_count):
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)
            for address in range(block_start, block_endex):
                assert Rack_IndexAt(blocks, address) == <ssize_t>block_index

            block_index += 1
            if block_index < block_count:
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                for address in range(block_endex, block_start):
                    assert Rack_IndexAt(blocks, address) == -1

        assert Rack_IndexAt(blocks, Block_Start(Rack_First__(blocks)) - 1) == -1
        assert Rack_IndexAt(blocks, Block_Endex(Rack_Last__(blocks)) + 1) == -1

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_IndexStart():
    cdef:
        Rack_* blocks = NULL
        size_t block_count
        size_t block_index
        const Block_* block
        addr_t block_start
        addr_t block_endex
        addr_t address

    try:
        blocks = Rack_Alloc(0)
        assert Rack_IndexStart(blocks, 0) == 0
        assert Rack_IndexStart(blocks, ADDR_MAX) == 0

        blocks = Rack_Free(blocks)
        blocks = create_template_rack()
        block_count = Rack_Length(blocks)

        for block_index in range(block_count):
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)
            for address in range(block_start, block_endex):
                assert Rack_IndexStart(blocks, address) == <ssize_t>block_index

            block_index += 1
            if block_index < block_count:
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                for address in range(block_endex, block_start):
                    assert Rack_IndexStart(blocks, address) == <ssize_t>block_index

        assert Rack_IndexStart(blocks, Block_Start(Rack_First__(blocks)) - 1) == 0
        assert Rack_IndexStart(blocks, Block_Endex(Rack_Last__(blocks)) + 1) == <ssize_t>block_count

    finally:
        blocks = Rack_Free(blocks)


def test_Rack_IndexEndex():
    cdef:
        Rack_* blocks = NULL
        size_t block_count
        size_t block_index
        const Block_* block
        addr_t block_start
        addr_t block_endex
        addr_t address

    try:
        blocks = Rack_Alloc(0)
        assert Rack_IndexEndex(blocks, 0) == 0
        assert Rack_IndexEndex(blocks, ADDR_MAX) == 0

        blocks = Rack_Free(blocks)
        blocks = create_template_rack()
        block_count = Rack_Length(blocks)

        for block_index in range(block_count):
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)
            block_index += 1
            for address in range(block_start, block_endex):
                assert Rack_IndexEndex(blocks, address) == <ssize_t>block_index

            if block_index < block_count:
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                for address in range(block_endex, block_start):
                    assert Rack_IndexEndex(blocks, address) == <ssize_t>block_index

        assert Rack_IndexEndex(blocks, Block_Start(Rack_First__(blocks)) - 1) == 0
        assert Rack_IndexEndex(blocks, Block_Endex(Rack_Last__(blocks)) + 1) == <ssize_t>block_count

    finally:
        blocks = Rack_Free(blocks)


# =====================================================================================================================

# Rover: done by _common.py


# ---------------------------------------------------------------------------------------------------------------------

# Memory: done by _common.py


# =====================================================================================================================

# Patch all Cython tests so that they can be discovered by pytest.
# Requires cython option: binding = True
def _patch_cytest():
    import functools
    import inspect

    def cytest(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            bound = inspect.signature(func).bind(*args, **kwargs)
            return func(*bound.args, **bound.kwargs)

        return wrapped

    g = globals()
    for key, value in g.items():
        if hasattr(value, '__name__'):
            if callable(value) and value.__name__.startswith('test_'):
                g[key] = cytest(value)


_patch_cytest()
