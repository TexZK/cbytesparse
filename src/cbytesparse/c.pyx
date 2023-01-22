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

r"""Cython implementation."""

cimport cython
from cpython.bytearray cimport PyByteArray_FromStringAndSize
from cpython.bytes cimport PyBytes_FromStringAndSize

from itertools import count as _count
from itertools import islice as _islice
from itertools import repeat as _repeat
from itertools import zip_longest as _zip_longest
from typing import Any
from typing import ByteString
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from bytesparse.base import STR_MAX_CONTENT_SIZE
from bytesparse.base import Address
from bytesparse.base import AddressValueMapping
from bytesparse.base import AnyBytes
from bytesparse.base import Block
from bytesparse.base import BlockIndex
from bytesparse.base import BlockIterable
from bytesparse.base import BlockList
from bytesparse.base import BlockSequence
from bytesparse.base import ClosedInterval
from bytesparse.base import EllipsisType
from bytesparse.base import ImmutableMemory
from bytesparse.base import MutableMemory
from bytesparse.base import OpenInterval
from bytesparse.base import Value

# Allocate an empty block, so that an empty view can be returned statically
cdef:
    Block_* _empty_block = Block_Acquire(Block_Alloc(0, 0, False))


# =====================================================================================================================

def collapse_blocks(
    blocks: BlockIterable,
) -> BlockList:
    r"""Collapses a generic sequence of blocks.

    Given a generic sequence of blocks, writes them in the same order,
    generating a new sequence of non-contiguous blocks, sorted by address.

    Arguments:
        blocks (sequence of blocks):
            Sequence of blocks to collapse.

    Returns:
        list of blocks: Collapsed block list.

    Examples:
        +---+---+---+---+---+---+---+---+---+---+
        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
        +===+===+===+===+===+===+===+===+===+===+
        |[0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9]|
        +---+---+---+---+---+---+---+---+---+---+
        |[A | B | C | D]|   |   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+
        |   |   |   |[E | F]|   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+
        |[$]|   |   |   |   |   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+
        |   |   |   |   |   |   |[x | y | z]|   |
        +---+---+---+---+---+---+---+---+---+---+
        |[$ | B | C | E | F | 5 | x | y | z | 9]|
        +---+---+---+---+---+---+---+---+---+---+

        >>> blocks = [
        ...     [0, b'0123456789'],
        ...     [0, b'ABCD'],
        ...     [3, b'EF'],
        ...     [0, b'$'],
        ...     [6, b'xyz'],
        ... ]
        >>> collapse_blocks(blocks)
        [[0, b'$BCEF5xyz9']]

        ~~~

        +---+---+---+---+---+---+---+---+---+---+
        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
        +===+===+===+===+===+===+===+===+===+===+
        |[0 | 1 | 2]|   |   |   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+
        |   |   |   |   |[A | B]|   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+
        |   |   |   |   |   |   |[x | y | z]|   |
        +---+---+---+---+---+---+---+---+---+---+
        |   |[$]|   |   |   |   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+
        |[0 | $ | 2]|   |[A | B | x | y | z]|   |
        +---+---+---+---+---+---+---+---+---+---+

        >>> blocks = [
        ...     [0, b'012'],
        ...     [4, b'AB'],
        ...     [6, b'xyz'],
        ...     [1, b'$'],
        ... ]
        >>> collapse_blocks(blocks)
        [[0, b'0$2'], [4, b'ABxyz']]
    """

    cdef:
        Memory_* memory = Memory_Alloc()
        list collapsed = []
        size_t block_index
        const Rack_* blocks2
        const Block_* block

    try:
        for block_start, block_data in blocks:
            Memory_Write(memory, block_start, block_data, True)
        blocks2 = memory.blocks

        for block_index in range(Rack_Length(blocks2)):
            block = Rack_Get__(blocks2, block_index)
            collapsed.append([Block_Start(block), Block_Bytes(block)])
    finally:
        Memory_Free(memory)

    return collapsed


# =====================================================================================================================

# FIXME: Not yet provided by the current Cython (0.29.x)
cdef void* PyMem_Calloc(size_t nelem, size_t elsize):
    cdef:
        void* ptr
        size_t total

    if CannotMulSizeU(nelem, elsize):
        return NULL  # overflow
    total = nelem * elsize

    ptr = PyMem_Malloc(total)
    if ptr:
        memset(ptr, 0, total)
    return ptr


# =====================================================================================================================

cdef size_t Downsize(size_t allocated, size_t requested) nogil:
    # Note: free margin will be either before and after allocated data
    cdef size_t resized

    if requested < allocated >> 1:
        # Major downsize; allocate as per request
        resized = requested

        # Align to next MARGIN; always gives some additional MARGIN
        with cython.cdivision(True):
            resized += (2 * MARGIN) - (resized % MARGIN)
    else:
        # Do not require reallocation
        resized = allocated

        # Align to next MARGIN; always gives some additional MARGIN
        if resized < 2 * MARGIN:
            resized = 2 * MARGIN

    return resized


cdef size_t Upsize(size_t allocated, size_t requested) nogil:
    # Note: free margin will be either before and after allocated data
    cdef size_t resized = requested

    # Moderate upsize; overallocate proportionally
    if resized <= allocated + (allocated >> 3):
        resized += resized >> 3

    # Align to next MARGIN; always gives some additional MARGIN
    with cython.cdivision(True):
        resized += (2 * MARGIN) - (resized % MARGIN)
    return resized


# ---------------------------------------------------------------------------------------------------------------------

cdef void Reverse(byte_t* buffer, size_t start, size_t endin) nogil:
    cdef:
        byte_t t

    while start < endin:
        t = buffer[start]
        buffer[start] = buffer[endin]
        buffer[endin] = t
        start += 1
        endin -= 1


cdef bint IsSequence(object obj) except -1:
    try:
        len(obj)
        obj[0:0]
        return True
    except TypeError:
        return False


# =====================================================================================================================

cdef bint CannotAddSizeU(size_t a, size_t b) nogil:
    return SIZE_MAX - a < b


cdef vint CheckAddSizeU(size_t a, size_t b) except -1:
    if CannotAddSizeU(a, b):
        raise OverflowError()


cdef size_t AddSizeU(size_t a, size_t b) except? 0xDEAD:
    CheckAddSizeU(a, b)
    return a + b


cdef bint CannotSubSizeU(size_t a, size_t b) nogil:
    return a < b


cdef vint CheckSubSizeU(size_t a, size_t b) except -1:
    if CannotSubSizeU(a, b):
        raise OverflowError()


cdef size_t SubSizeU(size_t a, size_t b) except? 0xDEAD:
    CheckSubSizeU(a, b)
    return a - b


cdef bint CannotMulSizeU(size_t a, size_t b) nogil:
    cdef:
        size_t r = a * b
    return a and b and (r < a or r < b)


cdef vint CheckMulSizeU(size_t a, size_t b) except -1:
    if CannotMulSizeU(a, b):
        raise OverflowError()


cdef size_t MulSizeU(size_t a, size_t b) except? 0xDEAD:
    CheckMulSizeU(a, b)
    return a * b


# ---------------------------------------------------------------------------------------------------------------------

cdef bint CannotAddSizeS(ssize_t a, ssize_t b) nogil:
    return ((b > 0 and a > SSIZE_MAX - b) or
            (b < 0 and a < SSIZE_MIN - b))


cdef vint CheckAddSizeS(ssize_t a, ssize_t b) except -1:
    if CannotAddSizeS(a, b):
        raise OverflowError()


cdef ssize_t AddSizeS(ssize_t a, ssize_t b) except? 0xDEAD:
    CheckAddSizeS(a, b)
    return a + b


cdef bint CannotSubSizeS(ssize_t a, ssize_t b) nogil:
    return ((b > 0 and a < SSIZE_MIN + b) or
            (b < 0 and a > SSIZE_MAX + b))


cdef vint CheckSubSizeS(ssize_t a, ssize_t b) except -1:
    if CannotSubSizeS(a, b):
        raise OverflowError()


cdef ssize_t SubSizeS(ssize_t a, ssize_t b) except? 0xDEAD:
    CheckSubSizeS(a, b)
    return a - b


cdef bint CannotMulSizeS(ssize_t a, ssize_t b) nogil:
    with cython.cdivision(True):
        if a > 0:
            if b > 0:
                return a > (SSIZE_MAX // b)
            else:
                return b < (SSIZE_MIN // a)
        else:
            if b > 0:
                return a < (SSIZE_MIN // b)
            else:
                return a and b < (SSIZE_MAX // a)


cdef vint CheckMulSizeS(ssize_t a, ssize_t b) except -1:
    if CannotMulSizeS(a, b):
        raise OverflowError()


cdef ssize_t MulSizeS(ssize_t a, ssize_t b) except? 0xDEAD:
    CheckMulSizeS(a, b)
    return a * b


# =====================================================================================================================

cdef bint CannotAddAddrU(addr_t a, addr_t b) nogil:
    return ADDR_MAX - a < b


cdef vint CheckAddAddrU(addr_t a, addr_t b) except -1:
    if CannotAddAddrU(a, b):
        raise OverflowError()


cdef addr_t AddAddrU(addr_t a, addr_t b) except? 0xDEAD:
    CheckAddAddrU(a, b)
    return a + b


cdef bint CannotSubAddrU(addr_t a, addr_t b) nogil:
    return a < b


cdef vint CheckSubAddrU(addr_t a, addr_t b) except -1:
    if CannotSubAddrU(a, b):
        raise OverflowError()


cdef addr_t SubAddrU(addr_t a, addr_t b) except? 0xDEAD:
    CheckSubAddrU(a, b)
    return a - b


cdef bint CannotMulAddrU(addr_t a, addr_t b) nogil:
    cdef:
        addr_t r = a * b
    return a and b and (r < a or r < b)


cdef vint CheckMulAddrU(addr_t a, addr_t b) except -1:
    if CannotMulAddrU(a, b):
        raise OverflowError()


cdef addr_t MulAddrU(addr_t a, addr_t b) except? 0xDEAD:
    CheckMulAddrU(a, b)
    return a * b


cdef bint CannotAddrToSizeU(addr_t a) nogil:
    return a > <addr_t>SIZE_MAX


cdef vint CheckAddrToSizeU(addr_t a) except -1:
    if CannotAddrToSizeU(a):
        raise OverflowError()


cdef size_t AddrToSizeU(addr_t a) except? 0xDEAD:
    CheckAddrToSizeU(a)
    return <size_t>a


# ---------------------------------------------------------------------------------------------------------------------

cdef bint CannotAddAddrS(saddr_t a, saddr_t b) nogil:
    return ((b > 0 and a > SADDR_MAX - b) or
            (b < 0 and a < SADDR_MIN - b))


cdef vint CheckAddAddrS(saddr_t a, saddr_t b) except -1:
    if CannotAddAddrS(a, b):
        raise OverflowError()


cdef saddr_t AddAddrS(saddr_t a, saddr_t b) except? 0xDEAD:
    CheckAddAddrS(a, b)
    return a + b


cdef bint CannotSubAddrS(saddr_t a, saddr_t b) nogil:
    return ((b > 0 and a < SADDR_MIN + b) or
            (b < 0 and a > SADDR_MAX + b))


cdef vint CheckSubAddrS(saddr_t a, saddr_t b) except -1:
    if CannotSubAddrS(a, b):
        raise OverflowError()


cdef saddr_t SubAddrS(saddr_t a, saddr_t b) except? 0xDEAD:
    CheckSubAddrS(a, b)
    return a - b


cdef bint CannotMulAddrS(saddr_t a, saddr_t b) nogil:
    with cython.cdivision(True):
        if a > 0:
            if b > 0:
                return a > (SADDR_MAX // b)
            else:
                return b < (SADDR_MIN // a)
        else:
            if b > 0:
                return a < (SADDR_MIN // b)
            else:
                return a and b < (SADDR_MAX // a)


cdef vint CheckMulAddrS(saddr_t a, saddr_t b) except -1:
    if CannotMulAddrS(a, b):
        raise OverflowError()


cdef saddr_t MulAddrS(saddr_t a, saddr_t b) except? 0xDEAD:
    CheckMulAddrS(a, b)
    return a * b


cdef bint CannotAddrToSizeS(saddr_t a) nogil:
    return a < <saddr_t>SSIZE_MIN or a > <saddr_t>SSIZE_MAX


cdef vint CheckAddrToSizeS(saddr_t a) except -1:
    if CannotAddrToSizeS(a):
        raise OverflowError()


cdef ssize_t AddrToSizeS(saddr_t a) except? 0xDEAD:
    CheckAddrToSizeS(a)
    return <ssize_t>a


# =====================================================================================================================

cdef Block_* Block_Alloc(addr_t address, size_t size, bint zero) except NULL:
    cdef:
        Block_* that = NULL
        size_t allocated
        size_t actual

    if size > SIZE_HMAX:
        raise OverflowError('size overflow')

    # Allocate as per request
    allocated = Upsize(0, size)
    if allocated > SIZE_HMAX:
        raise MemoryError()

    actual = Block_HEADING + (allocated * sizeof(byte_t))
    if zero:
        that = <Block_*>PyMem_Calloc(actual, 1)
    else:
        that = <Block_*>PyMem_Malloc(actual)
    if that == NULL:
        raise MemoryError()

    that.address = address
    that.references = 1  # acquired by default
    that.allocated = allocated
    that.start = MARGIN  # leave some initial room
    that.endex = that.start + size
    return that


cdef Block_* Block_Free(Block_* that):
    if that:
        PyMem_Free(that)
    return NULL


cdef size_t Block_Sizeof(const Block_* that):
    if that:
        return Block_HEADING + (that.allocated * sizeof(byte_t))
    return 0


cdef Block_* Block_Create(addr_t address, size_t size, const byte_t* buffer) except NULL:
    if not size or buffer:
        that = Block_Alloc(address, size, False)
        memcpy(&that.data[that.start], buffer, size * sizeof(byte_t))
        return that
    else:
        raise ValueError('null pointer')


cdef Block_* Block_Copy(const Block_* that) except NULL:
    cdef:
        Block_* ptr
        size_t size

    if that:
        size = Block_HEADING + (that.allocated * sizeof(byte_t))
        ptr = <Block_*>PyMem_Malloc(size)
        if ptr == NULL:
            raise MemoryError()

        memcpy(ptr, that, size)
        ptr.references = 1  # acquired by default
        return ptr
    else:
        raise ValueError('null pointer')


cdef Block_* Block_FromObject(addr_t address, object obj, bint nonnull) except NULL:
    cdef:
        byte_t value
        const byte_t[:] view
        size_t size
        const byte_t* ptr

    if isinstance(obj, int):
        value = <byte_t>obj
        return Block_Create(address, 1, &value)
    else:
        try:
            view = obj
        except TypeError:
            view = bytes(obj)
        size = len(view)
        if size:
            with cython.boundscheck(False):
                ptr = &view[0]
            return Block_Create(address, size, ptr)
        else:
            if nonnull:
                raise ValueError('invalid block data size')
            else:
                return Block_Alloc(address, 0, False)


cdef void Block_Reverse(Block_* that) nogil:
    cdef:
        size_t start = that.start
        size_t endex = that.endex
        size_t endin
        byte_t temp

    while start < endex:
        endin = endex - 1
        temp = that.data[start]
        that.data[start] = that.data[endin]
        that.data[endin] = temp
        endex = endin
        start += 1


cdef Block_* Block_Acquire(Block_* that) except NULL:
    if that:
        if that.references < SIZE_MAX:
            that.references += 1
            return that
        else:
            raise OverflowError()
    else:
        raise RuntimeError('null pointer')


cdef Block_* Block_Release_(Block_* that):
    if that:
        if that.references:
            that.references -= 1

        if that.references:
            return that
        else:
            PyMem_Free(that)

    return NULL


cdef Block_* Block_Release(Block_* that):
    if that:
        if that.references:
            that.references -= 1

        if not that.references:
            PyMem_Free(that)

    return NULL


cdef bint Block_Bool(const Block_* that) nogil:
    return that.start < that.endex


cdef size_t Block_Length(const Block_* that) nogil:
    return that.endex - that.start


cdef addr_t Block_Start(const Block_* that) nogil:
    return that.address


cdef addr_t Block_Endex(const Block_* that) nogil:
    return that.address + (that.endex - that.start)


cdef addr_t Block_Endin(const Block_* that) nogil:
    return that.address + (that.endex - that.start) - 1


cdef addr_t Block_BoundAddress(const Block_* that, addr_t address) nogil:
    cdef:
        addr_t block_start = that.address
        addr_t block_endex = block_start + that.endex - that.start

    if address < block_start:
        address = block_start  # trim to start
    elif address > block_endex:
        address = block_endex  # trim to end
    return address


cdef size_t Block_BoundAddressToOffset(const Block_* that, addr_t address) nogil:
    cdef:
        addr_t block_start = that.address
        addr_t block_endex = block_start + that.endex - that.start

    if address < block_start:
        address = block_start  # trim to start
    elif address > block_endex:
        address = block_endex  # trim to end
    return <size_t>(address - block_start)


cdef size_t Block_BoundOffset(const Block_* that, size_t offset) nogil:
    cdef:
        size_t size = that.endex - that.start

    if offset > size:
        offset = size  # trim to end
    return offset


cdef (addr_t, addr_t) Block_BoundAddressSlice(const Block_* that, addr_t start, addr_t endex) nogil:
    cdef:
        addr_t block_start = that.address
        addr_t block_endex = block_start + (that.endex - that.start)

    if start < block_start:
        start = block_start  # trim to start
    elif start > block_endex:
        start = block_endex  # trim to end

    if endex < block_start:
        endex = block_start  # trim to start
    elif endex > block_endex:
        endex = block_endex  # trim to end

    if endex < start:
        endex = start  # clamp negative length

    return start, endex


cdef (size_t, size_t) Block_BoundAddressSliceToOffset(const Block_* that, addr_t start, addr_t endex) nogil:
    cdef:
        addr_t block_start = that.address
        addr_t block_endex = block_start + (that.endex - that.start)

    if start < block_start:
        start = block_start  # trim to start
    elif start > block_endex:
        start = block_endex  # trim to end

    if endex < block_start:
        endex = block_start  # trim to start
    elif endex > block_endex:
        endex = block_endex  # trim to end

    if endex < start:
        endex = start  # clamp negative length

    return <size_t>(start - block_start), <size_t>(endex - block_start)


cdef (size_t, size_t) Block_BoundOffsetSlice(const Block_* that, size_t start, size_t endex) nogil:
    cdef:
        size_t size = that.endex - that.start

    if start > size:
        start = size  # trim to end

    if endex > size:
        endex = size  # trim to end

    if endex < start:
        endex = start  # clamp negative length

    return start, endex


cdef vint Block_CheckMutable(const Block_* that) except -1:
    if that.references > 1:
        raise RuntimeError('Existing exports of data: object cannot be re-sized')


cdef bint Block_Eq_(const Block_* that, size_t size, const byte_t* buffer) nogil:
    if size != that.endex - that.start:
        return False

    if size:
        if memcmp(&that.data[that.start], buffer, size):
            return False

    return True


cdef bint Block_Eq(const Block_* that, const Block_* other) nogil:
    # if that.address != other.address:
    #     return False

    return Block_Eq_(that, other.endex - other.start, &other.data[other.start])


cdef int Block_Cmp_(const Block_* that, size_t size, const byte_t* buffer) nogil:
    cdef:
        size_t size2 = that.endex - that.start
        size_t minsize = size2 if size2 < size else size
        const byte_t* buffer2 = &that.data[that.start]
        int sign = memcmp(buffer2, buffer, minsize)

    if size2 == size:
        return sign
    elif sign:
        return sign
    else:
        return -1 if size2 < size else +1


cdef int Block_Cmp(const Block_* that, const Block_* other) nogil:
    # if that.address != other.address:
    #     return -1 if that.address < other.address else +1

    return Block_Cmp_(that, other.endex - other.start, &other.data[other.start])


cdef ssize_t Block_Find__(const Block_* that, size_t start, size_t endex, byte_t value) nogil:
    cdef:
        size_t size = that.endex - that.start
        const byte_t* ptr
        const byte_t* end

    if start > size:
        start = size  # trim to end
    if endex > size:
        endex = size  # trim to end
    if endex < start:
        endex = start  # clamp negative length

    ptr = &that.data[that.start + start]
    end = &that.data[that.start + endex]

    while ptr != end:
        if ptr[0] == value:
            return <ssize_t>(<ptrdiff_t>ptr - <ptrdiff_t>&that.data[that.start])
        ptr += 1
    return -1


cdef ssize_t Block_Find_(const Block_* that, size_t start, size_t endex,
                         size_t size, const byte_t* buffer) nogil:
    cdef:
        size_t size2
        const byte_t* ptr
        const byte_t* end

    if size == 1:  # faster code for single byte
        return Block_Find__(that, start, endex, buffer[0])

    elif size:
        size2 = that.endex - that.start

        if start > size2:
            start = size2  # trim to end
        if endex > size2:
            endex = size2  # trim to end
        if endex < start:
            endex = start  # clamp negative length

        if size <= size2 and size <= endex - start:
            size2 = endex - size + 1

            if start > size2:
                start = size2  # trim to end
            if endex > size2:
                endex = size2  # trim to end

            ptr = &that.data[that.start + start]
            end = &that.data[that.start + endex]

            while ptr != end:
                if ptr[0] == buffer[0]:  # faster pruning
                    if not memcmp(ptr, buffer, size):
                        return <ssize_t>(<ptrdiff_t>ptr - <ptrdiff_t>&that.data[that.start])
                ptr += 1
    return -1


cdef ssize_t Block_Find(const Block_* that, ssize_t start, ssize_t endex,
                        size_t size, const byte_t* buffer) nogil:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
        if start < 0:
            start = 0  # trim to start

    if endex < 0:
        endex += ssize  # anchor to end
        if endex < 0:
            endex = 0  # trim to start

    return Block_Find_(that, <size_t>start, <size_t>endex, size, buffer)


cdef ssize_t Block_ReverseFind__(const Block_* that, size_t start, size_t endex, byte_t value) nogil:
    cdef:
        size_t size = that.endex - that.start
        const byte_t* ptr
        const byte_t* end

    if size:
        if start > size:
            start = size  # trim to end
        if endex > size:
            endex = size  # trim to end
        if endex < start:
            endex = start  # clamp negative length

        end = &that.data[that.start + start]
        ptr = &that.data[that.start + endex]

        while ptr != end:
            ptr -= 1
            if ptr[0] == value:
                return <ssize_t>(<ptrdiff_t>ptr - <ptrdiff_t>&that.data[that.start])
    return -1


cdef ssize_t Block_ReverseFind_(const Block_* that, size_t start, size_t endex,
                                size_t size, const byte_t* buffer) nogil:
    cdef:
        size_t size2
        const byte_t* ptr
        const byte_t* end

    if size == 1:  # faster code for single byte
        return Block_ReverseFind__(that, start, endex, buffer[0])

    elif size:
        size2 = that.endex - that.start

        if start > size2:
            start = size2  # trim to end
        if endex > size2:
            endex = size2  # trim to end
        if endex < start:
            endex = start  # clamp negative length

        if size <= size2 and size <= endex - start:
            size2 = endex - size + 1

            if start > size2:
                start = size2  # trim to end
            if endex > size2:
                endex = size2  # trim to end

            end = &that.data[that.start + start]
            ptr = &that.data[that.start + endex]

            while ptr != end:
                ptr -= 1
                if ptr[0] == buffer[0]:  # faster pruning
                    if not memcmp(ptr, buffer, size):
                        return <ssize_t>(<ptrdiff_t>ptr - <ptrdiff_t>&that.data[that.start])
    return -1


cdef ssize_t Block_ReverseFind(const Block_* that, ssize_t start, ssize_t endex,
                               size_t size, const byte_t* buffer) nogil:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
        if start < 0:
            start = 0  # trim to start

    if endex < 0:
        endex += ssize  # anchor to end
        if endex < 0:
            endex = 0  # trim to start

    return Block_ReverseFind_(that, <size_t>start, <size_t>endex, size, buffer)


cdef size_t Block_Count__(const Block_* that, size_t start, size_t endex, byte_t value) nogil:
    cdef:
        size_t count = 0
        size_t size = that.endex - that.start
        const byte_t* ptr
        const byte_t* end

    if start > size:
        start = size  # trim to end
    if endex > size:
        endex = size  # trim to end
    if endex < start:
        endex = start  # clamp negative length

    ptr = &that.data[that.start + start]
    end = &that.data[that.start + endex]

    while ptr != end:
        if ptr[0] == value:
            count += 1
        ptr += 1
    return count


cdef size_t Block_Count_(const Block_* that, size_t start, size_t endex,
                         size_t size, const byte_t* buffer) nogil:
    cdef:
        size_t count = 0
        size_t size2
        const byte_t* ptr
        const byte_t* end

    if size == 1:  # faster code for single byte
        return Block_Count__(that, start, endex, buffer[0])

    elif size:
        size2 = that.endex - that.start

        if start > size2:
            start = size2  # trim to end
        if endex > size2:
            endex = size2  # trim to end
        if endex < start:
            endex = start  # clamp negative length

        if size <= size2 and size <= endex - start:
            size2 = endex - size + 1

            if start > size2:
                start = size2  # trim to end
            if endex > size2:
                endex = size2  # trim to end

            ptr = &that.data[that.start + start]
            end = &that.data[that.start + endex]

            while ptr < end:
                if ptr[0] == buffer[0]:  # faster pruning
                    if not memcmp(ptr, buffer, size):
                        ptr += size - 1
                        count += 1
                ptr += 1
    return count


cdef size_t Block_Count(const Block_* that, ssize_t start, ssize_t endex,
                        size_t size, const byte_t* buffer) nogil:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
        if start < 0:
            start = 0  # trim to start

    if endex < 0:
        endex += ssize  # anchor to end
        if endex < 0:
            endex = 0  # trim to start

    return Block_Count_(that, <size_t>start, <size_t>endex, size, buffer)


cdef Block_* Block_Reserve_(Block_* that, size_t offset, size_t size, bint zero) except NULL:
    cdef:
        size_t actual
        size_t used
        size_t margin
        size_t allocated
        Block_* ptr

    Block_CheckMutable(that)

    if size:
        if ((size > SIZE_HMAX or
             CannotAddSizeU(that.endex, size) or
             that.endex + size > SIZE_HMAX)):
            raise OverflowError('size overflow')

        used = that.endex - that.start
        if offset > used:
            raise IndexError('index out of range')

        # Prefer the side where there is less data to shift, which also favors the extremes
        if offset >= (used >> 1):
            if size > that.allocated - that.endex:
                # Calculate the upsized allocation
                allocated = Upsize(that.allocated, that.allocated + size)
                if allocated > SIZE_HMAX:
                    raise MemoryError()

                # Reallocate, including the header
                ptr = <Block_*>PyMem_Realloc(that, Block_HEADING + (allocated * sizeof(byte_t)))
                if ptr == NULL:
                    raise MemoryError()

                # Reassign to that
                that = ptr
                that.allocated = allocated  # update

            # Shift elements to make room for reservation at the requested offset
            CheckAddSizeU(offset, that.start)
            offset += that.start
            used = that.endex - offset
            if used:
                memmove(&that.data[offset + size],
                        &that.data[offset],
                        used * sizeof(byte_t))
            if zero:
                memset(&that.data[offset], 0, size * sizeof(byte_t))  # pad with zeros
            that.endex += size

        else:
            if size <= that.start:
                # Shift elements to make room for reservation at the requested offset
                that.start -= size
                if offset:
                    memmove(&that.data[that.start],
                            &that.data[that.start + size],
                            offset * sizeof(byte_t))
                if zero:
                    memset(&that.data[that.start + offset], 0, size * sizeof(byte_t))  # pad with zeros

            else:
                # Calculate the upsized allocation
                CheckAddSizeU(that.allocated, size)
                allocated = Upsize(that.allocated, that.allocated + size)
                if allocated > SIZE_HMAX:
                    raise MemoryError()

                # Allocate a new chunk, including the header
                actual = Block_HEADING + (allocated * sizeof(byte_t))
                if zero:
                    ptr = <Block_*>PyMem_Calloc(actual, 1)
                else:
                    ptr = <Block_*>PyMem_Malloc(actual)
                if ptr == NULL:
                    raise MemoryError()

                # Prepare the new chunk aligning towards the end
                ptr.address = that.address
                ptr.references = that.references  # transfer ownership
                ptr.allocated = allocated
                ptr.endex = ptr.allocated - MARGIN  # leave some room
                ptr.start = ptr.endex - used - size

                # Shift/copy elements to make room for reservation at the requested offset
                if offset:
                    used -= offset  # prepare for later
                    memcpy(&ptr.data[ptr.start],
                           &that.data[that.start],
                           offset * sizeof(byte_t))
                if used:
                    memcpy(&ptr.data[ptr.start + offset + size],
                           &that.data[that.start + offset],
                           used * sizeof(byte_t))

                # Reassign to that
                PyMem_Free(that)
                that = ptr

    return that


cdef Block_* Block_Delete_(Block_* that, size_t offset, size_t size) except NULL:
    cdef:
        size_t allocated
        Block_* ptr

    Block_CheckMutable(that)

    if size:
        if ((size > SIZE_HMAX or
             CannotAddSizeU(offset, size) or
             offset + size > SIZE_HMAX or
             CannotAddSizeU(offset, that.start) or
             that.start > SIZE_HMAX)):
            raise OverflowError('size overflow')

        if that.endex < that.start + offset + size:
            raise IndexError('index out of range')

        # Calculate the downsized allocation
        allocated = Downsize(that.allocated, that.allocated - size)
        if allocated > SIZE_HMAX:
            raise MemoryError()

        if offset == 0:
            if allocated == that.allocated:
                # Just skip initial if not reallocated and no offset
                that.start += size
            else:
                # Shift elements to make for the deleted gap at the beginning
                offset += that.start
                memmove(&that.data[MARGIN],  # realign to initial MARGIN
                        &that.data[offset + size],
                        (that.endex - (offset + size)) * sizeof(byte_t))
                size = that.endex - that.start - size
                that.start = MARGIN
                that.endex = MARGIN + size
        else:
            # Shift elements to make for the deleted gap at the requested offset
            offset += that.start
            memmove(&that.data[offset],
                    &that.data[offset + size],
                    (that.endex - (offset + size)) * sizeof(byte_t))
            that.endex -= size

        if allocated != that.allocated:
            # Reallocate, including the header
            ptr = <Block_*>PyMem_Realloc(that, Block_HEADING + (allocated * sizeof(byte_t)))
            if ptr == NULL:
                raise MemoryError()

            # Reassign to that
            that = ptr
            that.allocated = allocated

    return that


cdef Block_* Block_Clear(Block_* that) except NULL:
    return Block_Delete_(that, 0, that.endex - that.start)


cdef byte_t* Block_At_(Block_* that, size_t offset) nogil:
    return &that.data[that.start + offset]


cdef const byte_t* Block_At__(const Block_* that, size_t offset) nogil:
    return &that.data[that.start + offset]


cdef byte_t Block_Get__(const Block_* that, size_t offset) nogil:
    return that.data[that.start + offset]


cdef int Block_Get_(const Block_* that, size_t offset) except -1:
    CheckAddSizeU(that.start, offset)
    offset += that.start

    if offset < that.endex:
        return <int><unsigned>that.data[offset]
    else:
        raise IndexError('index out of range')


cdef int Block_Get(const Block_* that, ssize_t offset) except -1:
    if offset < 0:
        offset += <ssize_t>(that.endex - that.start)  # anchor to end
        if offset < 0:
            raise IndexError('index out of range')

    return Block_Get_(that, <size_t>offset)


cdef byte_t Block_Set__(Block_* that, size_t offset, byte_t value) nogil:
    cdef:
        byte_t backup

    offset += that.start
    backup = that.data[offset]
    that.data[offset] = value
    return backup


cdef int Block_Set_(Block_* that, size_t offset, byte_t value) except -1:
    cdef:
        int backup

    # Block_CheckMutable(that)
    CheckAddSizeU(that.start, offset)
    offset += that.start

    if offset < that.endex:
        backup = <int><unsigned>that.data[offset]
        that.data[offset] = value
        return backup
    else:
        raise IndexError('index out of range')


cdef int Block_Set(Block_* that, ssize_t offset, byte_t value) except -1:
    if offset < 0:
        offset += <ssize_t>(that.endex - that.start)  # anchor to end
        if offset < 0:
            raise IndexError('index out of range')

    return Block_Set_(that, <size_t>offset, value)


cdef Block_* Block_Pop__(Block_* that, byte_t* value) except NULL:
    # Block_CheckMutable(that)

    if that.start < that.endex:
        if value:
            value[0] = that.data[that.endex - 1]  # backup

        return Block_Delete_(that, that.endex - that.start - 1, 1)
    else:
        raise IndexError('pop index out of range')


cdef Block_* Block_Pop_(Block_* that, size_t offset, byte_t* value) except NULL:
    # Block_CheckMutable(that)
    CheckAddSizeU(that.start, offset)

    if that.start + offset < that.endex:
        if value:
            value[0] = that.data[that.start + offset]  # backup

        return Block_Delete_(that, offset, 1)
    else:
        raise IndexError('pop index out of range')


cdef Block_* Block_Pop(Block_* that, ssize_t offset, byte_t* value) except NULL:
    if offset < 0:
        offset += <ssize_t>(that.endex - that.start)  # anchor to end
        if offset < 0:
            raise IndexError('pop index out of range')

    return Block_Pop_(that, <size_t>offset, value)


cdef Block_* Block_PopLeft(Block_* that, byte_t* value) except NULL:
    return Block_Pop_(that, 0, value)


cdef Block_* Block_Insert_(Block_* that, size_t offset, byte_t value) except NULL:
    # Insert the value at the requested offset
    that = Block_Reserve_(that, offset, 1, False)
    that.data[that.start + offset] = value
    return that


cdef Block_* Block_Insert(Block_* that, ssize_t offset, byte_t value) except NULL:
    cdef:
        ssize_t size = <ssize_t>(that.endex - that.start)

    if offset < 0:
        offset += size  # anchor to end
        if offset < 0:
            # raise IndexError('index out of range')
            offset = 0  # as per bytearray.insert

    elif offset > size:
        # raise IndexError('index out of range')
        offset = size  # as per bytearray.insert

    return Block_Insert_(that, <size_t>offset, value)


cdef Block_* Block_Append(Block_* that, byte_t value) except NULL:
    # Insert the value after the end
    that = Block_Reserve_(that, that.endex - that.start, 1, False)
    that.data[that.endex - 1] = value
    return that


cdef Block_* Block_AppendLeft(Block_* that, byte_t value) except NULL:
    # Insert the value after the end
    that = Block_Reserve_(that, 0, 1, False)
    that.data[that.start] = value
    return that


cdef Block_* Block_Extend_(Block_* that, size_t size, const byte_t* buffer) except NULL:
    if size:
        that = Block_Reserve_(that, that.endex - that.start, size, False)
        memmove(&that.data[that.endex - size], buffer, size * sizeof(byte_t))
    return that


cdef Block_* Block_Extend(Block_* that, const Block_* more) except NULL:
    that = Block_Extend_(that, Block_Length(more), Block_At__(more, 0))
    return that


cdef Block_* Block_ExtendLeft_(Block_* that, size_t size, const byte_t* buffer) except NULL:
    if size:
        that = Block_Reserve_(that, 0, size, False)
        memmove(&that.data[that.start], buffer, size * sizeof(byte_t))
    return that


cdef Block_* Block_ExtendLeft(Block_* that, const Block_* more) except NULL:
    that = Block_ExtendLeft_(that, Block_Length(more), Block_At__(more, 0))
    return that


cdef void Block_RotateLeft__(Block_* that, size_t offset) nogil:
    cdef:
        size_t size = that.endex - that.start
        byte_t* data = &that.data[that.start]
        byte_t first

    if size:
        if offset == 1:
            first = data[0]
            size -= 1
            while size:
                data[0] = data[1]
                data += 1
                size -= 1
            data[0] = first

        elif offset:
            Reverse(data, 0, offset - 1)
            Reverse(data, offset, size - 1)
            Reverse(data, 0, size - 1)


cdef void Block_RotateLeft_(Block_* that, size_t offset) nogil:
    cdef:
        size_t size = that.endex - that.start

    if size:
        if offset >= size:
            with cython.cdivision(True):
                offset = offset % size  # no "%=" to avoid zero check

        Block_RotateLeft__(that, offset)


cdef void Block_RotateRight__(Block_* that, size_t offset) nogil:
    cdef:
        size_t size = that.endex - that.start
        byte_t* data = &that.data[that.start]
        byte_t last

    if size:
        if offset == 1:
            size -= 1
            if size:
                data += size
                last = data[0]
                while size:
                    size -= 1
                    data -= 1
                    data[1] = data[0]
                data[0] = last

        elif offset:
            offset = size - offset
            Reverse(data, 0, offset - 1)
            Reverse(data, offset, size - 1)
            Reverse(data, 0, size - 1)


cdef void Block_RotateRight_(Block_* that, size_t offset) nogil:
    cdef:
        size_t size = that.endex - that.start

    if size:
        if offset >= size:
            with cython.cdivision(True):
                offset = offset % size  # no "%=" to avoid zero check

        Block_RotateRight__(that, offset)


cdef void Block_Rotate(Block_* that, ssize_t offset) nogil:
    if offset < 0:
        Block_RotateLeft_(that, <size_t>-offset)
    else:
        Block_RotateRight_(that, <size_t>offset)


cdef Block_* Block_Repeat(Block_* that, size_t times) except NULL:
    cdef:
        size_t size
        byte_t* src
        byte_t* dst

    if times == 1:
        return that

    elif times < 1:
        return Block_Clear(that)

    else:
        size = that.endex - that.start
        with cython.cdivision(True):
            if size > SIZE_HMAX // times:
                raise OverflowError()

        times -= 1
        that = Block_Reserve_(that, size, size * times, False)
        src = &that.data[that.start]
        dst = src

        while times:
            times -= 1
            dst += size
            memcpy(dst, src, size)  # whole repetition

        return that


cdef Block_* Block_RepeatToSize(Block_* that, size_t size) except NULL:
    cdef:
        size_t size2
        size_t times
        byte_t* src
        byte_t* dst

    size2 = that.endex - that.start

    if size2 == 0:
        raise RuntimeError('empty')

    if size == size2:
        return that

    elif size < size2:
        return Block_DelSlice_(that, size, size2)

    else:  # size > size2
        that = Block_Reserve_(that, size2, size - size2, False)

        if that.start + 1 == that.endex:  # single byte
            dst = &that.data[that.start]
            memset(dst, dst[0], size)

        else:  # multiple bytes
            with cython.cdivision(True):
                times = size // size2

            # Copy the final partial chunk
            src = &that.data[that.start]
            dst = &that.data[that.start + (size2 * times)]
            memcpy(dst, src, size - (size2 * times))

            # Copy the multiple times, skipping the first one
            dst = src + size2
            times -= 1
            while times:
                memcpy(dst, src, size2)
                dst += size2
                times -= 1

        return that


cdef vint Block_Read_(const Block_* that, size_t offset, size_t size, byte_t* buffer) except -1:
    if size:
        if size > SIZE_HMAX:
            raise OverflowError('size overflow')

        CheckAddSizeU(offset, that.start)
        offset += that.start

        CheckAddSizeU(offset, size)
        if that.endex < offset + size:
            raise IndexError('index out of range')

        memmove(buffer, &that.data[offset], size * sizeof(byte_t))


cdef Block_* Block_Write_(Block_* that, size_t offset, size_t size, const byte_t* buffer) except NULL:
    # Block_CheckMutable(that)

    if size:
        CheckAddSizeU(that.start, offset)
        offset += that.start

        CheckAddSizeU(offset, size)
        if that.endex < offset + size:
            that = Block_Reserve_(that, that.endex - that.start, (offset + size) - that.endex, False)

        memmove(&that.data[offset], buffer, size * sizeof(byte_t))
    return that


cdef vint Block_ReadSlice_(const Block_* that, size_t start, size_t endex,
                           size_t* size_, byte_t* buffer) except -1:
    cdef:
        size_t size = that.endex - that.start

    size_[0] = 0

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim source start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative source length
    elif endex > size:
        endex = size  # trim source end

    size = endex - start
    Block_Read_(that, start, size, buffer)
    size_[0] = size


cdef vint Block_ReadSlice(const Block_* that, ssize_t start, ssize_t endex,
                          size_t* size_, byte_t* buffer) except -1:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative source length

    Block_ReadSlice_(that, <size_t>start, <size_t>endex, size_, buffer)


cdef Block_* Block_GetSlice_(const Block_* that, size_t start, size_t endex) except NULL:
    cdef:
        size_t size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim source start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative source length
    elif endex > size:
        endex = size  # trim source end

    return Block_Create(that.address + start, endex - start, &that.data[that.start + start])


cdef Block_* Block_GetSlice(const Block_* that, ssize_t start, ssize_t endex) except NULL:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative source length

    return Block_GetSlice_(that, <size_t>start, <size_t>endex)


cdef Block_* Block_WriteSlice_(Block_* that, size_t start, size_t endex,
                               size_t size, const byte_t* buffer) except NULL:
    cdef:
        size_t size2   # source size

    size2 = size
    size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim target start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex > size:
        endex = size  # trim target end

    if endex < start:
        endex = start  # clamp negative target length
    size = endex - start

    if size2 > size:  # enlarge target at range end
        that = Block_Reserve_(that, endex, size2 - size, False)

    elif size > size2:  # shrink target at range end
        endex -= size - size2
        that = Block_Delete_(that, endex, size - size2)

    that = Block_Write_(that, start, size2, buffer)
    return that


cdef Block_* Block_WriteSlice(Block_* that, ssize_t start, ssize_t endex,
                              size_t size, const byte_t* buffer) except NULL:
    cdef:
        ssize_t ssize   # target size
        ssize_t ssize2  # source size
        ssize_t start2  # source start
        ssize_t endex2  # source end

    start2 = 0
    endex2 = <ssize_t>size

    ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        # start2 -= start  # skip initial source data  # as per bytearray
        start = 0  # trim target start
    if start2 > endex2:
        start2 = endex2  # clamp source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative target length

    if endex2 < start2:
        endex2 = start2  # clamp negative source length
    ssize2 = endex2 - start2

    that = Block_WriteSlice_(that, <size_t>start, <size_t>endex, <size_t>ssize2, &buffer[start2])
    return that


cdef Block_* Block_SetSlice_(Block_* that, size_t start, size_t endex,
                             const Block_* src, size_t start2, size_t endex2) except NULL:
    cdef:
        size_t size    # target size
        size_t size2   # source size

    size2 = src.endex - src.start

    if start2 > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start2 > size2:
        start2 = size2  # trim source start

    if endex2 > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex2 > size2:
        endex2 = size2  # trim source end

    if endex2 < start2:
        endex2 = start2  # clamp negative source length
    size2 = endex2 - start2

    size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim target start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex > size:
        endex = size  # trim target end

    if endex < start:
        endex = start  # clamp negative target length
    size = endex - start

    if size2 > size:  # enlarge target at range end
        that = Block_Reserve_(that, endex, size2 - size, False)

    elif size > size2:  # shrink target at range end
        endex -= size - size2
        that = Block_Delete_(that, endex, size - size2)

    that = Block_Write_(that, start, size2, &src.data[src.start + start2])
    return that


cdef Block_* Block_SetSlice(Block_* that, ssize_t start, ssize_t endex,
                            const Block_* src, ssize_t start2, ssize_t endex2) except NULL:
    cdef:
        ssize_t ssize   # target size
        ssize_t ssize2  # source size

    ssize = <ssize_t>(that.endex - that.start)
    ssize2 = <ssize_t>(src.endex - src.start)

    if start < 0:
        start += ssize  # anchor to target end
    if start < 0:
        # start2 -= start  # skip initial source data  # as per bytearray
        start = 0  # trim target start

    if endex < 0:
        endex += ssize  # anchor to target end
    if endex < start:
        endex = start  # clamp negative target length

    if start2 < 0:
        start2 += ssize2  # anchor to source end
    if start2 < 0:
        start2 = 0  # trim source start

    if endex2 < 0:
        endex2 += ssize2  # anchor to source end
    if endex2 < start2:
        endex2 = start2  # clamp negative source length

    that = Block_SetSlice_(that, <size_t>start, <size_t>endex, src, <size_t>start2, <size_t>endex2)
    return that


cdef Block_* Block_DelSlice_(Block_* that, size_t start, size_t endex) except NULL:
    cdef:
        size_t size

    size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative length
    elif endex > size:
        endex = size  # trim end

    that = Block_Delete_(that, start, (endex - start))
    return that


cdef Block_* Block_DelSlice(Block_* that, ssize_t start, ssize_t endex) except NULL:
    cdef:
        ssize_t ssize

    ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative length

    that = Block_DelSlice_(that, <size_t>start, <size_t>endex)
    return that


cdef bytes Block_Bytes(const Block_* that):
    cdef:
        char* ptr = <char*><void*>&that.data[that.start]
        ssize_t size = <ssize_t>(that.endex - that.start)

    return PyBytes_FromStringAndSize(ptr, size)


cdef bytearray Block_Bytearray(const Block_* that):
    cdef:
        char* ptr = <char*><void*>&that.data[that.start]
        ssize_t size = <ssize_t>(that.endex - that.start)

    return PyByteArray_FromStringAndSize(ptr, size)


cdef BlockView Block_View(Block_* that):
    cdef:
        BlockView view

    view = BlockView()
    that = Block_Acquire(that)
    view._block = that
    view._start = that.start
    view._endex = that.endex
    return view


cdef BlockView Block_ViewSlice_(Block_* that, size_t start, size_t endex):
    cdef:
        size_t size = that.endex - that.start
        BlockView view

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim source start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative source length
    elif endex > size:
        endex = size  # trim source end

    view = BlockView()
    that = Block_Acquire(that)
    view._block = that
    view._start = that.start + start
    view._endex = that.start + endex
    return view


cdef BlockView Block_ViewSlice(Block_* that, ssize_t start, ssize_t endex):
    cdef:
        ssize_t ssize

    ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative source length

    return Block_ViewSlice_(that, <size_t>start, <size_t>endex)


# ---------------------------------------------------------------------------------------------------------------------

cdef class BlockView:
    r"""Block viewer.

    Memory view around an underlying block slice, implementing Python's `buffer`
    protocol API.
    """

    cdef vint check_(BlockView self) except -1:
        if self._block == NULL:
            raise RuntimeError('null internal data pointer')

    cdef vint release_(BlockView self) except -1:
        if self._memview is not None:
            self._memview = None
            self._block = Block_Release_(self._block)

        self._block = Block_Release(self._block)

    def __bool__(
        self: BlockView,
    ) -> bool:
        r"""Has any data.

        Returns:
            bool: Non-null slice length.
        """

        self.check_()
        return self._start < self._endex

    def __bytes__(
        self: BlockView,
    ) -> bytes:
        r"""Converts into bytes.

        Returns:
            bytes: :class:`bytes` clone of the viewed slice.
        """
        cdef:
            char* ptr
            ssize_t size
            Block_* block = self._block

        self.check_()
        ptr = <char*><void*>&block.data[self._start]
        size = <ssize_t>(self._endex - self._start)
        return PyBytes_FromStringAndSize(ptr, size)

    def __cinit__(self: BlockView):
        self._block = NULL

    def __dealloc__(self: BlockView):
        self.release_()

    def __eq__(
        self: BlockView,
        other: Any,
    ) -> bool:

        self.check_()
        if other is None:
            return False

        if isinstance(other, BlockView):
            return Block_Eq(self._block, (<BlockView>other)._block)
        else:
            return self.memview == other

    def __getattr__(
        self: BlockView,
        attr: str,
    ) -> Any:

        self.check_()
        return getattr(self.memview, attr)

    def __getbuffer__(self: BlockView, Py_buffer* buffer, int flags):
        cdef:
            int CONTIGUOUS = PyBUF_C_CONTIGUOUS | PyBUF_F_CONTIGUOUS | PyBUF_ANY_CONTIGUOUS

        # if flags & PyBUF_WRITABLE:
        #     raise ValueError('read only access')

        self.check_()

        self._block = Block_Acquire(self._block)

        buffer.buf = &self._block.data[self._start]
        buffer.obj = self
        buffer.len = self._endex - self._start
        buffer.itemsize = 1
        buffer.readonly = not (flags & PyBUF_WRITABLE)
        buffer.ndim = 1
        buffer.format = <char*>'B' if flags & (PyBUF_FORMAT | CONTIGUOUS) else NULL
        buffer.shape = &buffer.len if flags & (PyBUF_ND | CONTIGUOUS) else NULL
        buffer.strides = &buffer.itemsize if flags & (PyBUF_STRIDES | CONTIGUOUS) else NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __getitem__(
        self: BlockView,
        item: Any,
    ) -> Any:

        self.check_()
        return self.memview[item]

    def __len__(
        self: BlockView,
    ) -> Address:
        r"""int: Slice length."""

        self.check_()
        return self._endex - self._start

    def __releasebuffer__(self: BlockView, Py_buffer* buffer):
        if self._block:
            self._block = Block_Release_(self._block)

    def __repr__(
        self: BlockView,
    ) -> str:

        return repr(self.__str__())

    def __sizeof__(
        self: BlockView,
    ) -> int:
        r"""int: Allocated byte size."""

        return sizeof(BlockView)

    def __str__(
        self: BlockView,
    ) -> str:
        cdef:
            const Block_* block = self._block
            size_t size = self._endex - self._start
            addr_t start
            addr_t endex

        self.check_()

        if size > STR_MAX_CONTENT_SIZE:
            start = block.address
            CheckAddAddrU(start, size)
            endex = start + size
            return f'<{type(self).__name__}[0x{start:X}:0x{endex:X}]@0x{<uintptr_t><void*>self:X}>'
        else:
            return self.__bytes__().decode('ascii')

    @property
    def acquired(
        self: BlockView,
    ) -> bool:
        r"""bool: Underlying block currently acquired."""

        return self._block != NULL

    def check(
        self: BlockView,
    ) -> None:
        r"""Checks for data consistency."""

        self.check_()

    def release(
        self: BlockView,
    ) -> None:
        r"""Forces object disposal.

        Useful to make sure that any memory blocks are unreferenced before automatic
        garbage collection.

        Any access to the object after calling this function could raise exceptions.
        """

        self.release_()

    @property
    def endex(
        self: BlockView,
    ) -> Address:
        r"""int: Slice exclusive end address."""

        self.check_()
        return self._block.address + self._endex - self._start

    @property
    def endin(
        self: BlockView,
    ) -> Address:
        r"""int: Slice inclusive end address."""

        return self.endex - 1

    @property
    def memview(
        self: BlockView,
    ) -> memoryview:
        r"""memoryview: Python :class:`memoryview` wrapper."""
        cdef:
            byte_t* data
            size_t size
            byte_t[:] view

        self.check_()

        if self._memview is None:
            self._block = Block_Acquire(self._block)
            data = &self._block.data[self._start]
            size = self._endex - self._start

            if size > 0:
                self._memview = <byte_t[:size]>data
            else:
                self._memview = b''

        return self._memview

    @property
    def start(
        self: BlockView,
    ) -> Address:
        r"""int: Slice inclusive start address."""

        self.check_()
        return self._block.address


# =====================================================================================================================

cdef Rack_* Rack_Alloc(size_t size) except NULL:
    cdef:
        Rack_* that = NULL
        size_t allocated
        size_t actual

    if size > SIZE_HMAX:
        raise OverflowError('size overflow')

    # Allocate as per request
    allocated = Upsize(0, size)
    if allocated > SIZE_HMAX:
        raise MemoryError()

    actual = Rack_HEADING + (allocated * sizeof(Block_*))
    that = <Rack_*>PyMem_Calloc(actual, 1)
    if that == NULL:
        raise MemoryError()

    that.allocated = allocated
    that.start = MARGIN  # leave some initial room
    that.endex = that.start + size
    return that


cdef Rack_* Rack_Free(Rack_* that):
    cdef:
        size_t index

    if that:
        # Decrement data referencing
        for index in range(that.start, that.endex):
            that.blocks[index] = Block_Release(that.blocks[index])
        PyMem_Free(that)
    return NULL


cdef size_t Rack_Sizeof(const Rack_* that):
    cdef:
        size_t index
        size_t size

    if that:
        size = Rack_HEADING
        for index in range(that.start, that.endex):
            size += Block_Sizeof(that.blocks[index])
        return size
    return 0


cdef Rack_* Rack_ShallowCopy(const Rack_* other) except NULL:
    cdef:
        Rack_* that = Rack_Alloc(other.endex - other.start)
        size_t start1 = that.start
        size_t start2 = other.start
        size_t offset

    try:
        for offset in range(that.endex - that.start):
            that.blocks[start1 + offset] = Block_Acquire(other.blocks[start2 + offset])
    except:
        that = Rack_Free(that)
        raise
    return that


cdef Rack_* Rack_Copy(const Rack_* other) except NULL:
    cdef:
        Rack_* that = Rack_Alloc(other.endex - other.start)
        size_t start1 = that.start
        size_t start2 = other.start
        size_t offset

    try:
        for offset in range(that.endex - that.start):
            that.blocks[start1 + offset] = Block_Copy(other.blocks[start2 + offset])
    except:
        that = Rack_Free(that)
        raise
    return that


cdef Rack_* Rack_FromObject(object obj, saddr_t offset) except NULL:
    cdef:
        Rack_* that = NULL
        size_t size
        size_t index
        addr_t address

    try:
        try:
            size = len(obj)
        except TypeError:
            that = Rack_Alloc(0)
            for address, data in obj:
                if offset < 0:
                    CheckSubAddrU(address, <addr_t>-offset)
                    address -= <addr_t>-offset
                elif offset > 0:
                    CheckAddAddrU(address, <addr_t>offset)
                    address += <addr_t>offset
                that = Rack_Append(that, Block_FromObject(address, data, True))
        else:
            that = Rack_Alloc(size)
            index = that.start
            for address, data in obj:
                if offset < 0:
                    CheckSubAddrU(address, <addr_t>-offset)
                    address -= <addr_t>-offset
                elif offset > 0:
                    CheckAddAddrU(address, <addr_t>offset)
                    address += <addr_t>offset
                that.blocks[index] = Block_FromObject(address, data, True)
                index += 1
        return that

    except:
        that = Rack_Free(that)
        raise


cdef void Rack_Reverse(Rack_* that) nogil:
    cdef:
        size_t index_start = that.start
        size_t index_endex = that.endex
        size_t index_endin
        Block_* block

    while index_start < index_endex:
        index_endin = index_endex - 1
        block = that.blocks[index_start]
        that.blocks[index_start] = that.blocks[index_endin]
        that.blocks[index_endin] = block
        index_endex = index_endin
        index_start += 1


cdef bint Rack_Bool(const Rack_* that) nogil:
    return that.start < that.endex


cdef size_t Rack_Length(const Rack_* that) nogil:
    return that.endex - that.start


cdef (addr_t, addr_t) Rack_BoundSlice(const Rack_* that, addr_t start, addr_t endex) nogil:
    cdef:
        const Block_* block
        addr_t block_start
        addr_t block_endex

    if that.start < that.endex:
        block = that.blocks[that.start]
        block_start = Block_Start(block)
        if start < block_start:
            start = block_start
        if endex < start:
            endex = start

        block = that.blocks[that.endex - 1]
        block_endex = Block_Endex(block)
        if endex > block_endex:
            endex = block_endex
        if start > endex:
            start = endex
    else:
        start = 0
        endex = 0

    return start, endex


cdef Rack_* Rack_Shift_(Rack_* that, addr_t offset) except NULL:
    cdef:
        size_t index
        Block_* block

    if offset:
        if that.start < that.endex:
            block = that.blocks[that.endex - 1]
            CheckAddAddrU(block.address, offset)

            for index in range(that.start, that.endex):
                block = that.blocks[index]
                block.address += offset
    return that


cdef Rack_* Rack_Shift(Rack_* that, saddr_t offset) except NULL:
    cdef:
        size_t index
        Block_* block
        addr_t offset_

    if offset:
        if that.start < that.endex:
            if offset < 0:
                block = that.blocks[that.start]
                offset_ = <addr_t>-offset
                CheckSubAddrU(block.address, offset_)

                for index in range(that.start, that.endex):
                    block = that.blocks[index]
                    block.address -= offset_
            else:
                block = that.blocks[that.endex - 1]
                offset_ = <addr_t>offset
                CheckAddAddrU(block.address, offset_)

                for index in range(that.start, that.endex):
                    block = that.blocks[index]
                    block.address += offset_
    return that


cdef bint Rack_Eq(const Rack_* that, const Rack_* other) except -1:
    cdef:
        size_t block_count = that.endex - that.start
        size_t block_index
        const Block_* block1
        const Block_* block2

    if block_count != other.endex - other.start:
        return False

    for block_index in range(block_count):
        block1 = Rack_Get__(that, block_index)
        block2 = Rack_Get__(other, block_index)

        if block1.address != block2.address:
            return False

        if not Block_Eq(block1, block2):
            return False

    return True


cdef Rack_* Rack_Reserve_(Rack_* that, size_t offset, size_t size) except NULL:
    cdef:
        size_t actual
        size_t used
        size_t margin
        size_t allocated
        Rack_* ptr
        size_t index
        Block_* node

    if size:
        if ((size > SIZE_HMAX or
             CannotAddSizeU(that.endex, size) or
             that.endex + size > SIZE_HMAX)):
            raise OverflowError('size overflow')

        used = that.endex - that.start
        if offset > used:
            raise IndexError('index out of range')

        # Prefer the side where there is less data to shift, which also favors the extremes
        if offset >= (used >> 1):
            if size > that.allocated - that.endex:
                # Calculate the upsized allocation
                allocated = Upsize(that.allocated, that.allocated + size)
                if allocated > SIZE_HMAX:
                    raise MemoryError()

                # Reallocate, including the header
                ptr = <Rack_*>PyMem_Realloc(that, Rack_HEADING + (allocated * sizeof(Block_*)))
                if ptr == NULL:
                    raise MemoryError()

                # Reassign to that
                that = ptr
                that.allocated = allocated  # update

            # Shift elements to make room for reservation at the requested offset
            CheckAddSizeU(offset, that.start)
            offset += that.start
            used = that.endex - offset
            if used:
                memmove(&that.blocks[offset + size],
                        &that.blocks[offset],
                        used * sizeof(Block_*))

            memset(&that.blocks[offset], 0, size * sizeof(Block_*))  # pad with zeros
            that.endex += size

        else:
            if size <= that.start:
                # Shift elements to make room for reservation at the requested offset
                that.start -= size
                if offset:
                    memmove(&that.blocks[that.start],
                            &that.blocks[that.start + size],
                            offset * sizeof(Block_*))

                memset(&that.blocks[that.start + offset], 0, size * sizeof(Block_*))  # pad with zeros

            else:
                # Calculate the upsized allocation
                CheckAddSizeU(that.allocated, size)
                allocated = Upsize(that.allocated, that.allocated + size)
                if allocated > SIZE_HMAX:
                    raise MemoryError()

                # Allocate a new chunk, including the header
                actual = Rack_HEADING + (allocated * sizeof(Block_*))
                ptr = <Rack_*>PyMem_Calloc(actual, 1)
                if ptr == NULL:
                    raise MemoryError()

                # Prepare the new chunk aligning towards the end
                ptr.allocated = allocated
                ptr.endex = ptr.allocated - MARGIN  # leave some room
                ptr.start = ptr.endex - used - size

                # Shift/copy elements to make room for reservation at the requested offset
                if offset:
                    used -= offset  # prepare for later
                    memcpy(&ptr.blocks[ptr.start],
                           &that.blocks[that.start],
                           offset * sizeof(Block_*))
                if used:
                    memcpy(&ptr.blocks[ptr.start + offset + size],
                           &that.blocks[that.start + offset],
                           used * sizeof(Block_*))

                # Reassign to that
                PyMem_Free(that)
                that = ptr

    return that


cdef Rack_* Rack_Delete_(Rack_* that, size_t offset, size_t size) except NULL:
    cdef:
        size_t allocated
        Rack_* ptr
        size_t index
        Block_* node

    if size:
        if ((size > SIZE_HMAX or
             CannotAddSizeU(offset, size) or
             offset + size > SIZE_HMAX or
             CannotAddSizeU(offset, that.start) or
             that.start > SIZE_HMAX)):
            raise OverflowError('size overflow')

        if that.endex < that.start + offset + size:
            raise IndexError('index out of range')

        # Calculate the downsized allocation
        allocated = Downsize(that.allocated, that.allocated - size)
        if allocated > SIZE_HMAX:
            raise MemoryError()

        # Release blocks within the deleted range
        offset += that.start
        for index in range(offset, offset + size):
            that.blocks[index] = Block_Release(that.blocks[index])

        if offset == 0:
            if allocated == that.allocated:
                # Just skip initial if not reallocated and no offset
                memset(&that.blocks[that.start], 0, size * sizeof(Block_*))  # cleanup margin
                that.start += size
            else:
                # Shift elements to make for the deleted gap at the beginning
                offset += that.start
                memmove(&that.blocks[MARGIN],  # realign to initial MARGIN
                        &that.blocks[offset + size],
                        (that.endex - (offset + size)) * sizeof(Block_*))
                size = that.endex - that.start - size
                that.start = MARGIN
                that.endex = MARGIN + size

                # Cleanup margins
                memset(&that.blocks[0], 0, that.start * sizeof(Block_*))
                memset(&that.blocks[that.endex], 0, (that.allocated - that.endex) * sizeof(Block_*))
        else:
            # Shift elements to make for the deleted gap at the requested offset
            memmove(&that.blocks[offset],
                    &that.blocks[offset + size],
                    (that.endex - (offset + size)) * sizeof(Block_*))
            that.endex -= size
            memset(&that.blocks[that.endex], 0, size * sizeof(Block_*))  # cleanup margin

        if allocated != that.allocated:
            # Reallocate, including the header
            ptr = <Rack_*>PyMem_Realloc(that, Rack_HEADING + (allocated * sizeof(Block_*)))
            if ptr == NULL:
                raise MemoryError()

            # Reassign to that
            that = ptr
            that.allocated = allocated

    return that


cdef Rack_* Rack_Clear(Rack_* that) except NULL:
    return Rack_Delete_(that, 0, that.endex - that.start)


cdef Rack_* Rack_Consolidate(Rack_* that) except NULL:
    cdef:
        size_t offset
        Block_* block

    for offset in range(that.start, that.endex):
        block = that.blocks[offset]
        if block.references > 1:
            that.blocks[offset] = Block_Copy(block)
            block = Block_Release(block)
    return that


cdef Block_** Rack_At_(Rack_* that, size_t offset) nogil:
    return &that.blocks[that.start + offset]


cdef const Block_** Rack_At__(const Rack_* that, size_t offset) nogil:
    return <const Block_**>&that.blocks[that.start + offset]


cdef Block_* Rack_First_(Rack_* that) nogil:
    return that.blocks[that.start]


cdef const Block_* Rack_First__(const Rack_* that) nogil:
    return that.blocks[that.start]


cdef Block_* Rack_Last_(Rack_* that) nogil:
    return that.blocks[that.endex - 1]


cdef const Block_* Rack_Last__(const Rack_* that) nogil:
    return that.blocks[that.endex - 1]


cdef Block_* Rack_Get__(const Rack_* that, size_t offset) nogil:
    return that.blocks[that.start + offset]


cdef Block_* Rack_Get_(const Rack_* that, size_t offset) except? NULL:
    CheckAddSizeU(that.start, offset)
    offset += that.start

    if offset < that.endex:
        return that.blocks[offset]
    else:
        raise IndexError('index out of range')


cdef Block_* Rack_Get(const Rack_* that, ssize_t offset) except? NULL:
    if offset < 0:
        offset += <ssize_t>(that.endex - that.start)  # anchor to end
        if offset < 0:
            raise IndexError('index out of range')

    return Rack_Get_(that, <size_t>offset)


cdef Block_* Rack_Set__(Rack_* that, size_t offset, Block_* value) nogil:
    cdef:
        Block_* backup

    offset += that.start
    backup = that.blocks[offset]
    that.blocks[offset] = value
    return backup


cdef vint Rack_Set_(Rack_* that, size_t offset, Block_* value, Block_** backup) except -1:
    CheckAddSizeU(that.start, offset)
    offset += that.start

    if offset < that.endex:
        if backup:
            backup[0] = that.blocks[offset]
        else:
            that.blocks[offset] = Block_Release(that.blocks[offset])
        that.blocks[offset] = value
    else:
        if backup:
            backup[0] = NULL
        raise IndexError('index out of range')


cdef vint Rack_Set(Rack_* that, ssize_t offset, Block_* value, Block_** backup) except -1:
    if offset < 0:
        offset += <ssize_t>(that.endex - that.start)  # anchor to end
        if offset < 0:
            raise IndexError('index out of range')

    Rack_Set_(that, <size_t>offset, value, backup)


cdef Rack_* Rack_Pop__(Rack_* that, Block_** value) except NULL:
    if that.start < that.endex:
        if value:
            value[0] = Block_Acquire(that.blocks[that.endex - 1])  # backup

        return Rack_Delete_(that, that.endex - that.start - 1, 1)
    else:
        if value:
            value[0] = NULL
        raise IndexError('pop index out of range')


cdef Rack_* Rack_Pop_(Rack_* that, size_t offset, Block_** value) except NULL:
    CheckAddSizeU(that.start, offset)

    if that.start + offset < that.endex:
        if value:
            value[0] = Block_Acquire(that.blocks[that.start + offset])  # backup

        return Rack_Delete_(that, offset, 1)
    else:
        if value:
            value[0] = NULL
        raise IndexError('pop index out of range')


cdef Rack_* Rack_Pop(Rack_* that, ssize_t offset, Block_** value) except NULL:
    if offset < 0:
        offset += <ssize_t>(that.endex - that.start)  # anchor to end
        if offset < 0:
            raise IndexError('pop index out of range')

    return Rack_Pop_(that, <size_t>offset, value)


cdef Rack_* Rack_PopLeft(Rack_* that, Block_** value) except NULL:
    return Rack_Pop_(that, 0, value)


cdef Rack_* Rack_Insert_(Rack_* that, size_t offset, Block_* value) except NULL:
    # Insert the value at the requested offset
    that = Rack_Reserve_(that, offset, 1)
    that.blocks[that.start + offset] = value
    return that


cdef Rack_* Rack_Insert(Rack_* that, ssize_t offset, Block_* value) except NULL:
    cdef:
        ssize_t size = <ssize_t>(that.endex - that.start)

    if offset < 0:
        offset += size  # anchor to end
        if offset < 0:
            # raise IndexError('index out of range')
            offset = 0  # as per bytearray.insert

    elif offset > size:
        # raise IndexError('index out of range')
        offset = size  # as per bytearray.insert

    return Rack_Insert_(that, <size_t>offset, value)


cdef Rack_* Rack_Append(Rack_* that, Block_* value) except NULL:
    # Insert the value after the end
    that = Rack_Reserve_(that, that.endex - that.start, 1)
    that.blocks[that.endex - 1] = value
    return that


cdef Rack_* Rack_AppendLeft(Rack_* that, Block_* value) except NULL:
    # Insert the value after the end
    that = Rack_Reserve_(that, 0, 1)
    that.blocks[that.start] = value
    return that


cdef Rack_* Rack_Extend_(Rack_* that, size_t size, Block_** buffer, bint direct) except NULL:
    cdef:
        size_t start
        size_t offset

    if size:
        that = Rack_Reserve_(that, that.endex - that.start, size)
        if direct:
            memmove(&that.blocks[that.endex - size], buffer, size * sizeof(Block_*))
        else:
            start = that.endex - size
            for offset in range(size):
                that.blocks[start + offset] = Block_Acquire(buffer[offset])
    return that


cdef Rack_* Rack_Extend(Rack_* that, Rack_* more) except NULL:
    that = Rack_Extend_(that, more.endex - more.start, &more.blocks[more.start], False)
    return that


cdef Rack_* Rack_ExtendLeft_(Rack_* that, size_t size, Block_** buffer, bint direct) except NULL:
    cdef:
        size_t start
        size_t offset

    if size:
        that = Rack_Reserve_(that, 0, size)
        if direct:
            memmove(&that.blocks[that.endex - size], buffer, size * sizeof(Block_*))
        else:
            start = that.start
            for offset in range(size):
                that.blocks[start + offset] = Block_Acquire(buffer[offset])
    return that


cdef Rack_* Rack_ExtendLeft(Rack_* that, Rack_* more) except NULL:
    that = Rack_ExtendLeft_(that, more.endex - more.start, &more.blocks[more.start], False)
    return that


cdef vint Rack_Read_(const Rack_* that, size_t offset,
                     size_t size, Block_** buffer, bint direct) except -1:
    if size:
        if size > SIZE_HMAX:
            raise OverflowError('size overflow')

        CheckAddSizeU(that.start, offset)
        offset += that.start

        CheckAddSizeU(offset, size)
        if that.endex <= offset + size:
            raise IndexError('index out of range')

        if direct:
            memmove(buffer, &that.blocks[offset], size * sizeof(Block_*))
        else:
            for offset in range(offset, offset + size):
                buffer[offset - that.start] = Block_Acquire(buffer[offset])


cdef Rack_* Rack_Write_(Rack_* that, size_t offset,
                        size_t size, Block_** buffer, bint direct) except NULL:
    if size:
        CheckAddSizeU(that.start, offset)
        offset += that.start

        CheckAddSizeU(offset, size)
        if that.endex < offset + size:
            that = Rack_Reserve_(that, that.endex - that.start, (offset + size) - that.endex)

        if direct:
            memmove(&that.blocks[offset], buffer, size * sizeof(Block_*))
        else:
            for offset in range(offset, offset + size):
                that.blocks[offset] = Block_Release(that.blocks[offset])
                that.blocks[offset] = Block_Acquire(buffer[offset - that.start])

    return that


cdef vint Rack_ReadSlice_(const Rack_* that, size_t start, size_t endex,
                          size_t* size_, Block_** buffer, bint direct) except -1:
    cdef:
        size_t size = that.endex - that.start

    size_[0] = 0

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim source start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative source length
    elif endex > size:
        endex = size  # trim source end

    size = endex - start
    Rack_Read_(that, start, size, buffer, direct)
    size_[0] = size


cdef vint Rack_ReadSlice(const Rack_* that, ssize_t start, ssize_t endex,
                         size_t* size_, Block_** buffer, bint direct) except -1:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative source length

    Rack_ReadSlice_(that, <size_t>start, <size_t>endex, size_, buffer, direct)


cdef Rack_* Rack_GetSlice_(const Rack_* that, size_t start, size_t endex) except NULL:
    cdef:
        Rack_* other = NULL
        size_t size = that.endex - that.start
        size_t offset
        size_t offset2

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim source start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative source length
    elif endex > size:
        endex = size  # trim source end

    try:
        size = endex - start
        other = Rack_Alloc(size)
        CheckAddSizeU(other.start, size)
        offset2 = that.start + start

        for offset in range(other.start, other.start + size):
            other.blocks[offset] = Block_Acquire(that.blocks[offset2])
            offset2 += 1

        return other
    except:
        other = Rack_Free(other)
        raise


cdef Rack_* Rack_GetSlice(const Rack_* that, ssize_t start, ssize_t endex) except NULL:
    cdef:
        ssize_t ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative source length

    return Rack_GetSlice_(that, <size_t>start, <size_t>endex)


cdef Rack_* Rack_WriteSlice_(Rack_* that, size_t start, size_t endex,
                             size_t size, Block_** buffer, bint direct) except NULL:
    cdef:
        size_t size2   # source size

    size2 = size
    size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim target start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex > size:
        endex = size  # trim target end

    if endex < start:
        endex = start  # clamp negative target length
    size = endex - start

    if size2 > size:  # enlarge target at range end
        that = Rack_Reserve_(that, endex, size2 - size)

    elif size > size2:  # shrink target at range end
        endex -= size - size2
        that = Rack_Delete_(that, endex, size - size2)

    that = Rack_Write_(that, start, size2, buffer, direct)
    return that


cdef Rack_* Rack_WriteSlice(Rack_* that, ssize_t start, ssize_t endex,
                            size_t size, Block_** buffer, bint direct) except NULL:
    cdef:
        ssize_t ssize   # target size
        ssize_t ssize2  # source size
        ssize_t start2  # source start
        ssize_t endex2  # source end

    start2 = 0
    endex2 = <ssize_t>size

    ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        # start2 -= start  # skip initial source data  # as per bytearray
        start = 0  # trim target start
    if start2 > endex2:
        start2 = endex2  # clamp source start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative target length

    if endex2 < start2:
        endex2 = start2  # clamp negative source length
    ssize2 = endex2 - start2

    that = Rack_WriteSlice_(that, <size_t>start, <size_t>endex, <size_t>ssize2, &buffer[start2], direct)
    return that


cdef Rack_* Rack_SetSlice_(Rack_* that, size_t start, size_t endex,
                           Rack_* src, size_t start2, size_t endex2) except NULL:
    cdef:
        size_t size   # target size
        size_t size2  # source size

    size2 = src.endex - src.start

    if start2 > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start2 > size2:
        start2 = size2  # trim source start

    if endex2 > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex2 > size2:
        endex2 = size2  # trim source end

    if endex2 < start2:
        endex2 = start2  # clamp negative source length
    size2 = endex2 - start2

    size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim target start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex > size:
        endex = size  # trim target end

    if endex < start:
        endex = start  # clamp negative target length
    size = endex - start

    if size2 > size:  # enlarge target at range end
        that = Rack_Reserve_(that, endex, size2 - size)

    elif size > size2:  # shrink target at range end
        endex -= size - size2
        that = Rack_Delete_(that, endex, size - size2)

    that = Rack_Write_(that, start, size2, &src.blocks[src.start + start2], False)
    return that


cdef Rack_* Rack_SetSlice(Rack_* that, ssize_t start, ssize_t endex,
                          Rack_* src, ssize_t start2, ssize_t endex2) except NULL:
    cdef:
        ssize_t ssize   # target size
        ssize_t ssize2  # source size

    ssize = <ssize_t>(that.endex - that.start)
    ssize2 = <ssize_t>(src.endex - src.start)

    if start < 0:
        start += ssize  # anchor to target end
    if start < 0:
        # start2 -= start  # skip initial source data  # as per bytearray
        start = 0  # trim target start

    if endex < 0:
        endex += ssize  # anchor to target end
    if endex < start:
        endex = start  # clamp negative target length

    if start2 < 0:
        start2 += ssize2  # anchor to source end
    if start2 < 0:
        start2 = 0  # trim source start

    if endex2 < 0:
        endex2 += ssize2  # anchor to source end
    if endex2 < start2:
        endex2 = start2  # clamp negative source length

    that = Rack_SetSlice_(that, <size_t>start, <size_t>endex, src, <size_t>start2, <size_t>endex2)
    return that


cdef Rack_* Rack_DelSlice_(Rack_* that, size_t start, size_t endex) except NULL:
    cdef:
        size_t size

    size = that.endex - that.start

    if start > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif start > size:
        start = size  # trim start

    if endex > SIZE_HMAX:
        raise OverflowError('size overflow')
    elif endex < start:
        endex = start  # clamp negative length
    elif endex > size:
        endex = size  # trim end

    that = Rack_Delete_(that, start, (endex - start))
    return that


cdef Rack_* Rack_DelSlice(Rack_* that, ssize_t start, ssize_t endex) except NULL:
    cdef:
        ssize_t ssize

    ssize = <ssize_t>(that.endex - that.start)

    if start < 0:
        start += ssize  # anchor to end
    if start < 0:
        start = 0  # trim start

    if endex < 0:
        endex += ssize  # anchor to end
    if endex < start:
        endex = start  # clamp negative length

    that = Rack_DelSlice_(that, <size_t>start, <size_t>endex)
    return that


cdef addr_t Rack_Start(const Rack_* that) nogil:
    return Block_Start(Rack_First__(that)) if Rack_Bool(that) else 0


cdef addr_t Rack_Endex(const Rack_* that) nogil:
    return Block_Endex(Rack_Last__(that)) if Rack_Bool(that) else 0


cdef ssize_t Rack_IndexAt(const Rack_* that, addr_t address) nogil:
    cdef:
        ssize_t left = 0
        ssize_t right = <ssize_t>(that.endex - that.start)
        ssize_t center
        const Block_* block

    if right:
        block = that.blocks[that.start]
        if address < Block_Start(block):
            return -1

        block = that.blocks[that.endex - 1]
        if Block_Endex(block) <= address:
            return -1
    else:
        return -1

    while left <= right:
        center = (left + right) >> 1
        block = that.blocks[that.start + center]

        if Block_Endex(block) <= address:
            left = center + 1
        elif address < Block_Start(block):
            right = center - 1
        else:
            return center
    else:
        return -1


cdef ssize_t Rack_IndexStart(const Rack_* that, addr_t address) nogil:
    cdef:
        ssize_t left = 0
        ssize_t right = <ssize_t>(that.endex - that.start)
        ssize_t center
        const Block_* block

    if right:
        block = that.blocks[that.start]
        if address <= Block_Start(block):
            return 0

        block = that.blocks[that.endex - 1]
        if Block_Endex(block) <= address:
            return right
    else:
        return 0

    while left <= right:
        center = (left + right) >> 1
        block = that.blocks[that.start + center]

        if Block_Endex(block) <= address:
            left = center + 1
        elif address < Block_Start(block):
            right = center - 1
        else:
            return center
    else:
        return left


cdef ssize_t Rack_IndexEndex(const Rack_* that, addr_t address) nogil:
    cdef:
        ssize_t left = 0
        ssize_t right = <ssize_t>(that.endex - that.start)
        ssize_t center
        const Block_* block

    if right:
        block = that.blocks[that.start]
        if address < Block_Start(block):
            return 0

        block = that.blocks[that.endex - 1]
        if Block_Endex(block) <= address:
            return right
    else:
        return 0

    while left <= right:
        center = (left + right) >> 1
        block = that.blocks[that.start + center]

        if Block_Endex(block) <= address:
            left = center + 1
        elif address < Block_Start(block):
            right = center - 1
        else:
            return center + 1
    else:
        return right + 1


# =====================================================================================================================

cdef Memory Memory_AsObject(Memory_* that):
    cdef:
        Memory memory = Memory()

    memory._ = Memory_Free(memory._)
    memory._ = that
    return memory


cdef Memory_* Memory_Alloc() except NULL:
    cdef:
        Rack_* blocks = Rack_Alloc(0)
        Memory_* that = NULL

    that = <Memory_*>PyMem_Calloc(Memory_HEADING, 1)
    if that == NULL:
        blocks = Rack_Free(blocks)
        raise MemoryError()

    that.blocks = blocks
    that.trim_start = 0
    that.trim_endex = ADDR_MAX
    that.trim_start_ = False
    that.trim_endex_ = False
    return that


cdef Memory_* Memory_Free(Memory_* that) except? NULL:
    if that:
        that.blocks = Rack_Free(that.blocks)
        PyMem_Free(that)
    return NULL


cdef size_t Memory_Sizeof(const Memory_* that):
    if that:
        return Memory_HEADING + Rack_Sizeof(that.blocks)
    return 0


cdef Memory_* Memory_Create(
    object start,
    object endex,
) except NULL:
    cdef:
        Memory_* that = <Memory_*>PyMem_Calloc(Memory_HEADING, 1)

    if that == NULL:
        raise MemoryError()
    try:
        if start is None:
            that.trim_start = ADDR_MIN
            that.trim_start_ = False
        else:
            that.trim_start = <addr_t>start
            that.trim_start_ = True

        if endex is None:
            that.trim_endex = ADDR_MAX
            that.trim_endex_ = False
        else:
            that.trim_endex = <addr_t>endex
            that.trim_endex_ = True

        if that.trim_endex < that.trim_start:
            that.trim_endex = that.trim_start

        that.blocks = Rack_Alloc(0)

    except:
        that = Memory_Free(that)
        raise

    return that


cdef Memory_* Memory_FromBlocks_(
    const Rack_* blocks,
    object offset,
    object start,
    object endex,
    bint validate,
) except NULL:
    cdef:
        Memory_* that = NULL
        Rack_* blocks_
        size_t block_index
        Block_* block
        addr_t offset_abs = 0
        bint offset_neg = offset < 0

    if blocks == NULL:
        raise ValueError('null blocks')

    if Rack_Bool(blocks):
        if offset_neg:
            offset_abs = <addr_t>-offset
            CheckSubAddrU(Block_Start(Rack_First__(that.blocks)), offset_abs)
        else:
            offset_abs = <addr_t>offset
            CheckAddAddrU(Block_Endex(Rack_Last__(that.blocks)), offset_abs)

    try:
        that = Memory_Create(start, endex)
        that.blocks = Rack_Free(that.blocks)
        that.blocks = Rack_Copy(blocks)
        blocks_ = that.blocks

        if offset_neg:
            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get_(blocks, block_index)
                block.address -= offset_abs
        else:
            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get_(blocks, block_index)
                block.address += offset_abs

        if validate:
            Memory_Validate(that)

    except:
        that = Memory_Free(that)
        raise

    return that


cdef Memory_* Memory_FromBlocks(
    object blocks,
    object offset,
    object start,
    object endex,
    bint validate,
) except NULL:
    cdef:
        bint offset_ = <bint>offset
        Memory_* that = Memory_Create(start, endex)
        addr_t delta
        const byte_t[:] block_view
        const byte_t* block_data
        size_t block_size
        addr_t block_start
        addr_t block_endex
        size_t block_offset
        Block_* block = NULL

    try:
        for block_start_, block_data_ in blocks:
            block_offset = 0
            block_start = <addr_t>(block_start_ + offset) if offset_ else <addr_t>block_start_
            if that.trim_endex <= block_start:
                break

            block_view = block_data_
            block_size = <size_t>len(block_view)
            CheckAddAddrU(block_start, block_size)
            block_endex = block_start + block_size
            if block_endex <= that.trim_start:
                continue

            # Trim before memory
            if block_start < that.trim_start:
                delta = that.trim_start - block_start
                CheckAddrToSizeU(delta)
                block_start += <size_t>delta
                block_size  -= <size_t>delta
                block_offset = <size_t>delta

            # Trim after memory
            if that.trim_endex < block_endex:
                delta = block_endex - that.trim_endex
                CheckAddrToSizeU(delta)
                block_endex -= <size_t>delta
                block_size  -= <size_t>delta

            if block_size:
                with cython.boundscheck(False):
                    block_data = &block_view[block_offset]
                block = Block_Create(block_start, block_size, block_data)
                that.blocks = Rack_Append(that.blocks, block)
                block = NULL
            else:
                if not that.trim_start_ and not that.trim_endex_:
                    raise ValueError('invalid block data size')

        if validate:
            Memory_Validate(that)

    except:
        block = Block_Release(block)
        that = Memory_Free(that)
        raise

    return that


cdef Memory_* Memory_FromBytes_(
    size_t data_size,
    const byte_t* data_ptr,
    object offset,
    object start,
    object endex,
) except NULL:
    cdef:
        bint offset_ = <bint>offset
        Memory_* that = Memory_Create(start, endex)
        addr_t delta
        const byte_t* block_data
        size_t block_size = data_size
        addr_t block_start = <addr_t>offset
        addr_t block_endex
        size_t block_offset = 0
        Block_* block = NULL

    try:
        if that.trim_endex <= block_start:
            return that

        CheckAddAddrU(block_start, block_size)
        block_endex = block_start + block_size
        if block_endex <= that.trim_start:
            return that

        # Trim before memory
        if block_start < that.trim_start:
            delta = that.trim_start - block_start
            CheckAddrToSizeU(delta)
            block_start += <size_t>delta
            block_size  -= <size_t>delta
            block_offset = <size_t>delta

        # Trim after memory
        if that.trim_endex < block_endex:
            delta = block_endex - that.trim_endex
            CheckAddrToSizeU(delta)
            block_endex -= <size_t>delta
            block_size  -= <size_t>delta

        if block_size:
            with cython.boundscheck(False):
                block_data = &data_ptr[block_offset]
            block = Block_Create(block_start, block_size, block_data)
            that.blocks = Rack_Append(that.blocks, block)
            block = NULL

    except:
        block = Block_Release(block)
        that = Memory_Free(that)
        raise

    return that


cdef Memory_* Memory_FromBytes(
    object data,
    object offset,
    object start,
    object endex,
) except NULL:
    cdef:
        const byte_t[:] data_view = data
        const byte_t* data_ptr
        size_t data_size

    with cython.boundscheck(False):
        data_ptr = &data_view[0]
        data_size = len(data_view)

    return Memory_FromBytes_(data_size, data_ptr, offset, start, endex)


cdef Memory_* Memory_FromItems(
    object items,
    object offset,
    object start,
    object endex,
    bint validate,
) except NULL:
    cdef:
        Memory_* that = Memory_Create(start, endex)

    try:
        if isinstance(items, Mapping):
            items = items.items()

        for address, value in items:
            Memory_Poke(that, address + offset, value)

        if validate:
            Memory_Validate(that)

    except:
        that = Memory_Free(that)
        raise

    return that


cdef Memory_* Memory_FromMemory_(
    const Memory_* memory,
    object offset,
    object start,
    object endex,
    bint validate,
) except NULL:
    cdef:
        Memory_* that = Memory_Copy(memory)

    try:
        that.trim_start = ADDR_MIN
        that.trim_endex = ADDR_MAX
        that.trim_start_ = False
        that.trim_endex_ = False

        Memory_Shift(that, offset)

        Memory_SetTrimStart(that, start)
        Memory_SetTrimEndex(that, endex)

        if validate:
            Memory_Validate(that)

    except:
        that = Memory_Free(that)
        raise

    return that


cdef Memory_* Memory_FromMemory(
    object memory,
    object offset,
    object start,
    object endex,
    bint validate,
) except NULL:

    if isinstance(memory, Memory):
        return Memory_FromMemory_((<Memory>memory)._, offset, start, endex, validate)
    else:
        return Memory_FromBlocks(memory.blocks(), offset, start, endex, validate)


cdef Memory_* Memory_FromValues(
    object values,
    object offset,
    object start,
    object endex,
    bint validate,
) except NULL:
    cdef:
        addr_t start_ = ADDR_MIN if start is None else <addr_t>start
        addr_t endex_ = ADDR_MAX if endex is None else <addr_t>endex
        Memory_* that = Memory_Create(start, endex)
        addr_t address = <addr_t>offset
        bint append = False
        byte_t value_

    if endex_ < start_:
        endex_ = start_

    try:
        for value in values:
            if endex_ <= address:
                break
            if start_ <= address:
                if value is not None:
                    value_ = <byte_t>value
                    if append:
                        Memory_Append_(that, value_)
                    else:
                        Memory_Poke_(that, address, value_)
                    append = True
                else:
                    append = False
            address += 1

        if validate:
            Memory_Validate(that)

    except:
        that = Memory_Free(that)
        raise

    return that


cdef bint Memory_EqSame_(const Memory_* that, const Memory_* other) except -1:
    return Rack_Eq(that.blocks, other.blocks)


cdef bint Memory_EqRaw_(const Memory_* that, size_t data_size, const byte_t* data_ptr) except -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_count
        const Block_* block
        size_t size

    block_count = Rack_Length(blocks)
    if block_count:
        if block_count != 1:
            return False

        block = Rack_First__(blocks)
        size = Block_Length(block)
        if data_size != size:
            return False

        if memcmp(Block_At__(block, 0), data_ptr, data_size):
            return False
        return True
    else:
        return not data_size


cdef bint Memory_EqView_(const Memory_* that, const byte_t[:] view) except -1:
    with cython.boundscheck(False):
        return Memory_EqRaw_(that, len(view), &view[0])


cdef bint Memory_EqIter_(const Memory_* that, object iterable) except -1:
    cdef:
        addr_t start = Memory_Start(that)
        addr_t endex = Memory_ContentEndex(that)
        Rover_* rover = Rover_Create(that, start, endex, 0, NULL, True, False)
        bint equal = True
        int item1_
        int item2_

    try:
        for item2 in iterable:
            item1_ = Rover_Next_(rover) if Rover_HasNext(rover) else -1
            item2_ = -1 if item2 is None else <int><unsigned><byte_t>item2

            if item1_ != item2_:
                equal = False
                break
        else:
            if Rover_HasNext(rover):
                equal = False
    finally:
        Rover_Free(rover)

    return equal


cdef bint Memory_EqMemory_(const Memory_* that, object memory) except -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index = 0
        const Block_* block
        size_t block_size
        const byte_t[:] view

    if Memory_ContentParts(that) != memory.content_parts:
        return False

    for block_start, block_data in memory.blocks():
        block = Rack_Get__(blocks, block_index)

        if Block_Start(block) != block_start:
            return False

        with cython.boundscheck(False):
            view = block_data
            if not Block_Eq_(block, len(view), &view[0]):
                return False

        block_index += 1

    return True


cdef bint Memory_Eq(const Memory_* that, object other) except -1:
    cdef:
        const byte_t[:] view

    if isinstance(other, Memory):
        return Memory_EqSame_(that, (<Memory>other)._)

    elif isinstance(other, ImmutableMemory):
        return Memory_EqMemory_(that, other)

    else:
        try:
            view = other
        except TypeError:
            return Memory_EqIter_(that, other)
        else:
            return Memory_EqView_(that, view)


cdef Memory_* Memory_Add(const Memory_* that, object value) except NULL:
    cdef:
        Memory_* memory = Memory_Copy(that)
    try:
        Memory_Extend(memory, value, 0)
    except:
        memory = Memory_Free(memory)
        raise
    return memory


cdef Memory_* Memory_IAdd(Memory_* that, object value) except NULL:
    Memory_Extend(that, value, 0)
    return that


cdef Memory_* Memory_Mul(const Memory_* that, addr_t times) except NULL:
    cdef:
        Memory_* memory = NULL
        addr_t offset
        addr_t size
        addr_t time

    if times and Rack_Bool(that.blocks):
        start = Memory_Start(that)
        size = Memory_Endex(that) - start
        offset = size  # adjust first write
        memory = Memory_Copy(that)
        try:
            for time in range(times - 1):
                Memory_WriteSame_(memory, offset, that, False)
                offset += size
        except:
            memory = Memory_Free(memory)
            raise

        return memory
    else:
        return Memory_Alloc()


cdef Memory_* Memory_IMul(Memory_* that, addr_t times) except NULL:
    cdef:
        Memory_* memory = NULL
        addr_t offset

    times = int(times)
    if times < 0:
        times = 0

    if times and Rack_Bool(that.blocks):
        start = Memory_Start(that)
        size = Memory_Endex(that) - start
        offset = size
        memory = Memory_Copy(that)
        try:
            for time in range(times - 1):
                Memory_WriteSame_(that, offset, memory, False)
                offset += size
        finally:
            memory = Memory_Free(memory)
    else:
        that.blocks = Rack_Clear(that.blocks)
    return that


cdef bint Memory_Bool(const Memory_* that) nogil:
    return Rack_Bool(that.blocks)


cdef addr_t Memory_Length(const Memory_* that) nogil:
    return Memory_Endex(that) - Memory_Start(that)


cdef bint Memory_IsEmpty(const Memory_* that) nogil:
    return not Rack_Bool(that.blocks)


cdef object Memory_ObjFind(const Memory_* that, object item, object start, object endex):
    offset = Memory_Find(that, item, start, endex)
    if offset >= 0:
        return offset
    else:
        return None


cdef object Memory_RevObjFind(const Memory_* that, object item, object start, object endex):
    offset = Memory_RevFind(that, item, start, endex)
    if offset >= 0:
        return offset
    else:
        return None


cdef addr_t Memory_FindUnbounded_(const Memory_* that, size_t size, const byte_t* buffer) except? -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        ssize_t offset
        addr_t start_
        addr_t offset_

    if size:
        for block_index in range(Rack_Length(blocks)):
            block = Rack_Get__(blocks, block_index)
            offset = Block_Find_(block, 0, SIZE_MAX, size, buffer)
            if offset >= 0:
                start_ = Block_Start(block)
                offset_ = <addr_t><size_t>offset
                CheckAddAddrU(start_, offset_)
                return start_ + offset_
    return ADDR_MAX


cdef addr_t Memory_FindBounded_(const Memory_* that, size_t size, const byte_t* buffer,
                                addr_t start, addr_t endex) except? -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        ssize_t offset
        size_t block_index_start
        size_t block_index_endex
        size_t slice_start
        size_t slice_endex
        addr_t start_
        addr_t offset_

    if size:
        if endex < start:
            endex = start
        block_index_start = Rack_IndexStart(blocks, start)
        block_index_endex = Rack_IndexEndex(blocks, endex)

        for block_index in range(block_index_start, block_index_endex):
            block = Rack_Get__(blocks, block_index)
            slice_start, slice_endex = Block_BoundAddressSliceToOffset(block, start, endex)
            offset = Block_Find_(block, slice_start, slice_endex, size, buffer)
            if offset >= 0:
                start_ = Block_Start(block)
                offset_ = <addr_t><size_t>offset
                CheckAddAddrU(start_, offset_)
                return start_ + offset_
    return ADDR_MAX


cdef object Memory_Find(const Memory_* that, object item, object start, object endex):
    cdef:
        addr_t start_
        addr_t endex_
        byte_t item_value
        const byte_t[:] item_view
        size_t item_size
        const byte_t* item_ptr
        addr_t address

    if isinstance(item, int):
        item_value = <byte_t>item
        item_size = 1
        item_ptr = &item_value
    else:
        item_view = item
        item_size = 1
        with cython.boundscheck(False):
            item_ptr = &item_view[0]

    # Faster code for unbounded slice
    if start is None and endex is None:
        address = Memory_FindUnbounded_(that, item_size, item_ptr)
        return -1 if address == ADDR_MAX else <object>address

    # Bounded slice
    start_, endex_ = Memory_Bound(that, start, endex)
    address = Memory_FindBounded_(that, item_size, item_ptr, start_, endex_)
    return -1 if address == ADDR_MAX else <object>address


cdef addr_t Memory_RevFindUnbounded_(const Memory_* that, size_t size, const byte_t* buffer) except? -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        ssize_t offset
        addr_t start_
        addr_t offset_

    if size:
        for block_index in range(Rack_Length(blocks), 0, -1):
            block = Rack_Get__(blocks, block_index - 1)
            offset = Block_ReverseFind_(block, 0, SIZE_MAX, size, buffer)
            if offset >= 0:
                start_ = Block_Start(block)
                offset_ = <addr_t><size_t>offset
                CheckAddAddrU(start_, offset_)
                return start_ + offset_
    return ADDR_MAX


cdef addr_t Memory_RevFindBounded_(const Memory_* that, size_t size, const byte_t* buffer,
                                   addr_t start, addr_t endex) except? -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        ssize_t offset
        size_t block_index_start
        size_t block_index_endex
        size_t slice_start
        size_t slice_endex
        addr_t start_
        addr_t offset_

    if size:
        if endex < start:
            endex = start
        block_index_start = Rack_IndexStart(blocks, start)
        block_index_endex = Rack_IndexEndex(blocks, endex)

        for block_index in range(block_index_endex, block_index_start, -1):
            block = Rack_Get__(blocks, block_index - 1)
            slice_start, slice_endex = Block_BoundAddressSliceToOffset(block, start, endex)
            offset = Block_ReverseFind_(block, slice_start, slice_endex, size, buffer)
            if offset >= 0:
                start_ = Block_Start(block)
                offset_ = <addr_t><size_t>offset
                CheckAddAddrU(start_, offset_)
                return start_ + offset_
    return ADDR_MAX


cdef object Memory_RevFind(const Memory_* that, object item, object start, object endex):
    cdef:
        addr_t start_
        addr_t endex_
        byte_t item_value
        const byte_t[:] item_view
        size_t item_size
        const byte_t* item_ptr
        addr_t address

    if isinstance(item, int):
        item_value = <byte_t>item
        item_size = 1
        item_ptr = &item_value
    else:
        item_view = item
        item_size = 1
        with cython.boundscheck(False):
            item_ptr = &item_view[0]

    # Faster code for unbounded slice
    if start is None and endex is None:
        address = Memory_RevFindUnbounded_(that, item_size, item_ptr)
        return -1 if address == ADDR_MAX else <object>address

    # Bounded slice
    start_, endex_ = Memory_Bound(that, start, endex)
    address = Memory_RevFindBounded_(that, item_size, item_ptr, start_, endex_)
    return -1 if address == ADDR_MAX else <object>address


cdef object Memory_Index(const Memory_* that, object item, object start, object endex):
    offset = Memory_Find(that, item, start, endex)
    if offset != -1:
        return offset
    else:
        raise ValueError('subsection not found')


cdef object Memory_RevIndex(const Memory_* that, object item, object start, object endex):
    offset = Memory_RevFind(that, item, start, endex)
    if offset != -1:
        return offset
    else:
        raise ValueError('subsection not found')


cdef bint Memory_Contains(const Memory_* that, object item) except -1:
    cdef:
        byte_t item_value
        const byte_t[:] item_view
        size_t item_size
        const byte_t* item_ptr
        addr_t address

    if isinstance(item, int):
        item_value = <byte_t>item
        item_size = 1
        item_ptr = &item_value
    else:
        item_view = item
        item_size = 1
        with cython.boundscheck(False):
            item_ptr = &item_view[0]

    address = Memory_FindUnbounded_(that, item_size, item_ptr)
    return address != ADDR_MAX


cdef addr_t Memory_CountUnbounded_(const Memory_* that, size_t size, const byte_t* buffer) except? -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        addr_t count = 0

    if size:
        for block_index in range(Rack_Length(blocks)):
            block = Rack_Get__(blocks, block_index)
            count += Block_Count_(block, 0, SIZE_MAX, size, buffer)
    return count


cdef addr_t Memory_CountBounded_(const Memory_* that, size_t size, const byte_t* buffer,
                                 addr_t start, addr_t endex) except? -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        addr_t count = 0
        size_t block_index_start
        size_t block_index_endex
        addr_t block_start
        addr_t block_endex
        size_t slice_start
        size_t slice_endex

    if size:
        if endex < start:
            endex = start
        block_index_start = Rack_IndexStart(blocks, start)
        block_index_endex = Rack_IndexEndex(blocks, endex)

        for block_index in range(block_index_start, block_index_endex):
            block = Rack_Get__(blocks, block_index)
            slice_start, slice_endex = Block_BoundAddressSliceToOffset(block, start, endex)
            count += Block_Count_(block, slice_start, slice_endex, size, buffer)
    return count


cdef addr_t Memory_Count(const Memory_* that, object item, object start, object endex) except? -1:
    cdef:
        addr_t start_
        addr_t endex_
        byte_t item_value
        const byte_t[:] item_view
        size_t item_size
        const byte_t* item_ptr

    if isinstance(item, int):
        item_value = <byte_t>item
        item_size = 1
        item_ptr = &item_value
    else:
        item_view = item
        item_size = 1
        with cython.boundscheck(False):
            item_ptr = &item_view[0]

    # Faster code for unbounded slice
    if start is None and endex is None:
        return Memory_CountUnbounded_(that, item_size, item_ptr)

    # Bounded slice
    start_, endex_ = Memory_Bound(that, start, endex)
    return Memory_CountBounded_(that, item_size, item_ptr, start_, endex_)


cdef object Memory_GetItem(const Memory_* that, object key):
    cdef:
        slice key_
        addr_t start
        addr_t endex
        Block_* pattern = NULL
        Memory memory
        int value

    if isinstance(key, slice):
        key_ = <slice>key
        key_start = key_.start
        key_endex = key_.stop
        start = Memory_Start(that) if key_start is None else <addr_t>key_start
        endex = Memory_Endex(that) if key_endex is None else <addr_t>key_endex
        key_step = key_.step

        if key_step is None or key_step is 1 or key_step == 1:
            return Memory_Extract_(that, start, endex, 0, NULL, 1, True)

        elif isinstance(key_step, int):
            if key_step > 1:
                return Memory_Extract_(that, start, endex, 0, NULL, <saddr_t>key_step, True)
            else:
                return Memory()  # empty

        else:
            pattern = Block_FromObject(0, key_step, True)
            try:
                memory = Memory_Extract_(that, start, endex, Block_Length(pattern), Block_At__(pattern, 0), 1, True)
            finally:
                Block_Free(pattern)  # orphan
            return memory
    else:
        value = Memory_Peek_(that, <addr_t>key)
        return None if value < 0 else value


cdef object Memory_SetItem(Memory_* that, object key, object value):
    cdef:
        slice key_
        addr_t start
        addr_t endex
        addr_t step = 0  # indefinite
        addr_t address
        addr_t slice_size
        Block_* value_ = NULL
        size_t value_size
        addr_t del_start
        addr_t del_endex
        size_t offset

    if isinstance(key, slice):
        key_ = <slice>key
        key_start = key_.start
        key_endex = key_.stop
        start = Memory_Start(that) if key_start is None else <addr_t>key_start
        endex = Memory_Endex(that) if key_endex is None else <addr_t>key_endex
        if endex < start:
            endex = start

        key_step = key_.step
        if isinstance(key_step, int):
            if key_step is None or key_step is 1 or key_step == 1:
                pass
            elif key_step > 1:
                step = <addr_t>key_step
            else:
                return  # empty range

        if value is None:
            # Clear range
            if not step:
                Memory_Erase__(that, start, endex, False)  # clear
            else:
                address = start
                while address < endex:
                    Memory_Erase__(that, address, address + 1, False)  # clear
                    if CannotAddAddrU(address, step):
                        break
                    address += step
            return  # nothing to write

        slice_size = endex - start
        if step:
            with cython.cdivision(True):
                slice_size = (slice_size + step - 1) // step
        CheckAddrToSizeU(slice_size)

        value_ = Block_FromObject(0, value, False)
        try:
            if isinstance(value, int):
                value_ = Block_Repeat(value_, <size_t>slice_size)
            value_size = Block_Length(value_)

            if value_size < slice_size:
                # Shrink: remove excess, overwrite existing
                if not step:
                    if CannotAddAddrU(start, value_size):
                        del_start = ADDR_MAX
                    else:
                        del_start = start + value_size
                    if CannotAddAddrU(del_start, (slice_size - value_size)):
                        del_endex = ADDR_MAX
                    else:
                        del_endex = del_start + (slice_size - value_size)
                    Memory_Erase__(that, del_start, del_endex, True)  # delete
                    if value_size:
                        Memory_WriteRaw_(that, start, value_size, Block_At__(value_, 0))
                else:
                    raise ValueError(f'attempt to assign bytes of size {value_size}'
                                     f' to extended slice of size {slice_size}')
            elif slice_size < value_size:
                # Enlarge: insert excess, overwrite existing
                if not step:
                    Memory_InsertRaw_(that, endex, value_size - slice_size, Block_At__(value_, slice_size))
                    Memory_WriteRaw_(that, start, slice_size, Block_At__(value_, 0))
                else:
                    raise ValueError(f'attempt to assign bytes of size {value_size}'
                                     f' to extended slice of size {slice_size}')
            else:
                # Same size: overwrite existing
                if not step:
                    Memory_WriteRaw_(that, start, value_size, Block_At__(value_, 0))
                else:
                    CheckMulAddrU(step, value_size)
                    CheckAddAddrU(start, step * value_size)
                    for offset in range(value_size):
                        Memory_Poke_(that, start + (step * offset), Block_Get__(value_, offset))
        finally:
            Block_Free(value_)  # orphan
    else:
        # below: self.poke(key, value)
        address = <addr_t>key
        if value is None:
            Memory_PokeNone_(that, address)
        else:
            if isinstance(value, int):
                Memory_Poke_(that, address, <byte_t>value)
            else:
                if len(value) != 1:
                    raise ValueError('expecting single item')
                Memory_Poke_(that, address, <byte_t>value[0])


cdef vint Memory_DelItem(Memory_* that, object key) except -1:
    cdef:
        slice key_
        addr_t start
        addr_t endex
        addr_t step
        addr_t address

    if Rack_Bool(that.blocks):
        if isinstance(key, slice):
            key_ = <slice>key
            key_start = key_.start
            key_endex = key_.stop
            start = Memory_Start(that) if key_start is None else <addr_t>key_start
            endex = Memory_Endex(that) if key_endex is None else <addr_t>key_endex

            if start < endex:
                key_step = key_.step
                if key_step is None or key_step is 1 or key_step == 1:
                    Memory_Erase__(that, start, endex, True)  # delete

                elif key_step > 1:
                    step = <addr_t>key_step - 1
                    address = start
                    while address < endex:
                        Memory_Erase__(that, address, address + 1, True)  # delete
                        address += step
                        endex -= 1
        else:
            address = <addr_t>key
            Memory_Erase__(that, address, address + 1, True)  # delete


cdef vint Memory_Append_(Memory_* that, byte_t value) except -1:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_count
        Block_* block

    block_count = Rack_Length(blocks)
    if block_count:
        block = Block_Append(Rack_Last_(blocks), value)
        Rack_Set__(blocks, block_count - 1, block)  # update pointer
    else:
        block = Block_Create(0, 1, &value)
        try:
            that.blocks = blocks = Rack_Append(blocks, block)
        except:
            Block_Free(block)  # orphan
            raise


cdef vint Memory_Append(Memory_* that, object item) except -1:
    if isinstance(item, int):
        Memory_Append_(that, <byte_t>item)
    else:
        if len(item) != 1:
            raise ValueError('expecting single item')
        Memory_Append_(that, <byte_t>item[0])


cdef vint Memory_ExtendSame_(Memory_* that, const Memory_* items, addr_t offset) except -1:
    cdef:
        addr_t content_endex = Memory_ContentEndex(that)

    CheckAddAddrU(content_endex, offset)
    offset += content_endex
    Memory_WriteSame_(that, offset, items, False)


cdef vint Memory_ExtendRaw_(Memory_* that, size_t items_size, const byte_t* items_ptr, addr_t offset) except -1:
    cdef:
        addr_t content_endex = Memory_ContentEndex(that)

    CheckAddAddrU(content_endex, offset)
    offset += content_endex
    CheckAddAddrU(offset, items_size)
    Memory_WriteRaw_(that, offset, items_size, items_ptr)


cdef vint Memory_Extend(Memory_* that, object items, object offset) except -1:
    cdef:
        const byte_t[:] items_view
        byte_t items_value
        size_t items_size
        const byte_t* items_ptr

    if offset < 0:
        raise ValueError('negative extension offset')

    if isinstance(items, Memory):
        Memory_ExtendSame_(that, (<Memory>items)._, <addr_t>offset)
    else:
        if isinstance(items, int):
            items_value = <byte_t>items
            items_size = 1
            items_ptr = &items_value
        else:
            items_view = items
            items_size = len(items_view)
            with cython.boundscheck(False):
                items_ptr = &items_view[0]

        Memory_ExtendRaw_(that, items_size, items_ptr, <addr_t>offset)


cdef int Memory_PopLast_(Memory_* that) except -2:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_count = Rack_Length(blocks)
        Block_* block
        size_t length
        byte_t backup

    if block_count:
        block = Rack_Last_(blocks)
        length = Block_Length(block)
        if length > 1:
            block = Block_Pop__(block, &backup)
            Rack_Set__(blocks, block_count - 1, block)  # update pointer
        elif length == 1:
            backup = Block_Get__(block, 0)
            that.blocks = blocks = Rack_Pop__(blocks, NULL)  # update pointer
        else:
            raise RuntimeError('empty block')
        return backup
    else:
        return -1


cdef int Memory_PopAt_(Memory_* that, addr_t address) except -2:
    cdef:
        int backup

    backup = Memory_Peek_(that, address)
    Memory_Erase__(that, address, address + 1, True)  # delete
    return backup


cdef object Memory_Pop(Memory_* that, object address, object default):
    cdef:
        int value

    if address is None:
        value = Memory_PopLast_(that)
    else:
        value = Memory_PopAt_(that, <addr_t>address)
    return default if value < 0 else value


cdef (addr_t, int) Memory_PopItem(Memory_* that) except *:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_count = Rack_Length(blocks)
        Block_* block
        size_t length
        addr_t address
        byte_t backup

    if block_count:
        block = Rack_Last_(blocks)
        length = Block_Length(block)
        address = Block_Start(block) + length - 1
        if length > 1:
            block = Block_Pop__(block, &backup)
            Rack_Set__(blocks, block_count - 1, block)  # update pointer
        elif length == 1:
            backup = Block_Get__(block, 0)
            that.blocks = blocks = Rack_Pop__(blocks, NULL)  # update pointer
        else:
            raise RuntimeError('empty block')
        return address, backup
    else:
        raise KeyError('empty')


cdef BlockView Memory_View_(const Memory_* that, addr_t start, addr_t endex):
    cdef:
        const Rack_* blocks
        Block_* block
        ssize_t block_index
        addr_t block_start
        addr_t block_endex

    if start < endex:
        blocks = that.blocks
        block_index = Rack_IndexAt(blocks, start)

        if block_index >= 0:
            block = Rack_Get__(blocks, <size_t>block_index)
            block_endex = Block_Endex(block)

            if endex <= block_endex:
                block_start = Block_Start(block)
                start -= block_start
                endex -= block_start
                return Block_ViewSlice_(block, <size_t>start, <size_t>endex)

        raise ValueError('non-contiguous data within range')
    else:
        return Block_ViewSlice_(_empty_block, 0, 0)


cdef BlockView Memory_View(const Memory_* that, object start, object endex):
    cdef:
        addr_t start_
        addr_t endex_

    start_, endex_ = Memory_Bound(that, start, endex)
    return Memory_View_(that, start_, endex_)


cdef Memory_* Memory_Copy(const Memory_* that) except NULL:
    cdef:
        Rack_* blocks = Rack_Copy(that.blocks)
        Memory_* memory = NULL

    memory = <Memory_*>PyMem_Calloc(Memory_HEADING, 1)
    if memory == NULL:
        blocks = Rack_Free(blocks)
        raise MemoryError()

    memory.blocks = blocks
    memory.trim_start = that.trim_start
    memory.trim_endex = that.trim_endex
    memory.trim_start_ = that.trim_start_
    memory.trim_endex_ = that.trim_endex_
    return memory


cdef Memory_* Memory_Cut_(const Memory_* that, addr_t start, addr_t endex, bint bound) except NULL:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_count
        size_t block_index
        size_t block_index_start
        size_t block_index_endex
        Memory_* memory = Memory_Alloc()
        Rack_* memory_blocks
        Block_* block1
        Block_* block2
        addr_t block_start
        addr_t block_endex

    if endex < start:
        endex = start

    if start < endex and Rack_Bool(blocks):
        # Copy all the blocks except those completely outside selection
        block_index_start = Rack_IndexStart(blocks, start)
        block_index_endex = Rack_IndexEndex(blocks, endex)
        block_count = block_index_endex - block_index_start

        if block_count:
            memory_blocks = memory.blocks
            block_count = block_index_endex - block_index_start
            try:
                memory.blocks = memory_blocks = Rack_Reserve_(memory_blocks, 0, block_count)
            except:
                memory = Memory_Free(memory)  # orphan
                raise
            else:
                for block_index in range(block_count):
                    block1 = Rack_Get_(blocks, block_index_start + block_index)
                    block1 = Block_Acquire(block1)
                    Rack_Set__(memory_blocks, block_index, block1)
            try:
                # Trim cloned data before the selection start address
                block_index = 0
                block1 = Rack_Get_(memory_blocks, block_index)
                block_start = Block_Start(block1)
                if block_start < start:
                    try:
                        block2 = NULL
                        block2 = Block_Copy(block1)
                        block2 = Block_DelSlice_(block2, 0, <size_t>(start - block_start))
                        block2.address = start
                    except:
                        block2 = Block_Free(block2)  # orphan
                        raise
                    else:
                        Rack_Set__(memory_blocks, block_index, block2)
                        Block_Release(block1)

                # Trim cloned data after the selection end address
                block_index = block_count - 1
                block1 = Rack_Get_(memory_blocks, block_index)
                block_endex = Block_Endex(block1)
                if endex < block_endex:
                    block_start = Block_Start(block1)
                    if block_start < endex:
                        try:
                            block2 = NULL
                            block2 = Block_Copy(block1)
                            block2 = Block_DelSlice_(block2, <size_t>(endex - block_start), Block_Length(block2))
                            block2.address = block_start
                        except:
                            block2 = Block_Free(block2)  # orphan
                            raise
                        else:
                            Rack_Set__(memory_blocks, block_index, block2)
                            Block_Release(block1)
                    else:
                        memory.blocks = memory_blocks = Rack_Pop__(memory_blocks, &block2)
                        block2 = Block_Release(block2)  # orphan

                Memory_Erase__(that, start, endex, False)  # clear

            except:
                memory = Memory_Free(memory)  # orphan
                raise

    if bound:
        memory.trim_start = start
        memory.trim_endex = endex
        memory.trim_start_ = True
        memory.trim_endex_ = True

    return memory


cdef object Memory_Cut(const Memory_* that, object start, object endex, bint bound):
    cdef:
        addr_t start_ = Memory_Start(that) if start is None else <addr_t>start
        addr_t endex_ = Memory_Endex(that) if endex is None else <addr_t>endex
        Memory_* memory = Memory_Cut_(that, start_, endex_, bound)

    return Memory_AsObject(memory)



cdef void Memory_Reverse(Memory_* that) nogil:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_count = Rack_Length(blocks)
        size_t block_index
        Block_* block
        addr_t start
        addr_t endex

    if block_count:
        start = Memory_Start(that)
        endex = Memory_Endex(that)

        for block_index in range(block_count):
            block = Rack_Get__(blocks, block_index)
            Block_Reverse(block)
            block.address = endex - Block_Endex(block) + start

        Rack_Reverse(that.blocks)


cdef bint Memory_Contiguous(const Memory_* that) nogil:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_count = Rack_Length(blocks)
        addr_t start
        addr_t endex

    if not block_count:
        start = that.trim_start
        endex = that.trim_endex
        if that.trim_start_ and that.trim_endex_ and start < endex:
            return False
        return True

    elif block_count == 1:
        start = that.trim_start
        if that.trim_start_:
            if start != Block_Start(Rack_First__(blocks)):
                return False
        endex = that.trim_endex
        if that.trim_endex_:
            if endex != Block_Endex(Rack_Last__(blocks)):
                return False
        return True

    return False


cdef object Memory_GetTrimStart(const Memory_* that):
    return that.trim_start if that.trim_start_ else None


cdef vint Memory_SetTrimStart(Memory_* that, object trim_start) except -1:
    cdef:
        addr_t trim_start_
        addr_t trim_endex_

    if trim_start is None:
        trim_start_ = ADDR_MIN
        that.trim_start_ = False
    else:
        trim_start_ = <addr_t>trim_start
        that.trim_start_ = True

    trim_endex_ = that.trim_endex
    if that.trim_start_ and that.trim_endex_ and trim_endex_ < trim_start_:
        that.trim_endex = trim_endex_ = trim_start_

    that.trim_start = trim_start_
    if that.trim_start_:
        Memory_Crop_(that, trim_start_, trim_endex_)


cdef object Memory_GetTrimEndex(const Memory_* that):
    return that.trim_endex if that.trim_endex_ else None


cdef vint Memory_SetTrimEndex(Memory_* that, object trim_endex) except -1:
    cdef:
        addr_t trim_start_
        addr_t trim_endex_

    if trim_endex is None:
        trim_endex_ = ADDR_MAX
        that.trim_endex_ = False
    else:
        trim_endex_ = <addr_t>trim_endex
        that.trim_endex_ = True

    trim_start_ = that.trim_start
    if that.trim_start_ and that.trim_endex_ and trim_endex_ < trim_start_:
        that.trim_start = trim_start_ = trim_endex_

    that.trim_endex = trim_endex_
    if that.trim_endex_:
        Memory_Crop_(that, trim_start_, trim_endex_)


cdef object Memory_GetTrimSpan(const Memory_* that):
    return (that.trim_start if that.trim_start_ else None,
            that.trim_endex if that.trim_endex_ else None)


cdef vint Memory_SetTrimSpan(Memory_* that, object trim_span) except -1:
    trim_start, trim_endex = trim_span

    if trim_start is None:
        trim_start_ = 0
        that.trim_start_ = False
    else:
        trim_start_ = <addr_t>trim_start
        that.trim_start_ = True

    if trim_endex is None:
        trim_endex_ = ADDR_MAX
        that.trim_endex_ = False
    else:
        trim_endex_ = <addr_t>trim_endex
        that.trim_endex_ = True

    if that.trim_start_ and that.trim_endex_ and trim_endex_ < trim_start_:
        trim_endex_ = trim_start_

    that.trim_start = trim_start_
    that.trim_endex = trim_endex_
    if that.trim_start_ or that.trim_endex_:
        Memory_Crop_(that, trim_start_, trim_endex_)


cdef addr_t Memory_Start(const Memory_* that) nogil:
    cdef:
        const Rack_* blocks

    if not that.trim_start_:
        # Return actual
        blocks = that.blocks
        if Rack_Bool(blocks):
            return Block_Start(Rack_First__(blocks))
        else:
            return 0
    else:
        return that.trim_start


cdef addr_t Memory_Endex(const Memory_* that) nogil:
    cdef:
        const Rack_* blocks

    if not that.trim_endex_:
        # Return actual
        blocks = that.blocks
        if Rack_Bool(blocks):
            return Block_Endex(Rack_Last__(blocks))
        else:
            return Memory_Start(that)
    else:
        return that.trim_endex


cdef (addr_t, addr_t) Memory_Span(const Memory_* that) nogil:
    return Memory_Start(that), Memory_Endex(that)


cdef object Memory_Endin(const Memory_* that):
    cdef:
        const Rack_* blocks

    if not that.trim_endex_:
        # Return actual
        blocks = that.blocks
        if Rack_Bool(blocks):
            return <object>Block_Endex(Rack_Last__(blocks)) - 1
        else:
            return <object>Memory_Start(that) - 1
    else:
        return <object>that.trim_endex - 1


cdef addr_t Memory_ContentStart(const Memory_* that) nogil:
    cdef:
        const Rack_* blocks = that.blocks

    if Rack_Bool(blocks):
        return Block_Start(Rack_First__(blocks))
    elif not that.trim_start_:
        return 0
    else:
        return that.trim_start


cdef addr_t Memory_ContentEndex(const Memory_* that) nogil:
    cdef:
        const Rack_* blocks = that.blocks

    if Rack_Bool(blocks):
        return Block_Endex(Rack_Last__(blocks))
    elif not that.trim_start_:
        return 0  # default to start
    else:
        return that.trim_start  # default to start


cdef (addr_t, addr_t) Memory_ContentSpan(const Memory_* that) nogil:
    return Memory_ContentStart(that), Memory_ContentEndex(that)


cdef object Memory_ContentEndin(const Memory_* that):
    cdef:
        const Rack_* blocks = that.blocks

    if Rack_Bool(blocks):
        return <object>Block_Endex(Rack_Last__(blocks)) - 1
    elif not that.trim_start_:  # default to start-1
        return -1
    else:
        return <object>that.trim_start - 1  # default to start-1


cdef addr_t Memory_ContentSize(const Memory_* that) nogil:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_index
        const Block_* block
        addr_t content_size = 0

    for block_index in range(Rack_Length(blocks)):
        block = Rack_Get__(blocks, block_index)
        content_size += Block_Length(block)
    return content_size


cdef size_t Memory_ContentParts(const Memory_* that) nogil:
    return Rack_Length(that.blocks)


cdef vint Memory_Validate(const Memory_* that) except -1:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_count = Rack_Length(blocks)

        addr_t start
        addr_t endex
        addr_t previous_endex = 0

        size_t block_index
        const Block_* block
        addr_t block_start
        addr_t block_endex

    start, endex = Memory_Bound(that, None, None)
    block_count = Rack_Length(blocks)

    if block_count:
        if endex <= start:
            raise ValueError('invalid bounds')

        for block_index in range(block_count):
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)

            if block_index:  # skip first
                if block_start <= previous_endex:
                    raise ValueError('invalid block interleaving')

            if block_endex <= block_start:
                raise ValueError('invalid block data size')

            if block_start < start or endex < block_endex:
                raise ValueError('invalid block bounds')

            previous_endex = block_endex

    else:
        if endex < start:
            raise ValueError('invalid bounds')


cdef (addr_t, addr_t) Memory_Bound_(const Memory_* that, addr_t start, addr_t endex,
                                    bint start_, bint endex_) nogil:
    cdef:
        addr_t trim_start
        addr_t trim_endex

    trim_start = that.trim_start
    trim_endex = that.trim_endex

    if not start_:
        if not that.trim_start_:
            if Rack_Bool(that.blocks):
                start = Block_Start(Rack_First__(that.blocks))
            else:
                start = 0
        else:
            start = trim_start
    else:
        if that.trim_start_:
            if start < trim_start:
                start = trim_start
        if endex_:
            if endex < start:
                endex = start

    if not endex_:
        if not that.trim_endex_:
            if Rack_Bool(that.blocks):
                endex = Block_Endex(Rack_Last__(that.blocks))
            else:
                endex = start
        else:
            endex = trim_endex
    else:
        if that.trim_endex_:
            if endex > trim_endex:
                endex = trim_endex
        if start > endex:
            start = endex

    return start, endex


cdef (addr_t, addr_t) Memory_Bound(const Memory_* that, object start, object endex) except *:
    cdef:
        bint start__ = start is not None
        bint endex__ = endex is not None
        addr_t start_ = <addr_t>start if start__ else 0
        addr_t endex_ = <addr_t>endex if endex__ else start_

    return Memory_Bound_(that, start_, endex_, start__, endex__)


cdef int Memory_Peek_(const Memory_* that, addr_t address) nogil:
    cdef:
        const Rack_* blocks = that.blocks
        ssize_t block_index
        const Block_* block

    block_index = Rack_IndexAt(blocks, address)
    if block_index < 0:
        return -1
    else:
        block = Rack_Get__(blocks, <size_t>block_index)
        return Block_Get__(block, address - Block_Start(block))


cdef object Memory_Peek(const Memory_* that, object address):
    cdef:
        int value

    value = Memory_Peek_(that, <addr_t>address)
    return None if value < 0 else value


cdef vint Memory_PokeNone_(Memory_* that, addr_t address) except -1:
    if address < that.trim_start:
        return 0
    if address >= that.trim_endex:
        return 0

    # Standard clear method
    Memory_Erase__(that, address, address + 1, False)  # clear
    return 0


cdef vint Memory_Poke_(Memory_* that, addr_t address, byte_t item) except -1:
    cdef:
        Rack_* blocks
        size_t block_count
        size_t block_index
        Block_* block
        addr_t block_start
        addr_t block_endex
        Block_* block2
        addr_t block_start2

    if address < that.trim_start:
        return 0
    if address >= that.trim_endex:
        return 0

    blocks = that.blocks
    block_index = Rack_IndexEndex(blocks, address) - 1
    block_count = Rack_Length(blocks)

    if block_index < block_count:
        block = Rack_Get__(blocks, block_index)
        block_start = Block_Start(block)
        block_endex = Block_Endex(block)

        if block_start <= address < block_endex:
            # Address within existing block, update directly
            address -= block_start
            Block_Set__(block, <size_t>address, item)
            return 0

        elif address == block_endex:
            # Address just after the end of the block, append
            block = Block_Append(block, item)
            Rack_Set__(blocks, block_index, block)  # update pointer

            block_index += 1
            if block_index < block_count:
                block2 = Rack_Get__(blocks, block_index)
                block_start2 = Block_Start(block2)

                if block_endex + 1 == block_start2:
                    # Merge with the following contiguous block
                    block = Block_Extend(block, block2)
                    Rack_Set__(blocks, block_index - 1, block)  # update pointer
                    that.blocks = blocks = Rack_Pop_(blocks, block_index, NULL)
            return 0

        else:
            block_index += 1
            if block_index < block_count:
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)

                if address + 1 == block_start:
                    # Prepend to the next block
                    block = Block_AppendLeft(block, item)
                    Rack_Set__(blocks, block_index, block)  # update pointer
                    block.address -= 1  # update address
                    return 0

    # There is no faster way than the standard block writing method
    Memory_Erase__(that, address, address + 1, False)  # clear
    Memory_Place__(that, address, 1, &item, False)  # write

    Memory_Crop_(that, that.trim_start, that.trim_endex)
    return 0


cdef vint Memory_Poke(Memory_* that, object address, object item) except -1:
    cdef:
        addr_t address_ = <addr_t>address

    if item is None:
        Memory_PokeNone_(that, address_)
    else:
        if isinstance(item, int):
            Memory_Poke_(that, address_, <byte_t>item)
        else:
            if len(item) != 1:
                raise ValueError('expecting single item')
            Memory_Poke_(that, address_, <byte_t>item[0])


cdef Memory_* Memory_Extract__(const Memory_* that, addr_t start, addr_t endex,
                               size_t pattern_size, const byte_t* pattern_ptr,
                               saddr_t step, bint bound) except NULL:
    cdef:
        const Rack_* blocks = that.blocks
        size_t block_count
        size_t block_index
        size_t block_index_start
        size_t block_index_endex
        Memory_* memory = Memory_Alloc()
        Rack_* blocks2
        Block_* block2
        addr_t offset
        Block_* pattern = NULL
        int value
        saddr_t skip
        Rover_* rover = NULL

    if endex < start:
        endex = start

    if step == 1:
        if start < endex and Rack_Bool(blocks):
            # Copy all the blocks except those completely outside selection
            block_index_start = Rack_IndexStart(blocks, start)
            block_index_endex = Rack_IndexEndex(blocks, endex)
            block_count = block_index_endex - block_index_start

            if block_count:
                memory_blocks = memory.blocks
                block_count = block_index_endex - block_index_start
                try:
                    memory.blocks = memory_blocks = Rack_Reserve_(memory_blocks, 0, block_count)

                    for block_index in range(block_count):
                        block2 = Rack_Get_(blocks, block_index_start + block_index)
                        block2 = Block_Copy(block2)
                        Rack_Set__(memory_blocks, block_index, block2)  # update pointer

                    # Trim cloned data before the selection start address
                    block_index = 0
                    block2 = Rack_Get_(memory_blocks, block_index)
                    block_start = Block_Start(block2)
                    if block_start < start:
                        block2 = Block_DelSlice_(block2, 0, <size_t>(start - block_start))
                        block2.address = start
                        Rack_Set__(memory_blocks, block_index, block2)

                    # Trim cloned data after the selection end address
                    block_index = block_count - 1
                    block2 = Rack_Get_(memory_blocks, block_index)
                    block_endex = Block_Endex(block2)
                    if endex < block_endex:
                        block_start = Block_Start(block2)
                        if block_start < endex:
                            block2 = Block_DelSlice_(block2, <size_t>(endex - block_start), Block_Length(block2))
                            block2.address = block_start
                            Rack_Set__(memory_blocks, block_index, block2)  # update pointer
                        else:
                            memory.blocks = memory_blocks = Rack_Pop__(memory_blocks, &block2)
                            block2 = Block_Release(block2)  # orphan
                except:
                    memory = Memory_Free(memory)  # orphan
                    raise

            if pattern_size and pattern_ptr:
                pattern = Block_Create(0, pattern_size, pattern_ptr)
                try:
                    Memory_Flood_(memory, start, endex, &pattern)
                except:
                    Block_Free(pattern)  # orphan
                    memory = Memory_Free(memory)  # orphan
                    raise
    else:
        if step > 1:
            block2 = NULL
            offset = start
            rover = Rover_Create(that, start, endex, pattern_size, pattern_ptr, True, False)
            try:
                while True:
                    value = Rover_Next_(rover)
                    if value < 0:
                        if block2:
                            memory.blocks = Rack_Append(memory.blocks, block2)
                            block2 = NULL
                    else:
                        if not block2:
                            block2 = Block_Alloc(offset, 0, False)
                        block2 = Block_Append(block2, <byte_t>value)

                    offset += 1
                    for skip in range(step - 1):
                        Rover_Next_(rover)
            except StopIteration:
                if block2:
                    memory.blocks = Rack_Append(memory.blocks, block2)
                    block2 = NULL
            finally:
                block2 = Block_Free(block2)  # orphan
                rover = Rover_Free(rover)

            if bound:
                endex = offset
    if bound:
        memory.trim_start = start
        memory.trim_endex = endex
        memory.trim_start_ = True
        memory.trim_endex_ = True

    return memory


cdef object Memory_Extract_(const Memory_* that, addr_t start, addr_t endex,
                            size_t pattern_size, const byte_t* pattern_ptr,
                            saddr_t step, bint bound):
    cdef:
        Memory_* memory_ = Memory_Extract__(that, start, endex, pattern_size, pattern_ptr, step, bound)

    return Memory_AsObject(memory_)


cdef object Memory_Extract(const Memory_* that, object start, object endex,
                           object pattern, object step, bint bound):
    cdef:
        addr_t start_
        addr_t endex_
        const byte_t[:] pattern_view
        byte_t pattern_value
        size_t pattern_size
        const byte_t* pattern_ptr
        saddr_t step_ = <saddr_t>1 if step is None else <saddr_t>step

    if pattern is None:
        pattern_size = 0
        pattern_ptr = NULL

    elif isinstance(pattern, int):
        pattern_value = <byte_t>pattern
        pattern_size = 1
        pattern_ptr = &pattern_value

    else:
        pattern_view = pattern
        pattern_size = len(pattern_view)
        with cython.boundscheck(False):
            pattern_ptr = &pattern_view[0]

    start_ = Memory_Start(that) if start is None else <addr_t>start
    endex_ = Memory_Endex(that) if endex is None else <addr_t>endex

    return Memory_Extract_(that, start_, endex_, pattern_size, pattern_ptr, step_, bound)


cdef vint Memory_ShiftLeft_(Memory_* that, addr_t offset) except -1:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_index
        Block_* block

    if offset and Rack_Bool(blocks):
        Memory_PretrimStart_(that, ADDR_MAX, offset)
        blocks = that.blocks

        for block_index in range(Rack_Length(blocks)):
            block = Rack_Get__(blocks, block_index)
            block.address -= offset


cdef vint Memory_ShiftRight_(Memory_* that, addr_t offset) except -1:
    cdef:
        Rack_* blocks = that.blocks
        size_t block_index
        Block_* block

    if offset and Rack_Bool(blocks):
        Memory_PretrimEndex_(that, ADDR_MIN, offset)
        blocks = that.blocks

        for block_index in range(Rack_Length(blocks)):
            block = Rack_Get__(blocks, block_index)
            block.address += offset


cdef vint Memory_Shift(Memory_* that, object offset) except -1:
    if offset:
        if offset < 0:
            Memory_ShiftLeft_(that, <addr_t>-offset)
        else:
            Memory_ShiftRight_(that, <addr_t>offset)


cdef vint Memory_Reserve_(Memory_* that, addr_t address, addr_t size) except -1:
    cdef:
        addr_t offset
        Rack_* blocks = that.blocks
        size_t block_count
        size_t block_index
        Block_* block
        addr_t block_start
        Block_* block2

    if size and Rack_Bool(blocks):
        Memory_PretrimEndex_(that, address, size)

        block_index = Rack_IndexStart(blocks, address)
        block_count = Rack_Length(blocks)

        if block_index < block_count:
            block = Rack_Get_(blocks, block_index)
            block_start = Block_Start(block)

            if address > block_start:
                # Split into two blocks, reserving emptiness
                CheckAddSizeU(block_count, 1)  # ensure free slot
                offset = address - block_start
                block2 = Block_GetSlice_(block, offset, SIZE_HMAX)
                try:
                    block = Block_DelSlice_(block, offset, SIZE_HMAX)

                    Rack_Set__(blocks, block_index, block)  # update pointer
                    block_index += 1

                    CheckAddAddrU(address, size)
                    block2.address = address + size
                    that.blocks = blocks = Rack_Insert(blocks, block_index, block2)
                except:
                    block2 = Block_Free(block2)  # orphan
                    raise
                block_index += 1

            for block_index in range(block_index, Rack_Length(blocks)):
                block = Rack_Get_(blocks, block_index)
                block.address += size


cdef vint Memory_Reserve(Memory_* that, object address, object size) except -1:
    Memory_Reserve_(that, <addr_t>address, <addr_t>size)


cdef vint Memory_Place__(Memory_* that, addr_t address, size_t size, const byte_t* buffer,
                         bint shift_after) except -1:
    cdef:
        Rack_* blocks
        size_t block_index
        Block_* block
        addr_t block_start
        addr_t block_endex
        Block_* block2
        addr_t block_start2
        size_t offset

    if size:
        blocks = that.blocks
        block_index = Rack_IndexStart(blocks, address)

        if block_index:
            block = Rack_Get_(blocks, block_index - 1)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)

            if block_endex == address:
                # Extend previous block
                block = Block_Extend_(block, size, buffer)
                Rack_Set__(blocks, block_index - 1, block)  # update pointer

                # Shift blocks after
                if shift_after:
                    for block_index in range(block_index, Rack_Length(blocks)):
                        block = Rack_Get_(blocks, block_index)
                        CheckAddAddrU(block.address, size)
                        block.address += size
                else:
                    if block_index < Rack_Length(blocks):
                        CheckAddAddrU(block_endex, size)
                        block_endex += size

                        block2 = Rack_Get_(blocks, block_index)
                        block_start2 = Block_Start(block2)

                        # Merge with next block
                        if block_endex == block_start2:
                            block = Block_Extend(block, block2)
                            Rack_Set__(blocks, block_index - 1, block)  # update pointer
                            that.blocks = blocks = Rack_Pop_(blocks, block_index, NULL)
                return 0

        if block_index < Rack_Length(blocks):
            block = Rack_Get_(blocks, block_index)
            block_start = Block_Start(block)

            if address < block_start:
                if shift_after:
                    # Insert a standalone block before
                    block = Block_Create(address, size, buffer)
                    try:
                        that.blocks = blocks = Rack_Insert(blocks, block_index, block)
                    except:
                        Block_Free(block)  # orphan
                        raise
                else:
                    CheckAddAddrU(address, size)
                    if address + size == block_start:
                        # Merge with next block
                        block = Rack_Get_(blocks, block_index)
                        block.address = address
                        block = Block_ExtendLeft_(block, size, buffer)
                        Rack_Set__(blocks, block_index, block)  # update pointer
                    else:
                        # Insert a standalone block before
                        block = Block_Create(address, size, buffer)
                        try:
                            that.blocks = blocks = Rack_Insert(blocks, block_index, block)
                        except:
                            Block_Free(block)  # orphan
                            raise
            else:
                # Insert buffer into the current block
                CheckSubAddrU(address, block_start)
                CheckAddrToSizeU(address - block_start)
                offset = <size_t>(address - block_start)
                block = Block_Reserve_(block, offset, size, False)
                block = Block_Write_(block, offset, size, buffer)
                Rack_Set__(blocks, block_index, block)  # update pointer

            # Shift blocks after
            if shift_after:
                for block_index in range(block_index + 1, Rack_Length(blocks)):
                    block = Rack_Get__(blocks, block_index)
                    CheckAddAddrU(block.address, size)
                    block.address += size

        else:
            # Append a standalone block after
            block = Block_Create(address, size, buffer)
            try:
                that.blocks = blocks = Rack_Append(blocks, block)
            except:
                Block_Free(block)  # orphan
                raise


cdef vint Memory_Erase__(Memory_* that, addr_t start, addr_t endex, bint shift_after) except -1:
    cdef:
        addr_t size
        addr_t offset

        Rack_* blocks = that.blocks
        size_t block_index
        size_t inner_start
        size_t inner_endex

        Block_* block = NULL
        addr_t block_start
        addr_t block_endex

        Block_* block2 = NULL
        addr_t block_start2

    if Rack_Bool(blocks) and endex > start:
        size = endex - start
        block_index = Rack_IndexStart(blocks, start)

        # Delete final/inner part of deletion start block
        if block_index < Rack_Length(blocks):
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            if start > block_start:
                if shift_after:
                    CheckAddrToSizeU(start - block_start)
                    CheckAddrToSizeU(endex - block_start)
                    block = Block_DelSlice_(block, start - block_start, endex - block_start)
                    Rack_Set__(blocks, block_index, block)  # update pointer
                else:
                    try:
                        CheckAddrToSizeU(start - block_start)
                        block = Block_GetSlice_(block, 0, start - block_start)
                        block.address = block_start
                        that.blocks = blocks = Rack_Insert_(blocks, block_index, block)
                    except:
                        block = Block_Free(block)  # orphan
                        raise
                block_index += 1  # skip this from inner part

        # Delete initial part of deletion end block
        inner_start = block_index
        for block_index in range(block_index, Rack_Length(blocks)):
            block = Rack_Get__(blocks, block_index)

            block_start = Block_Start(block)
            if endex <= block_start:
                break  # inner ends before here

            block_endex = Block_Endex(block)
            if endex < block_endex:
                offset = endex - block_start
                CheckAddrToSizeU(offset)
                CheckAddAddrU(block.address, offset)
                block = Block_DelSlice_(block, 0, <size_t>offset)
                block.address += offset  # update address
                Rack_Set__(blocks, block_index, block)  # update pointer
                break  # inner ends before here
        else:
            block_index = Rack_Length(blocks)
        inner_endex = block_index

        if shift_after:
            # Check if inner deletion can be merged
            if inner_start and inner_endex < Rack_Length(blocks):
                block = Rack_Get__(blocks, inner_start - 1)
                block_endex = Block_Endex(block)

                block2 = Rack_Get__(blocks, inner_endex)
                block_start2 = Block_Start(block2)

                if block_endex + size == block_start2:
                    block = Block_Extend(block, block2)  # merge deletion boundaries
                    Rack_Set__(blocks, inner_start - 1, block)  # update pointer
                    inner_endex += 1  # add to inner deletion
                    block_index += 1  # skip address update

            # Shift blocks after deletion
            for block_index in range(block_index, Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                CheckSubAddrU(block.address, size)
                block.address -= size  # update address

        # Delete inner full blocks
        if inner_start < inner_endex:
            that.blocks = blocks = Rack_DelSlice_(blocks, inner_start, inner_endex)


cdef vint Memory_InsertSame_(Memory_* that, addr_t address, Memory_* data) except -1:
    cdef:
        addr_t data_start = Memory_Start(data)
        addr_t data_endex = Memory_Endex(data)
        addr_t data_size = data_endex - data_start

    if data_size:
        Memory_Reserve_(that, address, data_size)
        Memory_WriteSame_(that, address, data, True)


cdef vint Memory_InsertRaw_(Memory_* that, addr_t address, size_t data_size, const byte_t* data_ptr) except -1:
    if data_size:
        Memory_Reserve_(that, address, data_size)
        Memory_WriteRaw_(that, address, data_size, data_ptr)


cdef vint Memory_Insert(Memory_* that, object address, object data) except -1:
    cdef:
        addr_t address_ = <addr_t>address
        const byte_t[:] data_view
        byte_t data_value
        size_t data_size
        const byte_t* data_ptr

    if isinstance(data, Memory):
        Memory_InsertSame_(that, address_, (<Memory>data)._)

    else:
        if isinstance(data, int):
            data_value = <byte_t>data
            data_size = 1
            data_ptr = &data_value
        else:
            data_view = data
            data_size = len(data_view)
            with cython.boundscheck(False):
                data_ptr = &data_view[0]

        Memory_InsertRaw_(that, address_, data_size, data_ptr)


cdef vint Memory_Delete_(Memory_* that, addr_t start, addr_t endex) except -1:
    if start < endex:
        Memory_Erase__(that, start, endex, True)  # delete


cdef vint Memory_Delete(Memory_* that, object start, object endex) except -1:
    cdef:
        addr_t start_ = Memory_Start(that) if start is None else <addr_t> start
        addr_t endex_ = Memory_Endex(that) if endex is None else <addr_t> endex

    Memory_Delete_(that, start_, endex_)


cdef vint Memory_Clear_(Memory_* that, addr_t start, addr_t endex) except -1:
    if start < endex:
        Memory_Erase__(that, start, endex, False)  # clear


cdef vint Memory_Clear(Memory_* that, object start, object endex) except -1:
    cdef:
        addr_t start_ = Memory_Start(that) if start is None else <addr_t> start
        addr_t endex_ = Memory_Endex(that) if endex is None else <addr_t> endex

    Memory_Clear_(that, start_, endex_)


cdef vint Memory_PretrimStart_(Memory_* that, addr_t endex_max, addr_t size) except -1:
    cdef:
        addr_t content_start
        addr_t trim_start
        addr_t endex

    if size:
        trim_start = that.trim_start if that.trim_start_ else ADDR_MIN
        if CannotAddAddrU(trim_start, size):
            endex = ADDR_MAX
        else:
            endex = trim_start + size

        if endex > endex_max:
            endex = endex_max

        content_start = Memory_ContentStart(that)
        Memory_Erase__(that, content_start, endex, False)  # clear


cdef vint Memory_PretrimStart(Memory_* that, object endex_max, object size) except -1:
        cdef:
            addr_t endex_max_ = ADDR_MAX if endex_max is None else <addr_t>endex_max

        Memory_PretrimStart_(that, endex_max_, <addr_t>size)


cdef vint Memory_PretrimEndex_(Memory_* that, addr_t start_min, addr_t size) except -1:
    cdef:
        addr_t content_endex
        addr_t trim_endex
        addr_t start

    if size:
        trim_endex = that.trim_endex if that.trim_endex_ else ADDR_MAX
        if CannotSubAddrU(trim_endex, size):
            start = ADDR_MIN
        else:
            start = trim_endex - size

        if start < start_min:
            start = start_min

        content_endex = Memory_ContentEndex(that)
        Memory_Erase__(that, start, content_endex, False)  # clear


cdef vint Memory_PretrimEndex(Memory_* that, object start_min, object size) except -1:
        cdef:
            addr_t start_min_ = ADDR_MIN if start_min is None else <addr_t>start_min

        Memory_PretrimEndex_(that, start_min_, <addr_t>size)


cdef vint Memory_Crop_(Memory_* that, addr_t start, addr_t endex) except -1:
    cdef:
        addr_t block_start
        addr_t block_endex

    # Trim blocks exceeding before memory start
    if Rack_Bool(that.blocks):
        block_start = Block_Start(Rack_First_(that.blocks))

        if block_start < start:
            Memory_Erase__(that, block_start, start, False)  # clear

    # Trim blocks exceeding after memory end
    if Rack_Bool(that.blocks):
        block_endex = Block_Endex(Rack_Last_(that.blocks))

        if endex < block_endex:
            Memory_Erase__(that, endex, block_endex, False)  # clear


cdef vint Memory_Crop(Memory_* that, object start, object endex) except -1:
    cdef:
        addr_t start_
        addr_t endex_

    start_, endex_ = Memory_Bound(that, start, endex)
    Memory_Crop_(that, start_, endex_)


cdef vint Memory_WriteSame_(Memory_* that, addr_t address, const Memory_* data, bint clear) except -1:
    cdef:
        addr_t start = Memory_Start(data)
        addr_t endex = Memory_Endex(data)
        addr_t size = endex - start
        addr_t delta

        addr_t trim_start
        addr_t trim_endex
        bint trim_start_
        bint trim_endex_

        const Rack_* blocks
        size_t block_count
        size_t block_index
        const Block_* block
        addr_t block_start
        addr_t block_endex
        size_t block_size
        size_t block_offset

    if not size:
        return 0

    blocks = data.blocks
    block_count = Rack_Length(blocks)
    if block_count:
        CheckAddAddrU(Block_Endex(Rack_Last__(blocks)), address)

    CheckAddAddrU(endex, address)
    endex += address
    trim_start_ = that.trim_start_
    trim_start = that.trim_start

    if trim_start_ and endex <= trim_start:
        return 0

    CheckAddAddrU(start, address)
    start += address
    trim_endex_ = that.trim_endex_
    trim_endex = that.trim_endex

    if trim_endex_ and trim_endex <= start:
        return 0

    if clear:
        # Clear anything between source data boundaries
        Memory_Erase__(that, start, endex, False)  # clear
    else:
        # Clear only overwritten ranges
        for block_index in range(block_count):
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block) + address
            block_endex = Block_Endex(block) + address
            Memory_Erase__(that, block_start, block_endex, False)  # clear

    for block_index in range(block_count):
        block = Rack_Get__(blocks, block_index)

        block_endex = Block_Endex(block) + address
        if trim_start_ and block_endex <= trim_start:
            continue

        block_start = Block_Start(block) + address
        if trim_endex_ and trim_endex <= block_start:
            break

        block_size = block_endex - block_start
        block_offset = 0

        # Trim before memory
        if trim_start_ and block_start < trim_start:
            delta = trim_start - block_start
            CheckAddrToSizeU(delta)
            block_start += <size_t>delta
            block_size  -= <size_t>delta
            block_offset = <size_t>delta

        # Trim after memory
        if trim_endex_ and trim_endex < block_endex:
            delta = block_endex - trim_endex
            CheckAddrToSizeU(delta)
            block_endex -= <size_t>delta
            block_size  -= <size_t>delta

        Memory_Place__(that, block_start, block_size, Block_At__(block, block_offset), False)  # write


cdef vint Memory_WriteRaw_(Memory_* that, addr_t address, size_t data_size, const byte_t* data_ptr) except -1:
    cdef:
        addr_t size = data_size
        addr_t start
        addr_t endex
        addr_t offset

        addr_t trim_start
        addr_t trim_endex
        bint trim_start_
        bint trim_endex_

        Rack_* blocks
        size_t block_count
        Block_* block

    if not size:
        return 0
    elif size == 1:
        Memory_Poke(that, address, data_ptr[0])  # faster
        return 0

    CheckAddAddrU(address, size)
    start = address
    endex = start + size

    trim_start_ = that.trim_start_
    trim_start = that.trim_start
    if trim_start_ and endex <= trim_start:
        return 0

    trim_endex_ = that.trim_endex_
    trim_endex = that.trim_endex
    if trim_endex_ and trim_endex <= start:
        return 0

    # Trim before memory
    if trim_start_ and start < trim_start:
        offset = trim_start - start
        CheckAddrToSizeU(offset)
        start += offset
        size -= <size_t>offset
        data_ptr += <size_t>offset

    # Trim after memory
    if trim_endex_ and trim_endex < endex:
        offset = endex - trim_endex
        CheckAddrToSizeU(offset)
        endex -= offset
        size -= <size_t>offset

    # Check if extending the actual content
    CheckAddrToSizeU(size)
    blocks = that.blocks
    block_count = Rack_Length(blocks)
    if block_count:
        block = Rack_Last__(blocks)
        if start == Block_Endex(block):
            block = Block_Extend_(block, <size_t>size, data_ptr)  # might be faster
            Rack_Set__(blocks, block_count - 1, block)  # update pointer
            return 0

    # Standard write method
    Memory_Erase__(that, start, endex, False)  # clear
    Memory_Place__(that, start, <size_t>size, data_ptr, False)  # write
    return 0


cdef vint Memory_Write(Memory_* that, object address, object data, bint clear) except -1:
    cdef:
        addr_t address_ = <addr_t>address
        const byte_t[:] data_view
        byte_t data_value
        size_t data_size
        const byte_t* data_ptr
        addr_t data_start
        addr_t data_endex

    if isinstance(data, Memory):
        Memory_WriteSame_(that, address_, (<Memory>data)._, clear)

    elif isinstance(data, ImmutableMemory):
        for block_start, block_data in data:
            data_start = <addr_t>block_start
            data_view = block_data
            with cython.boundscheck(False):
                data_ptr = &data_view[0]
            data_size = <size_t>len(data_view)
            CheckAddAddrU(data_start, data_size)
            data_endex = data_start + data_size
            CheckAddAddrU(data_start, address_)
            CheckAddAddrU(data_endex, address_)

            Memory_WriteRaw_(that, data_start + address_, data_size, data_ptr)

    else:
        if isinstance(data, int):
            data_value = <byte_t>data
            data_size = 1
            data_ptr = &data_value
        else:
            data_view = data
            data_size = <size_t>len(data_view)
            with cython.boundscheck(False):
                data_ptr = &data_view[0]

        Memory_WriteRaw_(that, address_, data_size, data_ptr)


cdef vint Memory_Fill_(Memory_* that, addr_t start, addr_t endex, Block_** pattern, addr_t start_) except -1:
    cdef:
        size_t offset
        size_t size

    if start < endex:
        CheckAddrToSizeU(endex - start)
        if not Block_Bool(pattern[0]):
            raise ValueError('non-empty pattern required')

        if start > start_:
            offset = start - start_
            CheckAddrToSizeU(offset)
            Block_RotateLeft_(pattern[0], <size_t>offset)

        # Resize the pattern to the target range
        size = <size_t>(endex - start)
        pattern[0] = Block_RepeatToSize(pattern[0], size)

        # Standard write method
        Memory_Erase__(that, start, endex, False)  # clear
        Memory_Place__(that, start, size, Block_At__(pattern[0], 0), False)  # write


cdef vint Memory_Fill(Memory_* that, object start, object endex, object pattern) except -1:
        cdef:
            addr_t start__
            addr_t start_
            addr_t endex_
            Block_* pattern_ = NULL

        start_, endex_ = Memory_Bound(that, start, endex)
        if start_ < endex_:
            pattern_ = Block_FromObject(0, pattern, False)  # size checked later on
            try:
                start__ = Memory_Start(that) if start is None else <addr_t>start
                Memory_Fill_(that, start_, endex_, &pattern_, start__)
            finally:
                Block_Free(pattern_)  # orphan


cdef vint Memory_Flood_(Memory_* that, addr_t start, addr_t endex, Block_** pattern) except -1:
    cdef:
        Rack_* blocks
        const Block_* block
        addr_t block_start
        addr_t block_endex
        size_t block_index_start
        size_t block_index_endex
        addr_t offset

    if start < endex:
        blocks = that.blocks
        block_index_start = Rack_IndexStart(blocks, start)

        # Check if touching previous block
        if block_index_start:
            block = Rack_Get__(blocks, block_index_start - 1)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)
            if block_endex == start:
                block_index_start -= 1

        # Manage block near start
        if block_index_start < Rack_Length(blocks):
            block = Rack_Get__(blocks, block_index_start)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)

            if block_start <= start and endex <= block_endex:
                return 0  # no emptiness to flood

            if block_start < start:
                offset = start - block_start
                CheckAddrToSizeU(offset)
                Block_RotateRight_(pattern[0], <size_t>offset)
                start = block_start

        # Manage block near end
        block_index_endex = Rack_IndexEndex(blocks, endex)
        if block_index_start < block_index_endex:
            block = Rack_Get__(blocks, block_index_endex - 1)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)
            if endex < block_endex:
                endex = block_endex

        CheckAddrToSizeU(endex - start)
        if not Block_Bool(pattern[0]):
            raise ValueError('non-empty pattern required')

        size = <size_t>(endex - start)
        pattern[0] = Block_RepeatToSize(pattern[0], size)
        pattern[0].address = start

        for block_index in range(block_index_start, block_index_endex):
            block = Rack_Get__(blocks, block_index)
            offset = Block_Start(block) - start
            # CheckAddrToSizeU(offset)  # implied
            pattern[0] = Block_Write_(pattern[0], <size_t>offset, Block_Length(block), Block_At__(block, 0))

        that.blocks = blocks = Rack_DelSlice_(blocks, block_index_start, block_index_endex)
        that.blocks = blocks = Rack_Insert_(blocks, block_index_start, pattern[0])


cdef vint Memory_Flood(Memory_* that, object start, object endex, object pattern) except -1:
        cdef:
            addr_t start_
            addr_t endex_
            Block_* pattern_ = NULL

        start_, endex_ = Memory_Bound(that, start, endex)
        if start_ < endex_:
            pattern_ = Block_FromObject(0, pattern, False)  # size checked later on
            try:
                Memory_Flood_(that, start_, endex_, &pattern_)
            except:
                Block_Free(pattern_)  # orphan
                raise


# =====================================================================================================================

cdef Rover_* Rover_Alloc() except NULL:
    cdef:
        Rover_* that = <Rover_*>PyMem_Calloc(Rover_HEADING, 1)

    if that == NULL:
        raise MemoryError()
    return that


cdef Rover_* Rover_Free(Rover_* that) except? NULL:
    if that:
        Rover_Release(that)
        PyMem_Free(that)
    return NULL


cdef size_t Rover_Sizeof(const Rover_* that):
    if that:
        return Rover_HEADING
    return 0


cdef Rover_* Rover_Create(
    const Memory_* memory,
    addr_t start,
    addr_t endex,
    size_t pattern_size,
    const byte_t* pattern_data,
    bint forward,
    bint infinite,
) except NULL:
    cdef:
        Block_* block = NULL
        addr_t offset
        size_t pattern_offset

    if forward:
        if endex < start:
            endex = start
    else:
        if start > endex:
            start = endex

    if (not pattern_data) != (not pattern_size):
        raise ValueError('non-empty pattern required')

    if pattern_size and not forward:
        with cython.cdivision(True):
            pattern_offset = <size_t>((endex - start) % pattern_size)
    else:
        pattern_offset = 0

    that = Rover_Alloc()

    that.forward = forward
    that.infinite = infinite
    that.start = start
    that.endex = endex
    that.address = start if forward else endex

    that.pattern_size = pattern_size
    that.pattern_data = pattern_data
    that.pattern_offset = pattern_offset

    that.memory = memory
    that.block_count = Rack_Length(memory.blocks)

    try:
        if that.block_count:
            if forward:
                that.block_index = Rack_IndexStart(memory.blocks, start)
                if that.block_index < that.block_count:
                    block = Rack_Get_(memory.blocks, that.block_index)
                    that.block_start = Block_Start(block)
                    that.block_endex = Block_Endex(block)

                    offset = start if start >= that.block_start else that.block_start
                    if offset > that.block_endex:
                        offset = that.block_endex
                    offset -= that.block_start
                    CheckAddrToSizeU(offset)

                    block = Block_Acquire(block)
                    that.block = block
                    that.block_ptr = Block_At__(block, <size_t>offset)
            else:
                that.block_index = Rack_IndexEndex(memory.blocks, endex)
                if that.block_index:
                    block = Rack_Get_(memory.blocks, that.block_index - 1)
                    that.block_start = Block_Start(block)
                    that.block_endex = Block_Endex(block)

                    offset = endex if endex >= that.block_start else that.block_start
                    if offset > that.block_endex:
                        offset = that.block_endex
                    offset -= that.block_start
                    CheckAddrToSizeU(offset)

                    block = Block_Acquire(block)
                    that.block = block
                    that.block_ptr = Block_At__(block, <size_t>offset)
    except:
        that = Rover_Free(that)
        raise

    return that


cdef addr_t Rover_Length(const Rover_* that) nogil:
    return that.endex - that.start


cdef bint Rover_HasNext(const Rover_* that) nogil:
    if that.forward:
        return that.address < that.endex
    else:
        return that.address > that.start


cdef int Rover_Next_(Rover_* that) except -2:
    cdef:
        Block_* block = NULL
        int value = -1

    try:
        if that.forward:
            while True:  # loop to move to the next block when necessary
                if that.address < that.endex:
                    if that.block_index < that.block_count:
                        if that.address < that.block_start:
                            that.address += 1
                            if that.pattern_size:
                                value = <int><unsigned>that.pattern_data[that.pattern_offset]
                            else:
                                value = -1
                            break

                        elif that.address < that.block_endex:
                            that.address += 1
                            value = that.block_ptr[0]
                            that.block_ptr += 1
                            break

                        else:
                            that.block_index += 1
                            if that.block_index < that.block_count:
                                that.block = Block_Release(that.block)
                                that.block = NULL
                                block = Rack_Get_(that.memory.blocks, that.block_index)
                                block = Block_Acquire(block)
                                that.block = block
                                that.block_start = Block_Start(block)
                                that.block_endex = Block_Endex(block)
                                that.block_ptr = Block_At_(block, 0)
                            continue
                    else:
                        that.address += 1
                        if that.pattern_size:
                            value = <int><unsigned>that.pattern_data[that.pattern_offset]
                        else:
                            value = -1
                        break

                elif that.infinite:
                    if that.pattern_size:
                        value = <int><unsigned>that.pattern_data[that.pattern_offset]
                    else:
                        value = -1

                else:
                    raise StopIteration()

            if that.pattern_size:
                if that.pattern_offset < that.pattern_size - 1:
                    that.pattern_offset += 1
                else:
                    that.pattern_offset = 0
        else:
            if that.pattern_size:
                if that.pattern_offset > 0:
                    that.pattern_offset -= 1
                else:
                    that.pattern_offset = that.pattern_size - 1

            while True:  # loop to move to the next block when necessary
                if that.address > that.start:
                    if that.block_index:
                        if that.address > that.block_endex:
                            that.address -= 1
                            if that.pattern_size:
                                value = <int><unsigned>that.pattern_data[that.pattern_offset]
                            else:
                                value = -1
                            break

                        elif that.address > that.block_start:
                            that.address -= 1
                            that.block_ptr -= 1
                            value = that.block_ptr[0]
                            break

                        else:
                            that.block_index -= 1
                            if that.block_index:
                                that.block = Block_Release(that.block)
                                that.block = NULL
                                block = Rack_Get_(that.memory.blocks, that.block_index - 1)
                                block = Block_Acquire(block)
                                that.block = block
                                that.block_start = Block_Start(block)
                                that.block_endex = Block_Endex(block)
                                that.block_ptr = Block_At__(block, Block_Length(block))
                            value = -1
                            continue
                    else:
                        that.address -= 1
                        if that.pattern_size:
                            value = <int><unsigned>that.pattern_data[that.pattern_offset]
                        else:
                            value = -1
                        break

                elif that.infinite:
                    if that.pattern_size:
                        value = <int><unsigned>that.pattern_data[that.pattern_offset]
                    else:
                        value = -1

                else:
                    raise StopIteration()

        return value

    except:
        that.block = Block_Release(that.block)  # preempt
        raise


cdef object Rover_Next(Rover_* that):
    cdef:
        int value = Rover_Next_(that)

    return None if value < 0 else <object>value


cdef vint Rover_Release(Rover_* that) except -1:
    that.address = that.endex if that.forward else that.start
    that.block = Block_Release(that.block)
    that.memory = NULL


cdef bint Rover_Forward(const Rover_* that) nogil:
    return that.forward


cdef bint Rover_Infinite(const Rover_* that) nogil:
    return that.infinite


cdef addr_t Rover_Address(const Rover_* that) nogil:
    return that.address


cdef addr_t Rover_Start(const Rover_* that) nogil:
    return that.start


cdef addr_t Rover_Endex(const Rover_* that) nogil:
    return that.endex


# =====================================================================================================================

cdef class Memory:
    r"""Virtual memory.

    This class is a handy wrapper around `blocks`, so that it can behave mostly
    like a :obj:`bytearray`, but on sparse chunks of data.

    Please look at examples of each method to get a glimpse of the features of
    this class.

    On creation, at most one of `memory`, `blocks`, or `data` can be specified.

    The Cython implementation limits the address range to that of the integral
    type ``uint_fast64_t``.

    Attributes:
        _trim_start (int):
            Memory trimming start address. Any data before this address is
            automatically discarded; disabled if ``None``.

        _trim_endex (int):
            Memory trimming exclusive end address. Any data at or after this
            address is automatically discarded; disabled if ``None``.

    Arguments:
        start (int):
            Optional memory start address.
            Anything before will be trimmed away.

        endex (int):
            Optional memory exclusive end address.
            Anything at or after it will be trimmed away.

    Examples:
        >>> from cbytesparse.c import Memory

        >>> memory = Memory()
        >>> memory.to_blocks()
        []

        >>> memory = Memory.from_bytes(b'Hello, World!', offset=5)
        >>> memory.to_blocks()
        [[5, b'Hello, World!']]
    """

    def __add__(
        self: Memory,
        value: Union[AnyBytes, ImmutableMemory],
    ) -> Memory:
        cdef:
            Memory_* memory_ = Memory_Add(self._, value)
            Memory memory = Memory_AsObject(memory_)

        return memory

    def __bool__(
        self: Memory,
    ) -> bool:
        r"""Has any items.

        Returns:
            bool: Has any items.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory()
            >>> bool(memory)
            False

            >>> memory = Memory.from_bytes(b'Hello, World!', offset=5)
            >>> bool(memory)
            True
        """

        return Memory_Bool(self._)

    def __bytes__(
        self: Memory,
    ) -> bytes:
        r"""Creates a bytes clone.

        Returns:
            :obj:`bytes`: Cloned data.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).
        """
        cdef:
            const Memory_* memory = self._
            BlockView view = Memory_View_(memory, Memory_Start(memory), Memory_Endex(memory))

        try:
            data = view.__bytes__()
        finally:
            view.release()
        return data

    def __cinit__(self):
        r"""Cython constructor."""
        self._ = NULL

    def __contains__(
        self: Memory,
        item: Union[AnyBytes, Value],
    ) -> bool:
        r"""Checks if some items are contained.

        Arguments:
            item (items):
                Items to find. Can be either some byte string or an integer.

        Returns:
            bool: Item is contained.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[1 | 2 | 3]|   |[x | y | z]|
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'123'], [9, b'xyz']])
            >>> b'23' in memory
            True
            >>> ord('y') in memory
            True
            >>> b'$' in memory
            False
        """

        return Memory_Contains(self._, item)

    def __copy__(
        self: Memory,
    ) -> ImmutableMemory:
        r"""Creates a shallow copy.

        Note:
            The Cython implementation actually creates a deep copy.

        Returns:
            :obj:`ImmutableMemory`: Shallow copy.
        """
        cdef:
            Memory_* memory_ = Memory_Copy(self._)
            Memory memory = Memory_AsObject(memory_)

        return memory

    def __dealloc__(self):
        r"""Cython deallocation method."""
        self._ = Memory_Free(self._)

    def __deepcopy__(
        self: Memory,
    ) -> ImmutableMemory:
        r"""Creates a deep copy.

        Returns:
            :obj:`ImmutableMemory`: Deep copy.
        """
        cdef:
            Memory_* memory_ = Memory_Copy(self._)
            Memory memory = Memory_AsObject(memory_)

        return memory

    def __delitem__(
        self: Memory,
        key: Union[Address, slice],
    ) -> None:
        r"""Deletes data.

        Arguments:
            key (slice or int):
                Deletion range or address.

        Note:
            This method is typically not optimized for a :class:`slice` where
            its `step` is an integer greater than 1.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | y | z]|   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> del memory[4:9]
            >>> memory.to_blocks()
            [[1, b'ABCyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | D]|   |[$]|   |[x | z]|   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | D]|   |[$]|   |[x | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | D]|   |   |[x]|   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> del memory[9]
            >>> memory.to_blocks()
            [[1, b'ABCD'], [6, b'$'], [8, b'xz']]
            >>> del memory[3]
            >>> memory.to_blocks()
            [[1, b'ABD'], [5, b'$'], [7, b'xz']]
            >>> del memory[2:10:3]
            >>> memory.to_blocks()
            [[1, b'AD'], [5, b'x']]
        """

        Memory_DelItem(self._, key)

    def __eq__(
        self: Memory,
        other: Any,
    ) -> bool:
        r"""Equality comparison.

        Arguments:
            other (Memory):
                Data to compare with `self`.

                If it is a :obj:`ImmutableMemory`, all of its blocks must match.

                If it is a :obj:`bytes`, a :obj:`bytearray`, or a
                :obj:`memoryview`, it is expected to match the first and only
                stored block.

                Otherwise, it must match the first and only stored block,
                via iteration over the stored values.

        Returns:
            bool: `self` is equal to `other`.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> data = b'Hello, World!'
            >>> memory = Memory.from_bytes(data)
            >>> memory == data
            True
            >>> memory.shift(1)
            >>> memory == data
            True

            >>> data = b'Hello, World!'
            >>> memory = Memory.from_bytes(data)
            >>> memory == list(data)
            True
            >>> memory.shift(1)
            >>> memory == list(data)
            True
        """

        return Memory_Eq(self._, other)

    def __getitem__(
        self: Memory,
        key: Union[Address, slice],
    ) -> Any:
        r"""Gets data.

        Arguments:
            key (slice or int):
                Selection range or address.
                If it is a :obj:`slice` with bytes-like `step`, the latter is
                interpreted as the filling pattern.

        Returns:
            items: Items from the requested range.

        Note:
            This method is typically not optimized for a :class:`slice` where
            its `step` is an integer greater than 1.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|
            +---+---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67| 68|   | 36|   |120|121|122|
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory[9]  # -> ord('y') = 121
            121
            >>> memory[:3].to_blocks()
            [[1, b'AB']]
            >>> memory[3:10].to_blocks()
            [[3, b'CD'], [6, b'$'], [8, b'xy']]
            >>> bytes(memory[3:10:b'.'])
            b'CD.$.xy'
            >>> memory[memory.endex]
            None
            >>> bytes(memory[3:10:3])
            b'C$y'
            >>> memory[3:10:2].to_blocks()
            [[3, b'C'], [6, b'y']]
            >>> bytes(memory[3:10:2])
            Traceback (most recent call last):
                ...
            ValueError: non-contiguous data within range
        """

        return Memory_GetItem(self._, key)

    def __iadd__(
        self: Memory,
        value: Union[AnyBytes, ImmutableMemory],
    ) -> 'MutableMemory':

        Memory_IAdd(self._, value)
        return self

    def __imul__(
        self: Memory,
        times: int,
    ) -> 'MutableMemory':
        cdef:
            addr_t times_ = 0 if times < 0 else <addr_t>times

        Memory_IMul(self._, times_)
        return self

    def __init__(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ):

        self._ = Memory_Create(start, endex)

    def __iter__(
        self: Memory,
    ) -> Iterator[Optional[Value]]:
        r"""Iterates over values.

        Iterates over values between :attr:`start` and :attr:`endex`.

        Yields:
            int: Value as byte integer, or ``None``.
        """

        yield from self.values()

    def __len__(
        self: Memory,
    ) -> Address:
        r"""Actual length.

        Computes the actual length of the stored items, i.e.
        (:attr:`endex` - :attr:`start`).
        This will consider any trimmings being active.

        Returns:
            int: Memory length.
        """

        return Memory_Length(self._)

    def __mul__(
        self: Memory,
        times: int,
    ) -> Memory:
        cdef:
            addr_t times_ = 0 if times < 0 else <addr_t>times
            Memory_* memory_ = Memory_Mul(self._, times_)
            Memory memory = Memory_AsObject(memory_)

        return memory

    def __repr__(
        self: Memory,
    ) -> str:

        return f'<{type(self).__name__}[0x{self.start:X}:0x{self.endex:X}]@0x{id(self):X}>'

    def __reversed__(
        self: Memory,
    ) -> Iterator[Optional[Value]]:
        r"""Iterates over values, reversed order.

        Iterates over values between :attr:`start` and :attr:`endex`, in
        reversed order.

        Yields:
            int: Value as byte integer, or ``None``.
        """

        yield from self.rvalues()

    def __setitem__(
        self: Memory,
        key: Union[Address, slice],
        value: Optional[Union[AnyBytes, Value, ImmutableMemory]],
    ) -> None:
        r"""Sets data.

        Arguments:
            key (slice or int):
                Selection range or address.

            value (items):
                Items to write at the selection address.
                If `value` is null, the range is cleared.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |[A]|   |   |   |   |[y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |[A]|   |[C]|   |   | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |[A | 1 | C]|   |[2 | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
            >>> memory[7:10] = None
            >>> memory.to_blocks()
            [[5, b'AB'], [10, b'yz']]
            >>> memory[7] = b'C'
            >>> memory[9] = b'x'
            >>> memory.to_blocks() == [[5, b'ABC'], [9, b'xyz']]
            True
            >>> memory[6:12:3] = None
            >>> memory.to_blocks()
            [[5, b'A'], [7, b'C'], [10, b'yz']]
            >>> memory[6:13:3] = b'123'
            >>> memory.to_blocks()
            [[5, b'A1C'], [9, b'2yz3']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |   |   |   |   |[A | B | C]|   |[x | y | z]|
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |[$]|   |[A | B | C]|   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |[$]|   |[A | B | 4 | 5 | 6 | 7 | 8 | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |[$]|   |[A | B | 4 | 5 | < | > | 8 | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
            >>> memory[0:4] = b'$'
            >>> memory.to_blocks()
            [[0, b'$'], [2, b'ABC'], [6, b'xyz']]
            >>> memory[4:7] = b'45678'
            >>> memory.to_blocks()
            [[0, b'$'], [2, b'AB45678yz']]
            >>> memory[6:8] = b'<>'
            >>> memory.to_blocks()
            [[0, b'$'], [2, b'AB45<>8yz']]
        """

        Memory_SetItem(self._, key, value)

    def __sizeof__(
        self: Memory,
    ) -> int:
        r"""Allocated byte size.

        Computes the total allocated size for the instance and its blocks.

        Returns:
            int: Allocated byte size.
        """

        return sizeof(Memory) + Memory_Sizeof(self._)

    def __str__(
        self: Memory,
    ) -> str:
        r"""String representation.

        If :attr:`content_size` is lesser than ``STR_MAX_CONTENT_SIZE``, then
        the memory is represented as a list of blocks.

        If exceeding, it is equivalent to :meth:`__repr__`.

        Returns:
            str: String representation.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [7, b'xyz']])
            >>> str(memory)
            <[[1, b'ABC'], [7, b'xyz']]>
        """
        cdef:
            const Memory_* memory = self._
            addr_t size = Memory_ContentSize(memory)
            addr_t start
            addr_t endex
            const Rack_* blocks
            size_t block_index
            const Block_* block
            list inner_list

        if size < STR_MAX_CONTENT_SIZE:
            trim_start = f'{memory.trim_start}, ' if memory.trim_start_ else ''
            trim_endex = f', {memory.trim_endex}' if memory.trim_endex_ else ''

            inner_list = []
            blocks = memory.blocks

            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                inner_list.append(f'[{Block_Start(block)}, {Block_Bytes(block)!r}]')

            inner = ', '.join(inner_list)

            return f'<{trim_start}[{inner}]{trim_endex}>'
        else:
            return repr(self)

    def _block_index_at(
        self: Memory,
        address: Address,
    ) -> Optional[BlockIndex]:
        r"""Locates the block enclosing an address.

        Returns the index of the block enclosing the given address.

        Arguments:
            address (int):
                Address of the target item.

        Returns:
            int: Block index if found, ``None`` otherwise.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   | 0 | 0 | 0 | 0 |   | 1 |   | 2 | 2 | 2 |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> [memory._block_index_at(i) for i in range(12)]
            [None, 0, 0, 0, 0, None, 1, None, 2, 2, 2, None]
        """
        cdef:
            ssize_t block_index

        block_index = Rack_IndexAt(self._.blocks, address)
        return None if block_index < 0 else block_index

    def _block_index_endex(
        self: Memory,
        address: Address,
    ) -> BlockIndex:
        r"""Locates the last block before an address range.

        Returns the index of the last block whose end address is lesser than or
        equal to `address`.

        Useful to find the termination block index in a ranged search.

        Arguments:
            address (int):
                Exclusive end address of the scanned range.

        Returns:
            int: First block index before `address`.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> [memory._block_index_endex(i) for i in range(12)]
            [0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
        """

        return Rack_IndexEndex(self._.blocks, address)

    def _block_index_start(
        self: Memory,
        address: Address,
    ) -> BlockIndex:
        r"""Locates the first block inside of an address range.

        Returns the index of the first block whose start address is greater than
        or equal to `address`.

        Useful to find the initial block index in a ranged search.

        Arguments:
            address (int):
                Inclusive start address of the scanned range.

        Returns:
            int: First block index since `address`.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 2 | 2 | 2 | 3 |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> [memory._block_index_start(i) for i in range(12)]
            [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
        """

        return Rack_IndexStart(self._.blocks, address)

    def _pretrim_endex(
        self: Memory,
        start_min: Optional[Address],
        size: Address,
    ) -> None:
        r"""Trims final data.

        Low-level method to manage trimming of data starting from an address.

        Arguments:
            start_min (int):
                Starting address of the erasure range.
                If ``None``, :attr:`trim_endex` minus `size` is considered.

            size (int):
                Size of the erasure range.

        See Also:
            :meth:`_pretrim_endex_backup`
        """

        Memory_PretrimEndex(self._, start_min, size)

    def _pretrim_endex_backup(
        self: Memory,
        start_min: Optional[Address],
        size: Address,
    ) -> ImmutableMemory:
        r"""Backups a `_pretrim_endex()` operation.

        Arguments:
            start_min (int):
                Starting address of the erasure range.
                If ``None``, :attr:`trim_endex` minus `size` is considered.

            size (int):
                Size of the erasure range.

        Returns:
            :obj:`ImmutableMemory`: Backup memory region.

        See Also:
            :meth:`_pretrim_endex`
        """
        cdef:
            addr_t start_min_
            addr_t size_ = <addr_t>size
            const Memory_* memory = self._
            addr_t start

        if memory.trim_endex_ and size_ > 0:
            start = memory.trim_endex
            CheckSubAddrU(start, size)
            start -= size
            if start_min is not None:
                start_min_ = <addr_t>start_min
                if start < start_min_:
                    start = start_min_
            return Memory_Extract_(memory, start, Memory_Endex(memory), 0, NULL, 1, True)
        else:
            return Memory()

    def _pretrim_start(
        self: Memory,
        endex_max: Optional[Address],
        size: Address,
    ) -> None:
        r"""Trims initial data.

        Low-level method to manage trimming of data starting from an address.

        Arguments:
            endex_max (int):
                Exclusive end address of the erasure range.
                If ``None``, :attr:`trim_start` plus `size` is considered.

            size (int):
                Size of the erasure range.

        See Also:
            :meth:`_pretrim_start_backup`
        """

        Memory_PretrimStart(self._, endex_max, size)

    def _pretrim_start_backup(
        self: Memory,
        endex_max: Optional[Address],
        size: Address,
    ) -> ImmutableMemory:
        r"""Backups a `_pretrim_start()` operation.

        Arguments:
            endex_max (int):
                Exclusive end address of the erasure range.
                If ``None``, :attr:`trim_start` plus `size` is considered.

            size (int):
                Size of the erasure range.

        Returns:
            :obj:`ImmutableMemory`: Backup memory region.

        See Also:
            :meth:`_pretrim_start`
        """
        cdef:
            addr_t endex_max_
            addr_t size_ = <addr_t>size
            const Memory_* memory = self._
            addr_t endex

        if memory.trim_start_ and size_ > 0:
            endex = memory.trim_start
            CheckAddAddrU(endex, size)
            endex += size
            if endex_max is not None:
                endex_max_ = <addr_t>endex_max
                if endex > endex_max_:
                    endex = endex_max_
            return Memory_Extract_(memory, Memory_Start(memory), endex, 0, NULL, 1, True)
        else:
            return Memory()

    def append(
        self: Memory,
        item: Union[AnyBytes, Value],
    ) -> None:
        r"""Appends a single item.

        Arguments:
            item (int):
                Value to append. Can be a single byte string or integer.

        See Also:
            :meth:`append_backup`
            :meth:`append_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory()
            >>> memory.append(b'$')
            >>> memory.to_blocks()
            [[0, b'$']]

            ~~~

            >>> memory = Memory()
            >>> memory.append(3)
            >>> memory.to_blocks()
            [[0, b'\x03']]
        """

        return Memory_Append(self._, item)

    # noinspection PyMethodMayBeStatic
    def append_backup(
        self: Memory,
    ) -> None:
        r"""Backups an `append()` operation.

        Returns:
            None: Nothing.

        See Also:
            :meth:`append`
            :meth:`append_restore`
        """

        return None

    def append_restore(
        self: Memory,
    ) -> None:
        r"""Restores an `append()` operation.

        See Also:
            :meth:`append`
            :meth:`append_backup`
        """

        Memory_PopLast_(self._)

    def block_span(
        self: Memory,
        address: Address,
    ) -> Tuple[Optional[Address], Optional[Address], Optional[Value]]:
        r"""Span of block data.

        It searches for the biggest chunk of data adjacent to the given
        address.

        If the address is within a gap, its bounds are returned, and its
        value is ``None``.

        If the address is before or after any data, bounds are ``None``.

        Arguments:
            address (int):
                Reference address.

        Returns:
            tuple: Start bound, exclusive end bound, and reference value.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory()
            >>> memory.block_span(0)
            (None, None, None)

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |[A | B | B | B | C]|   |   |[C | C | D]|   |
            +---+---+---+---+---+---+---+---+---+---+---+
            | 65| 66| 66| 66| 67|   |   | 67| 67| 68|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[0, b'ABBBC'], [7, b'CCD']])
            >>> memory.block_span(2)
            (0, 5, 66)
            >>> memory.block_span(4)
            (0, 5, 67)
            >>> memory.block_span(5)
            (5, 7, None)
            >>> memory.block_span(10)
            (10, None, None)
        """
        cdef:
            addr_t address_ = <addr_t>address
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            const Block_* block
            addr_t block_start
            addr_t block_endex
            byte_t value

        block_index = Rack_IndexStart(blocks, address_)

        if block_index < block_count:
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)

            if block_start <= address_ < block_endex:
                # Address within a block
                CheckSubAddrU(address_, block_start)
                CheckAddrToSizeU(address_ - block_start)
                value = Block_Get__(block, <size_t>(address_ - block_start))
                return block_start, block_endex, value  # block span

            elif block_index:
                # Address within a gap
                block_endex = block_start  # end gap before next block
                block = Rack_Get__(blocks, block_index - 1)
                block_start = Block_Endex(block)  # start gap after previous block
                return block_start, block_endex, None  # gap span

            else:
                # Address before content
                return None, block_start, None  # open left

        else:
            # Address after content
            if block_count:
                block = Rack_Last__(blocks)
                block_start = Block_Start(block)
                block_endex = Block_Endex(block)
                return block_endex, None, None  # open right

            else:
                return None, None, None  # fully open
    def blocks(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Tuple[Address, memoryview]]:
        r"""Iterates over blocks.

        Iterates over data blocks within an address range.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Yields:
            (start, memoryview): Start and data view of each block/slice.

        See Also:
            :meth:`intervals`
            :meth:`to_blocks`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> [[s, bytes(d)] for s, d in memory.blocks()]
            [[1, b'AB'], [5, b'x'], [7, b'123']]
            >>> [[s, bytes(d)] for s, d in memory.blocks(2, 9)]
            [[2, b'B'], [5, b'x'], [7, b'12']]
            >>> [[s, bytes(d)] for s, d in memory.blocks(3, 5)]
            []
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            addr_t slice_start
            addr_t slice_endex
            BlockView slice_view

        if block_count:
            if start is None and endex is None:  # faster
                for block_index in range(block_count):
                    block = Rack_Get__(blocks, block_index)
                    yield Block_Start(block), Block_View(block)
            else:
                block_index_start = 0 if start is None else Rack_IndexStart(blocks, start)
                block_index_endex = block_count if endex is None else Rack_IndexEndex(blocks, endex)
                start_, endex_ = Memory_Bound(self._, start, endex)

                for block_index in range(block_index_start, block_index_endex):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    block_endex = Block_Endex(block)
                    slice_start = block_start if start_ < block_start else start_
                    slice_endex = endex_ if endex_ < block_endex else block_endex
                    if slice_start < slice_endex:
                        slice_start -= block_start
                        slice_endex -= block_start
                        slice_view = Block_ViewSlice_(block, <size_t>slice_start, <size_t>slice_endex)
                        yield (block_start + slice_start), slice_view

    def bound(
        self: Memory,
        start: Optional[Address],
        endex: Optional[Address],
    ) -> ClosedInterval:
        r"""Bounds addresses.

        It bounds the given addresses to stay within memory limits.
        ``None`` is used to ignore a limit for the `start` or `endex`
        directions.

        In case of stored data, :attr:`content_start` and
        :attr:`content_endex` are used as bounds.

        In case of trimming limits, :attr:`trim_start` or :attr:`trim_endex`
        are used as bounds, when not ``None``.

        In case `start` and `endex` are in the wrong order, one clamps
        the other if present (see the Python implementation for details).

        Returns:
            tuple of int: Bounded `start` and `endex`, closed interval.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().bound(None, None)
            (0, 0)
            >>> Memory().bound(None, 100)
            (0, 100)

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.bound(0, 30)
            (0, 30)
            >>> memory.bound(2, 6)
            (2, 6)
            >>> memory.bound(None, 6)
            (1, 6)
            >>> memory.bound(2, None)
            (2, 8)

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[[[|   |[A | B | C]|   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[3, b'ABC']], start=1, endex=8)
            >>> memory.bound(None, None)
            (1, 8)
            >>> memory.bound(0, 30)
            (1, 8)
            >>> memory.bound(2, 6)
            (2, 6)
            >>> memory.bound(2, None)
            (2, 8)
            >>> memory.bound(None, 6)
            (1, 6)
        """

        return Memory_Bound(self._, start, endex)

    def clear(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:
        r"""Clears an address range.

        Arguments:
            start (int):
                Inclusive start address for clearing.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for clearing.
                If ``None``, :attr:`endex` is considered.

        See Also:
            :meth:`clear_backup`
            :meth:`clear_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |[A]|   |   |   |   |[y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
            >>> memory.clear(6, 10)
            >>> memory.to_blocks()
            [[5, b'A'], [10, b'yz']]
        """

        Memory_Clear(self._, start, endex)

    def clear_backup(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:
        r"""Backups a `clear()` operation.

        Arguments:
            start (int):
                Inclusive start address for clearing.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for clearing.
                If ``None``, :attr:`endex` is considered.

        Returns:
            :obj:`ImmutableMemory`: Backup memory region.

        See Also:
            :meth:`clear`
            :meth:`clear_restore`
        """
        cdef:
            const Memory_* memory = self._
            addr_t start_ = Memory_Start(memory) if start is None else <addr_t>start
            addr_t endex_ = Memory_Endex(memory) if endex is None else <addr_t>endex

        return Memory_Extract_(memory, start_, endex_, 0, NULL, 1, True)

    def clear_restore(
        self: Memory,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores a `clear()` operation.

        Arguments:
            backup (:obj:`ImmutableMemory`):
                Backup memory region to restore.

        See Also:
            :meth:`clear`
            :meth:`clear_backup`
        """

        Memory_Write(self._, 0, backup, True)

    def content_blocks(
        self,
        block_index_start: Optional[BlockIndex] = None,
        block_index_endex: Optional[BlockIndex] = None,
        block_index_step: Optional[BlockIndex] = None,
    ) -> Iterator[Union[Tuple[Address, Union[memoryview, bytearray]], Block]]:
        r"""Iterates over blocks.

        Iterates over data blocks within a block index range.

        Arguments:
            block_index_start (int):
                Inclusive block start index.
                A negative index is referred to :attr:`content_parts`.
                If ``None``, ``0`` is considered.

            block_index_endex (int):
                Exclusive block end index.
                A negative index is referred to :attr:`content_parts`.
                If ``None``, :attr:`content_parts` is considered.

            block_index_step (int):
                Block index step, which can be negative.
                If ``None``, ``1`` is considered.

        Yields:
            (start, memoryview): Start and data view of each block/slice.

        See Also:
            :attr:`content_parts`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> [[s, bytes(d)] for s, d in memory.content_blocks()]
            [[1, b'AB'], [5, b'x'], [7, b'123']]
            >>> [[s, bytes(d)] for s, d in memory.content_blocks(1, 2)]
            [[5, b'x']]
            >>> [[s, bytes(d)] for s, d in memory.content_blocks(3, 5)]
            []
            >>> [[s, bytes(d)] for s, d in memory.content_blocks(block_index_start=-2)]
            [[5, b'x'], [7, b'123']]
            >>> [[s, bytes(d)] for s, d in memory.content_blocks(block_index_endex=-1)]
            [[1, b'AB'], [5, b'x']]
            >>> [[s, bytes(d)] for s, d in memory.content_blocks(block_index_step=2)]
            [[1, b'AB'], [7, b'123']]
        """
        cdef:
            const Rack_* blocks = self._.blocks
            ssize_t block_count = <ssize_t>Rack_Length(blocks)
            ssize_t block_index
            ssize_t block_index_start_
            ssize_t block_index_endex_
            ssize_t block_index_step_
            const Block_* block

        if block_count:
            if block_index_start is None:
                block_index_start_ = 0
            else:
                if block_index_start < 0:
                    block_index_start += block_count
                    if block_index_start < 0:
                        block_index_start = 0
                block_index_start_ = <ssize_t>block_index_start
                if block_index_start_ > block_count:
                    block_index_start_ = block_count

            if block_index_endex is None:
                block_index_endex_ = block_count
            else:
                if block_index_endex < 0:
                    block_index_endex += block_count
                    if block_index_endex < 0:
                        block_index_endex = 0
                block_index_endex_ = <ssize_t>block_index_endex
                if block_index_endex_ > block_count:
                    block_index_endex_ = block_count

            if block_index_step is None:
                block_index_step_ = 1
            else:
                block_index_step_ = <ssize_t>block_index_step

            for block_index in range(block_index_start_, block_index_endex_, block_index_step_):
                block = Rack_Get__(blocks, <size_t>block_index)
                yield Block_Start(block), Block_View(block)

    @property
    def content_endex(
        self: Memory,
    ) -> Address:
        r"""int: Exclusive content end address.

        This property holds the exclusive end address of the memory content.
        By default, it is the current maximmum exclusive end address of
        the last stored block.

        If the memory has no data and no trimming, :attr:`start` is returned.

        Trimming is considered only for an empty memory.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().content_endex
            0
            >>> Memory(endex=8).content_endex
            0
            >>> Memory(start=1, endex=8).content_endex
            1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_endex
            8

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC']], endex=8)
            >>> memory.content_endex
            4
        """

        return Memory_ContentEndex(self._)

    @property
    def content_endin(
        self: Memory,
    ) -> Address:
        r"""int: Inclusive content end address.

        This property holds the inclusive end address of the memory content.
        By default, it is the current maximmum inclusive end address of
        the last stored block.

        If the memory has no data and no trimming, :attr:`start` minus one is
        returned.

        Trimming is considered only for an empty memory.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().content_endin
            -1
            >>> Memory(endex=8).content_endin
            -1
            >>> Memory(start=1, endex=8).content_endin
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_endin
            7

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC']], endex=8)
            >>> memory.content_endin
            3
        """

        return Memory_ContentEndin(self._)

    def content_items(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Tuple[Address, Value]]:
        r"""Iterates over content address and value pairs.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Yields:
            int: Content address and value pairs.

        See Also:
            meth:`content_keys`
            meth:`content_values`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> dict(memory.content_items())
            {1: 65, 2: 66, 5: 120, 7: 49, 8: 50, 9: 51}
            >>> dict(memory.content_items(2, 9))
            {2: 66, 5: 120, 7: 49, 8: 50}
            >>> dict(memory.content_items(3, 5))
            {}
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            addr_t slice_start
            addr_t slice_endex
            addr_t address
            size_t offset

        if block_count:
            if start is None and endex is None:  # faster
                for block_index in range(block_count):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    for offset in range(Block_Length(block)):
                        yield (block_start + offset), Block_Get__(block, offset)
            else:
                block_index_start = 0 if start is None else Rack_IndexStart(blocks, start)
                block_index_endex = block_count if endex is None else Rack_IndexEndex(blocks, endex)
                start_, endex_ = Memory_Bound(self._, start, endex)

                for block_index in range(block_index_start, block_index_endex):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    block_endex = Block_Endex(block)
                    slice_start = block_start if start_ < block_start else start_
                    slice_endex = endex_ if endex_ < block_endex else block_endex
                    for address in range(slice_start, slice_endex):
                        offset = <size_t>(address - block_start)
                        yield address, Block_Get__(block, offset)

    def content_keys(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Address]:
        r"""Iterates over content addresses.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Yields:
            int: Content addresses.

        See Also:
            meth:`content_items`
            meth:`content_values`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> list(memory.content_keys())
            [1, 2, 5, 7, 8, 9]
            >>> list(memory.content_keys(2, 9))
            [2, 5, 7, 8]
            >>> list(memory.content_keys(3, 5))
            []
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            addr_t slice_start
            addr_t slice_endex
            addr_t address

        if block_count:
            if start is None and endex is None:  # faster
                for block_index in range(block_count):
                    block = Rack_Get__(blocks, block_index)
                    for address in range(Block_Start(block), Block_Endex(block)):
                        yield address
            else:
                block_index_start = 0 if start is None else Rack_IndexStart(blocks, start)
                block_index_endex = block_count if endex is None else Rack_IndexEndex(blocks, endex)
                start_, endex_ = Memory_Bound(self._, start, endex)

                for block_index in range(block_index_start, block_index_endex):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    block_endex = Block_Endex(block)
                    slice_start = block_start if start_ < block_start else start_
                    slice_endex = endex_ if endex_ < block_endex else block_endex
                    for address in range(slice_start, slice_endex):
                        yield address

    @property
    def content_parts(
        self: Memory,
    ) -> int:
        r"""Number of blocks.

        Returns:
            int: The number of blocks.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().content_parts
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_parts
            2

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC']], endex=8)
            >>> memory.content_parts
            1
        """

        return Memory_ContentParts(self._)

    @property
    def content_size(
        self: Memory,
    ) -> Address:
        r"""Actual content size.

        Returns:
            int: The sum of all block lengths.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().content_size
            0
            >>> Memory(start=1, endex=8).content_size
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_size
            6

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC']], endex=8)
            >>> memory.content_size
            3
        """

        return Memory_ContentSize(self._)

    @property
    def content_span(
        self: Memory,
    ) -> ClosedInterval:
        r"""tuple of int: Memory content address span.

        A :attr:`tuple` holding both :attr:`content_start` and
        :attr:`content_endex`.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().content_span
            (0, 0)
            >>> Memory(start=1).content_span
            (1, 1)
            >>> Memory(endex=8).content_span
            (0, 0)
            >>> Memory(start=1, endex=8).content_span
            (1, 1)

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_span
            (1, 8)
        """

        return Memory_ContentSpan(self._)

    @property
    def content_start(
        self: Memory,
    ) -> Address:
        r"""int: Inclusive content start address.

        This property holds the inclusive start address of the memory content.
        By default, it is the current minimum inclusive start address of
        the first stored block.

        If the memory has no data and no trimming, 0 is returned.

        Trimming is considered only for an empty memory.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().content_start
            0
            >>> Memory(start=1).content_start
            1
            >>> Memory(start=1, endex=8).content_start
            1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_start
            1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[[[|   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'xyz']], start=1)
            >>> memory.content_start
            5
        """

        return Memory_ContentStart(self._)

    def content_values(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Value]:
        r"""Iterates over content values.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Yields:
            int: Content values.

        See Also:
            meth:`content_items`
            meth:`content_keys`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> list(memory.content_values())
            [65, 66, 120, 49, 50, 51]
            >>> list(memory.content_values(2, 9))
            [66, 120, 49, 50]
            >>> list(memory.content_values(3, 5))
            []
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            addr_t slice_start
            addr_t slice_endex
            addr_t address
            size_t offset

        if block_count:
            if start is None and endex is None:  # faster
                for block_index in range(block_count):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    for offset in range(Block_Length(block)):
                        yield Block_Get__(block, offset)
            else:
                block_index_start = 0 if start is None else Rack_IndexStart(blocks, start)
                block_index_endex = block_count if endex is None else Rack_IndexEndex(blocks, endex)
                start_, endex_ = Memory_Bound(self._, start, endex)

                for block_index in range(block_index_start, block_index_endex):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    block_endex = Block_Endex(block)
                    slice_start = block_start if start_ < block_start else start_
                    slice_endex = endex_ if endex_ < block_endex else block_endex
                    for address in range(slice_start, slice_endex):
                        offset = <size_t>(address - block_start)
                        yield Block_Get__(block, offset)

    @property
    def contiguous(
        self: Memory,
    ) -> bool:
        r"""bool: Contains contiguous data.

        The memory is considered to have contiguous data if there is no empty
        space between blocks.

        If trimming is defined, there must be no empty space also towards it.
        """

        return Memory_Contiguous(self._)

    def copy(
        self: Memory,
    ) -> ImmutableMemory:
        r"""Creates a deep copy.

        Returns:
            :obj:`ImmutableMemory`: Deep copy.
        """

        return self.__deepcopy__()

    def count(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> int:
        r"""Counts items.

        Arguments:
            item (items):
                Reference value to count.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The number of items equal to `value`.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[B | a | t]|   |[t | a | b]|
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'Bat'], [9, b'tab']])
            >>> memory.count(b'a')
            2
        """

        return Memory_Count(self._, item, start, endex)

    def crop(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:
        r"""Keeps data within an address range.

        Arguments:
            start (int):
                Inclusive start address for cropping.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for cropping.
                If ``None``, :attr:`endex` is considered.

        See Also:
            :meth:`crop_backup`
            :meth:`crop_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |   |[B | C]|   |[x]|   |   |   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
            >>> memory.crop(6, 10)
            >>> memory.to_blocks()
            [[6, b'BC'], [9, b'x']]
        """

        Memory_Crop(self._, start, endex)

    def crop_backup(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Tuple[Optional[ImmutableMemory], Optional[ImmutableMemory]]:
        r"""Backups a `crop()` operation.

        Arguments:
            start (int):
                Inclusive start address for cropping.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for cropping.
                If ``None``, :attr:`endex` is considered.

        Returns:
            :obj:`ImmutableMemory` pair: Backup memory regions.

        See Also:
            :meth:`crop`
            :meth:`crop_restore`
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Memory_* memory = self._
            const Rack_* blocks = memory.blocks
            addr_t block_start
            addr_t block_endex
            Memory backup_start = None
            Memory backup_endex = None

        if Rack_Bool(blocks):
            if start is not None:
                start_ = <addr_t>start
                block_start = Block_Start(Rack_First__(blocks))
                if block_start < start_:
                    backup_start = Memory_Extract_(memory, block_start, start_, 0, NULL, 1, True)

            if endex is not None:
                endex_ = <addr_t>endex
                block_endex = Block_Endex(Rack_Last__(blocks))
                if endex_ < block_endex:
                    backup_endex = Memory_Extract_(memory, endex_, block_endex, 0, NULL, 1, True)

        return backup_start, backup_endex

    def crop_restore(
        self: Memory,
        backup_start: Optional[ImmutableMemory],
        backup_endex: Optional[ImmutableMemory],
    ) -> None:
        r"""Restores a `crop()` operation.

        Arguments:
            backup_start (:obj:`ImmutableMemory`):
                Backup memory region to restore at the beginning.

            backup_endex (:obj:`ImmutableMemory`):
                Backup memory region to restore at the end.

        See Also:
            :meth:`crop`
            :meth:`crop_backup`
        """

        if backup_start is not None:
            Memory_Write(self._, 0, backup_start, True)
        if backup_endex is not None:
            Memory_Write(self._, 0, backup_endex, True)

    def cut(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        bound: bool = True,
    ) -> ImmutableMemory:
        r"""Cuts a slice of memory.

        Arguments:
            start (int):
                Inclusive start address for cutting.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for cutting.
                If ``None``, :attr:`endex` is considered.

            bound (bool):
                The selected address range is applied to the resulting memory
                as its trimming range. This retains information about any
                initial and final emptiness of that range, which would be lost
                otherwise.

        Returns:
            :obj:`Memory`: A copy of the memory from the selected range.
        """

        return Memory_Cut(self._, start, endex, bound)

    def delete(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:
        r"""Deletes an address range.

        Arguments:
            start (int):
                Inclusive start address for deletion.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for deletion.
                If ``None``, :attr:`endex` is considered.

        See Also:
            :meth:`delete_backup`
            :meth:`delete_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12| 13|
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | y | z]|   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
            >>> memory.delete(6, 10)
            >>> memory.to_blocks()
            [[5, b'Ayz']]
        """

        Memory_Delete(self._, start, endex)

    def delete_backup(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:
        r"""Backups a `delete()` operation.

        Arguments:
            start (int):
                Inclusive start address for deletion.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for deletion.
                If ``None``, :attr:`endex` is considered.

        Returns:
            :obj:`ImmutableMemory`: Backup memory region.

        See Also:
            :meth:`delete`
            :meth:`delete_restore`
        """
        cdef:
            const Memory_* memory = self._
            addr_t start_ = Memory_Start(memory) if start is None else <addr_t>start
            addr_t endex_ = Memory_Endex(memory) if endex is None else <addr_t>endex

        return Memory_Extract_(memory, start_, endex_, 0, NULL, 1, True)

    def delete_restore(
        self: Memory,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores a `delete()` operation.

        Arguments:
            backup (:obj:`ImmutableMemory`):
                Backup memory region

        See Also:
            :meth:`delete`
            :meth:`delete_backup`
        """
        cdef:
            Memory_* memory = self._
            const Memory_* backup_

        if isinstance(backup, Memory):
            backup_ = (<Memory>backup)._
            Memory_Reserve_(memory, Memory_Start(backup_), Memory_Length(backup_))
            Memory_WriteSame_(memory, 0, backup_, True)
        else:
            Memory_Reserve(memory, backup.start, len(backup))
            Memory_Write(memory, 0, backup, True)

    @property
    def endex(
        self: Memory,
    ) -> Address:
        r"""int: Exclusive end address.

        This property holds the exclusive end address of the virtual space.
        By default, it is the current maximmum exclusive end address of
        the last stored block.

        If  :attr:`trim_endex` not ``None``, that is returned.

        If the memory has no data and no trimming, :attr:`start` is returned.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().endex
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.endex
            8

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC']], endex=8)
            >>> memory.endex
            8
        """

        return Memory_Endex(self._)

    @property
    def endin(
        self: Memory,
    ) -> Address:
        r"""int: Inclusive end address.

        This property holds the inclusive end address of the virtual space.
        By default, it is the current maximmum inclusive end address of
        the last stored block.

        If  :attr:`trim_endex` not ``None``, that minus one is returned.

        If the memory has no data and no trimming, :attr:`start` is returned.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().endin
            -1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.endin
            7

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC']], endex=8)
            >>> memory.endin
            7
        """

        return Memory_Endin(self._)

    def equal_span(
        self: Memory,
        address: Address,
    ) -> Tuple[Optional[Address], Optional[Address], Optional[Value]]:
        r"""Span of homogeneous data.

        It searches for the biggest chunk of data adjacent to the given
        address, with the same value at that address.

        If the address is within a gap, its bounds are returned, and its
        value is ``None``.

        If the address is before or after any data, bounds are ``None``.

        Arguments:
            address (int):
                Reference address.

        Returns:
            tuple: Start bound, exclusive end bound, and reference value.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory()
            >>> memory.equal_span(0)
            (None, None, None)

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |[A | B | B | B | C]|   |   |[C | C | D]|   |
            +---+---+---+---+---+---+---+---+---+---+---+
            | 65| 66| 66| 66| 67|   |   | 67| 67| 68|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[0, b'ABBBC'], [7, b'CCD']])
            >>> memory.equal_span(2)
            (1, 4, 66)
            >>> memory.equal_span(4)
            (4, 5, 67)
            >>> memory.equal_span(5)
            (5, 7, None)
            >>> memory.equal_span(10)
            (10, None, None)
        """
        cdef:
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            addr_t address_ = <addr_t>address
            addr_t start
            addr_t endex
            size_t offset
            byte_t value

        block_index = Rack_IndexStart(blocks, address_)

        if block_index < block_count:
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)

            if block_start <= address_ < block_endex:
                # Address within a block
                CheckSubAddrU(address_, block_start)
                CheckAddrToSizeU(address - block_start)
                offset = <size_t>(address_ - block_start)
                start = offset
                CheckAddAddrU(offset, 1)
                endex = offset + 1
                value = Block_Get__(block, offset)

                for start in range(start + 1, 0, -1):
                    if Block_Get__(block, start - 1) != value:
                        break
                else:
                    start = 0

                for endex in range(endex, Block_Length(block)):
                    if Block_Get__(block, endex) != value:
                        break
                else:
                    endex = Block_Length(block)

                block_endex = block_start + endex
                block_start = block_start + start
                return block_start, block_endex, value  # equal data span

            elif block_index:
                # Address within a gap
                block_endex = block_start  # end gap before next block
                block = Rack_Get__(blocks, block_index - 1)
                block_start = Block_Endex(block)  # start gap after previous block
                return block_start, block_endex, None  # gap span

            else:
                # Address before content
                return None, block_start, None  # open left

        else:
            # Address after content
            if block_count:
                block = Rack_Last__(blocks)
                block_start = Block_Start(block)
                block_endex = Block_Endex(block)
                return block_endex, None, None  # open right

            else:
                return None, None, None  # fully open

    def extend(
        self: Memory,
        items: Union[AnyBytes, ImmutableMemory],
        offset: Address = 0,
    ) -> None:
        r"""Concatenates items.

        Equivalent to ``self += items``.

        Arguments:
            items (items):
                Items to append at the end of the current virtual space.

            offset (int):
                Optional offset w.r.t. :attr:`content_endex`.

        See Also:
            :meth:`extend_backup`
            :meth:`extend_restore`
        """

        return Memory_Extend(self._, items, offset)

    def extend_backup(
        self: Memory,
        offset: Address = 0,
    ) -> Address:
        r"""Backups an `extend()` operation.

        Arguments:
            offset (int):
                Optional offset w.r.t. :attr:`content_endex`.

        Returns:
            int: Content exclusive end address.

        See Also:
            :meth:`extend`
            :meth:`extend_restore`
        """

        if offset < 0:
            raise ValueError('negative extension offset')
        return Memory_ContentEndex(self._) + <addr_t>offset

    def extend_restore(
        self: Memory,
        content_endex: Address,
    ) -> None:
        r"""Restores an `extend()` operation.

        Arguments:
            content_endex (int):
                Content exclusive end address to restore.

        See Also:
            :meth:`extend`
            :meth:`extend_backup`
        """

        Memory_Clear_(self._, <addr_t>content_endex, ADDR_MAX)

    def extract(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
        step: Optional[Address] = None,
        bound: bool = True,
    ) -> ImmutableMemory:
        r"""Selects items from a range.

        Arguments:
            start (int):
                Inclusive start of the extracted range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the extracted range.
                If ``None``, :attr:`endex` is considered.

            pattern (items):
                Optional pattern of items to fill the emptiness.

            step (int):
                Optional address stepping between bytes extracted from the
                range. It has the same meaning of Python's :attr:`slice.step`,
                but negative steps are ignored.
                Please note that a `step` greater than 1 could take much more
                time to process than the default unitary step.

            bound (bool):
                The selected address range is applied to the resulting memory
                as its trimming range. This retains information about any
                initial and final emptiness of that range, which would be lost
                otherwise.

        Returns:
            :obj:`ImmutableMemory`: A copy of the memory from the selected range.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.extract().to_blocks()
            [[1, b'ABCD'], [6, b'$'], [8, b'xyz']]
            >>> memory.extract(2, 9).to_blocks()
            [[2, b'BCD'], [6, b'$'], [8, b'x']]
            >>> memory.extract(start=2).to_blocks()
            [[2, b'BCD'], [6, b'$'], [8, b'xyz']]
            >>> memory.extract(endex=9).to_blocks()
            [[1, b'ABCD'], [6, b'$'], [8, b'x']]
            >>> memory.extract(5, 8).span
            (5, 8)
            >>> memory.extract(pattern=b'.').to_blocks()
            [[1, b'ABCD.$.xyz']]
            >>> memory.extract(pattern=b'.', step=3).to_blocks()
            [[1, b'AD.z']]
        """

        return Memory_Extract(self._, start, endex, pattern, step, bound)

    def fill(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
    ) -> None:
        r"""Overwrites a range with a pattern.

        Arguments:
            start (int):
                Inclusive start address for filling.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for filling.
                If ``None``, :attr:`endex` is considered.

            pattern (items):
                Pattern of items to fill the range.

        See Also:
            :meth:`fill_backup`
            :meth:`fill_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[1 | 2 | 3 | 1 | 2 | 3 | 1 | 2]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> memory.fill(pattern=b'123')
            >>> memory.to_blocks()
            [[1, b'12312312']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | 1 | 2 | 3 | 1 | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> memory.fill(3, 7, b'123')
            >>> memory.to_blocks()
            [[1, b'AB1231yz']]
        """

        Memory_Fill(self._, start, endex, pattern)

    def fill_backup(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:
        r"""Backups a `fill()` operation.

        Arguments:
            start (int):
                Inclusive start address for filling.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for filling.
                If ``None``, :attr:`endex` is considered.

        Returns:
            :obj:`ImmutableMemory`: Backup memory region.

        See Also:
            :meth:`fill`
            :meth:`fill_restore`
        """
        cdef:
            const Memory_* memory = self._
            addr_t start_ = Memory_Start(memory) if start is None else <addr_t>start
            addr_t endex_ = Memory_Endex(memory) if endex is None else <addr_t>endex

        return Memory_Extract_(memory, start_, endex_, 0, NULL, 1, True)

    def fill_restore(
        self: Memory,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores a `fill()` operation.

        Arguments:
            backup (:obj:`ImmutableMemory`):
                Backup memory region to restore.

        See Also:
            :meth:`fill`
            :meth:`fill_backup`
        """

        Memory_Write(self._, 0, backup, True)

    def find(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:
        r"""Index of an item.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`endex` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The index of the first item equal to `value`, or -1.
        """

        return Memory_Find(self._, item, start, endex)

    def flood(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
    ) -> None:
        r"""Fills emptiness between non-touching blocks.

        Arguments:
            start (int):
                Inclusive start address for flooding.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for flooding.
                If ``None``, :attr:`endex` is considered.

            pattern (items):
                Pattern of items to fill the range.

        See Also:
            :meth:`flood_backup`
            :meth:`flood_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | 1 | 2 | x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> memory.flood(pattern=b'123')
            >>> memory.to_blocks()
            [[1, b'ABC12xyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | 2 | 3 | x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> memory.flood(3, 7, b'123')
            >>> memory.to_blocks()
            [[1, b'ABC23xyz']]
        """

        Memory_Flood(self._, start, endex, pattern)

    def flood_backup(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> List[OpenInterval]:
        r"""Backups a `flood()` operation.

        Arguments:
            start (int):
                Inclusive start address for filling.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for filling.
                If ``None``, :attr:`endex` is considered.

        Returns:
            list of open intervals: Backup memory gaps.

        See Also:
            :meth:`flood`
            :meth:`flood_restore`
        """

        return list(self.gaps(start, endex))

    def flood_restore(
        self: Memory,
        gaps: List[OpenInterval],
    ) -> None:
        r"""Restores a `flood()` operation.

        Arguments:
            gaps (list of open intervals):
                Backup memory gaps to restore.

        See Also:
            :meth:`flood`
            :meth:`flood_backup`
        """
        cdef:
            Memory_* memory = self._

        for gap_start, gap_endex in gaps:
            Memory_Clear(memory, gap_start, gap_endex)

    @classmethod
    def from_blocks(
        cls: Type[Memory],
        blocks: BlockList,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> ImmutableMemory:
        r"""Creates a virtual memory from blocks.

        Arguments:
            blocks (list of blocks):
                A sequence of non-overlapping blocks, sorted by address.

            offset (int):
                Some address offset applied to all the blocks.

            start (int):
                Optional memory start address.
                Anything before will be trimmed away.

            endex (int):
                Optional memory exclusive end address.
                Anything at or after it will be trimmed away.

            copy (bool):
                Forces copy of provided input data.

            validate (bool):
                Validates the resulting :obj:`ImmutableMemory` object.

        Returns:
            :obj:`ImmutableMemory`: The resulting memory object.

        Raises:
            :obj:`ValueError`: Some requirements are not satisfied.

        See Also:
            :meth:`to_blocks`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+
            |   |   |   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> blocks = [[1, b'ABC'], [5, b'xyz']]
            >>> memory = Memory.from_blocks(blocks)
            >>> memory.to_blocks()
            [[1, b'ABC'], [5, b'xyz']]
            >>> memory = Memory.from_blocks(blocks, offset=3)
            >>> memory.to_blocks()
            [[4, b'ABC'], [8, b'xyz']]

            ~~~

            >>> # Loads data from an Intel HEX record file
            >>> # NOTE: Record files typically require collapsing!
            >>> import hexrec.records as hr
            >>> blocks = hr.load_blocks('records.hex')
            >>> memory = Memory.from_blocks(collapse_blocks(blocks))
            >>> memory
                ...
        """
        cdef:
            Memory_* memory_ = Memory_FromBlocks(blocks, offset, start, endex, validate)

        return Memory_AsObject(memory_)

    @classmethod
    def from_bytes(
        cls: Type[Memory],
        data: AnyBytes,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> ImmutableMemory:
        r"""Creates a virtual memory from a byte-like chunk.

        Arguments:
            data (byte-like data):
                A byte-like chunk of data (e.g. :obj:`bytes`,
                :obj:`bytearray`, :obj:`memoryview`).

            offset (int):
                Start address of the block of data.

            start (int):
                Optional memory start address.
                Anything before will be trimmed away.

            endex (int):
                Optional memory exclusive end address.
                Anything at or after it will be trimmed away.

            copy (bool):
                Forces copy of provided input data into the underlying data
                structure.

            validate (bool):
                Validates the resulting :obj:`ImmutableMemory` object.

        Returns:
            :obj:`ImmutableMemory`: The resulting memory object.

        Raises:
            :obj:`ValueError`: Some requirements are not satisfied.

        See Also:
            :meth:`to_bytes`

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory.from_bytes(b'')
            >>> memory.to_blocks()
            []

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |   |[A | B | C | x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_bytes(b'ABCxyz', 2)
            >>> memory.to_blocks()
            [[2, b'ABCxyz']]
        """
        cdef:
            Memory_* memory_ = Memory_FromBytes(data, offset, start, endex)

        return Memory_AsObject(memory_)

    @classmethod
    def from_items(
        cls: Type[Memory],
        items: Union[AddressValueMapping,
                     Iterable[Tuple[Address, Optional[Value]]],
                     Mapping[Address, Optional[Union[Value, AnyBytes]]],
                     ImmutableMemory],
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        validate: bool = True,
    ) -> Memory:
        r"""Creates a virtual memory from a iterable address/byte mapping.

        Arguments:
            items (iterable address/byte mapping):
                An iterable mapping of address to byte values.
                Values of ``None`` are translated as gaps.
                When an address is stated multiple times, the last is kept.

            offset (int):
                An address offset applied to all the values.

            start (int):
                Optional memory start address.
                Anything before will be trimmed away.

            endex (int):
                Optional memory exclusive end address.
                Anything at or after it will be trimmed away.

            validate (bool):
                Validates the resulting :obj:`ImmutableMemory` object.

        Returns:
            :obj:`ImmutableMemory`: The resulting memory object.

        Raises:
            :obj:`ValueError`: Some requirements are not satisfied.

        See Also:
            :meth:`to_bytes`

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory.from_values({})
            >>> memory.to_blocks()
            []

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |   |[A | Z]|   |[x]|   |   |   |
            +---+---+---+---+---+---+---+---+---+

            >>> items = [
            ...     (0, ord('A')),
            ...     (1, ord('B')),
            ...     (3, ord('x')),
            ...     (1, ord('Z')),
            ... ]
            >>> memory = Memory.from_items(items, offset=2)
            >>> memory.to_blocks()
            [[2, b'AZ'], [5, b'x']]
        """
        cdef:
            Memory_* memory_ = Memory_FromItems(items, offset, start, endex, validate)

        return Memory_AsObject(memory_)

    @classmethod
    def from_memory(
        cls: Type[Memory],
        memory: ImmutableMemory,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> ImmutableMemory:
        r"""Creates a virtual memory from another one.

        Arguments:
            memory (Memory):
                A :obj:`ImmutableMemory` to copy data from.

            offset (int):
                Some address offset applied to all the blocks.

            start (int):
                Optional memory start address.
                Anything before will be trimmed away.

            endex (int):
                Optional memory exclusive end address.
                Anything at or after it will be trimmed away.

            copy (bool):
                Forces copy of provided input data into the underlying data
                structure.

            validate (bool):
                Validates the resulting :obj:`MemorImmutableMemory` object.

        Returns:
            :obj:`ImmutableMemory`: The resulting memory object.

        Raises:
            :obj:`ValueError`: Some requirements are not satisfied.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory1 = Memory.from_bytes(b'ABC', 5)
            >>> memory2 = Memory.from_memory(memory1)
            >>> memory2.to_blocks()
            [[5, b'ABC']]
            >>> memory1 == memory2
            True
            >>> memory1 is memory2
            False
            >>> memory1.to_blocks() is memory2.to_blocks()
            False

            ~~~

            >>> memory1 = Memory.from_bytes(b'ABC', 10)
            >>> memory2 = Memory.from_memory(memory1, -3)
            >>> memory2.to_blocks()
            [[7, b'ABC']]
            >>> memory1 == memory2
            False

            ~~~

            >>> memory1 = Memory.from_bytes(b'ABC', 10)
            >>> memory2 = Memory.from_memory(memory1, copy=False)
            >>> all((b1[1] is b2[1])  # compare block data
            ...     for b1, b2 in zip(memory1.to_blocks(), memory2.to_blocks()))
            True
        """
        cdef:
            Memory_* memory_ = Memory_FromMemory(memory, offset, start, endex, validate)

        return Memory_AsObject(memory_)

    @classmethod
    def from_values(
        cls: Type[Memory],
        values: Iterable[Optional[Value]],
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        validate: bool = True,
    ) -> Memory:
        r"""Creates a virtual memory from a byte-like sequence.

        Arguments:
            values (iterable byte-like sequence):
                An iterable sequence of byte values.
                Values of ``None`` are translated as gaps.

            offset (int):
                An address offset applied to all the values.

            start (int):
                Optional memory start address.
                Anything before will be trimmed away.

            endex (int):
                Optional memory exclusive end address.
                Anything at or after it will be trimmed away.

            validate (bool):
                Validates the resulting :obj:`ImmutableMemory` object.

        Returns:
            :obj:`ImmutableMemory`: The resulting memory object.

        Raises:
            :obj:`ValueError`: Some requirements are not satisfied.

        See Also:
            :meth:`to_bytes`

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory.from_values(range(0))
            >>> memory.to_blocks()
            []

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |   |[A | B | C | D | E]|   |   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_values(range(ord('A'), ord('F')), offset=2)
            >>> memory.to_blocks()
            [[2, b'ABCDE']]
        """
        cdef:
            Memory_* memory_ = Memory_FromValues(values, offset, start, endex, validate)

        return Memory_AsObject(memory_)

    @classmethod
    def fromhex(
        cls,
        string: str,
    ) -> ImmutableMemory:
        r"""Creates a virtual memory from an hexadecimal string.

        Arguments:
            string (str):
                Hexadecimal string.

        Returns:
            :obj:`ImmutableMemory`: The resulting memory object.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory.fromhex('')
            >>> bytes(memory)
            b''

            ~~~

            >>> memory = Memory.fromhex('48656C6C6F2C20576F726C6421')
            >>> bytes(memory)
            b'Hello, World!'
        """
        cdef:
            Memory_* memory_ = Memory_FromBytes(bytes.fromhex(string), 0, None, None)

        return Memory_AsObject(memory_)

    def gaps(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[OpenInterval]:
        r"""Iterates over block gaps.

        Iterates over gaps emptiness bounds within an address range.
        If a yielded bound is ``None``, that direction is infinitely empty
        (valid before or after global data bounds).

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Yields:
            pair of addresses: Block data interval boundaries.

        See Also:
            :meth:`intervals`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> list(memory.gaps())
            [(None, 1), (3, 5), (6, 7), (10, None)]
            >>> list(memory.gaps(0, 11))
            [(0, 1), (3, 5), (6, 7), (10, 11)]
            >>> list(memory.gaps(*memory.span))
            [(3, 5), (6, 7)]
            >>> list(memory.gaps(2, 6))
            [(3, 5)]
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex

        if block_count:
            start__ = start
            endex__ = endex
            start_, endex_ = Memory_Bound(self._, start, endex)

            if start__ is None:
                block = Rack_First__(blocks)
                start_ = Block_Start(block)  # override trim start
                yield None, start_
                block_index_start = 0
            else:
                block_index_start = Rack_IndexStart(blocks, start_)

            if endex__ is None:
                block_index_endex = block_count
            else:
                block_index_endex = Rack_IndexEndex(blocks, endex_)

            for block_index in range(block_index_start, block_index_endex):
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                if start_ < block_start:
                    yield start_, block_start
                start_ = Block_Endex(block)

            if endex__ is None:
                yield start_, None
            elif start_ < endex_:
                yield start_, endex_

        else:
            yield None, None

    def get(
        self: Memory,
        address: Address,
        default: Optional[Value] = None,
    ) -> Optional[Value]:
        r"""Gets the item at an address.

        Returns:
            int: The item at `address`, `default` if empty.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.get(3)  # -> ord('C') = 67
            67
            >>> memory.get(6)  # -> ord('$') = 36
            36
            >>> memory.get(10)  # -> ord('z') = 122
            122
            >>> memory.get(0)  # -> empty -> default = None
            None
            >>> memory.get(7)  # -> empty -> default = None
            None
            >>> memory.get(11)  # -> empty -> default = None
            None
            >>> memory.get(0, 123)  # -> empty -> default = 123
            123
            >>> memory.get(7, 123)  # -> empty -> default = 123
            123
            >>> memory.get(11, 123)  # -> empty -> default = 123
            123
        """
        cdef:
            addr_t address_ = <addr_t>address
            const Memory_* memory = self._
            const Rack_* blocks = memory.blocks
            ssize_t block_index = Rack_IndexAt(blocks, address_)
            const Block_* block

        if block_index < 0:
            return default
        else:
            block = Rack_Get__(blocks, <size_t>block_index)
            return Block_Get__(block, <size_t>(address_ - Block_Start(block)))

    def hex(
        self: Memory,
        *args: Any,  # see docstring
    ) -> str:
        r"""Converts into an hexadecimal string.

        Arguments:
            sep (str):
                Separator string between bytes.
                Defaults to an emoty string if not provided.
                Available since Python 3.8.

            bytes_per_sep (int):
                Number of bytes grouped between separators.
                Defaults to one byte per group.
                Available since Python 3.8.

        Returns:
            str: Hexadecimal string representation.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().hex() == ''
            True

            ~~~

            >>> memory = Memory.from_bytes(b'Hello, World!')
            >>> memory.hex()
            48656c6c6f2c20576f726c6421
            >>> memory.hex('.')
            48.65.6c.6c.6f.2c.20.57.6f.72.6c.64.21
            >>> memory.hex('.', 4)
            48.656c6c6f.2c20576f.726c6421
        """
        cdef:
            const Memory_* memory = self._
            bytes data

        if not Rack_Bool(memory.blocks):
            return ''

        data = self.__bytes__()
        return data.hex(*args)

    def index(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:
        r"""Index of an item.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The index of the first item equal to `value`.

        Raises:
            :obj:`ValueError`: Item not found.
        """

        return Memory_Index(self._, item, start, endex)

    def insert(
        self: Memory,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
    ) -> None:
        r"""Inserts data.

        Inserts data, moving existing items after the insertion address by the
        size of the inserted data.

        Arguments::
            address (int):
                Address of the insertion point.

            data (bytes):
                Data to insert.

        See Also:
            :meth:`insert_backup`
            :meth:`insert_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |   |[x | y | z]|   |[$]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |   |[x | y | 1 | z]|   |[$]|
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> memory.insert(10, b'$')
            >>> memory.to_blocks()
            [[1, b'ABC'], [6, b'xyz'], [10, b'$']]
            >>> memory.insert(8, b'1')
            >>> memory.to_blocks()
            [[1, b'ABC'], [6, b'xy1z'], [11, b'$']]
        """

        Memory_Insert(self._, address, data)

    def insert_backup(
        self: Memory,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
    ) -> Tuple[Address, ImmutableMemory]:
        r"""Backups an `insert()` operation.

        Arguments:
            address (int):
                Address of the insertion point.

            data (bytes):
                Data to insert.

        Returns:
            (int, :obj:`ImmutableMemory`): Insertion address, backup memory region.

        See Also:
            :meth:`insert`
            :meth:`insert_restore`
        """

        size = 1 if isinstance(data, int) else len(data)
        backup = self._pretrim_endex_backup(address, size)
        return address, backup

    def insert_restore(
        self: Memory,
        address: Address,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores an `insert()` operation.

        Arguments:
            address (int):
                Address of the insertion point.

            backup (:obj:`Memory`):
                Backup memory region to restore.

        See Also:
            :meth:`insert`
            :meth:`insert_backup`
        """
        cdef:
            Memory_* memory = self._
            const Memory_* backup_
            addr_t address_ = <addr_t>address
            addr_t size

        if isinstance(backup, Memory):
            backup_ = (<Memory>backup)._
            size = Memory_Length(backup_)
            CheckAddAddrU(address_, size)
            Memory_Delete_(memory, address_, address_ + size)
            Memory_WriteSame_(memory, 0, backup_, True)
        else:
            size = <addr_t>len(backup)
            CheckAddAddrU(address_, size)
            Memory_Delete_(memory, address_, address_ + size)
            Memory_Write(memory, 0, backup, True)

    def intervals(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[ClosedInterval]:
        r"""Iterates over block intervals.

        Iterates over data boundaries within an address range.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Yields:
            pair of addresses: Block data interval boundaries.

        See Also:
            :meth:`blocks`
            :meth:`gaps`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> list(memory.intervals())
            [(1, 3), (5, 6), (7, 10)]
            >>> list(memory.intervals(2, 9))
            [(2, 3), (5, 6), (7, 9)]
            >>> list(memory.intervals(3, 5))
            []
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            size_t slice_start
            size_t slice_endex

        if block_count:
            block_index_start = 0 if start is None else Rack_IndexStart(blocks, <addr_t>start)
            block_index_endex = block_count if endex is None else Rack_IndexEndex(blocks, <addr_t>endex)
            start_, endex_ = Memory_Bound(self._, start, endex)

            for block_index in range(block_index_start, block_index_endex):
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                block_endex = Block_Endex(block)
                slice_start = block_start if start_ < block_start else start_
                slice_endex = endex_ if endex_ < block_endex else block_endex
                if slice_start < slice_endex:
                    yield slice_start, slice_endex

    def items(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Tuple[Address, Optional[Value]]]:
        r"""Iterates over address and value pairs.

        Iterates over address and value pairs, from `start` to `endex`.
        Implemets the interface of :obj:`dict`.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.
                If ``Ellipsis``, the iterator is infinite.

            pattern (items):
                Pattern of values to fill emptiness.

        Yields:
            int: Range address and value pairs.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> from itertools import islice
            >>> memory = Memory()
            >>> list(memory.items(endex=8))
            [(0, None), (1, None), (2, None), (3, None), (4, None), (5, None), (6, None), (7, None)]
            >>> list(memory.items(3, 8))
            [(3, None), (4, None), (5, None), (6, None), (7, None)]
            >>> list(islice(memory.items(3, ...), 7))
            [(3, None), (4, None), (5, None), (6, None), (7, None), (8, None), (9, None)]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67|   |   |120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.items())
            [(1, 65), (2, 66), (3, 67), (4, None), (5, None), (6, 120), (7, 121), (8, 122)]
            >>> list(memory.items(3, 8))
            [(3, 67), (4, None), (5, None), (6, 120), (7, 121)]
            >>> list(islice(memory.items(3, ...), 7))
            [(3, 67), (4, None), (5, None), (6, 120), (7, 121), (8, 122), (9, None)]
        """
        cdef:
            addr_t start_
            addr_t endex_
            Rover_* rover = NULL
            byte_t pattern_value
            const byte_t[:] pattern_view
            size_t pattern_size = 0
            const byte_t* pattern_data = NULL
            addr_t address
            int value

        if start is None:
            start_ = Memory_Start(self._)
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = Memory_Endex(self._)
        elif endex is Ellipsis:
            endex_ = ADDR_MAX
        else:
            endex_ = <addr_t>endex

        if pattern is not None:
            if isinstance(pattern, int):
                pattern_value = <byte_t>pattern
                pattern_size = 1
                pattern_data = &pattern_value
            else:
                try:
                    pattern_view = pattern
                except TypeError:
                    pattern_view = bytes(pattern)
                with cython.boundscheck(False):
                    pattern_size = len(pattern_view)
                    pattern_data = &pattern_view[0]

        try:
            rover = Rover_Create(self._, start_, endex_, pattern_size, pattern_data, True, endex is Ellipsis)
            while Rover_HasNext(rover):
                address = Rover_Address(rover)
                value = Rover_Next_(rover)
                yield address, (None if value < 0 else value)
        finally:
            rover = Rover_Free(rover)

    def keys(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
    ) -> Iterator[Address]:
        r"""Iterates over addresses.

        Iterates over addresses, from `start` to `endex`.
        Implemets the interface of :obj:`dict`.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.
                If ``Ellipsis``, the iterator is infinite.

        Yields:
            int: Range address.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> from itertools import islice
            >>> memory = Memory()
            >>> list(memory.keys())
            []
            >>> list(memory.keys(endex=8))
            [0, 1, 2, 3, 4, 5, 6, 7]
            >>> list(memory.keys(3, 8))
            [3, 4, 5, 6, 7]
            >>> list(islice(memory.keys(3, ...), 7))
            [3, 4, 5, 6, 7, 8, 9]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.keys())
            [1, 2, 3, 4, 5, 6, 7, 8]
            >>> list(memory.keys(endex=8))
            [1, 2, 3, 4, 5, 6, 7]
            >>> list(memory.keys(3, 8))
            [3, 4, 5, 6, 7]
            >>> list(islice(memory.keys(3, ...), 7))
            [3, 4, 5, 6, 7, 8, 9]
        """
        cdef:
            addr_t start_
            addr_t endex_

        if start is None:
            start_ = Memory_Start(self._)
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = Memory_Endex(self._)
        elif endex is Ellipsis:
            endex_ = ADDR_MAX
        else:
            endex_ = <addr_t>endex

        while start_ < endex_:
            yield start_
            start_ += 1

    def ofind(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Optional[Address]:
        r"""Index of an item.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The index of the first item equal to `value`, or ``None``.
        """

        return Memory_ObjFind(self._, item, start, endex)

    def peek(
        self: Memory,
        address: Address,
    ) -> Optional[Value]:
        r"""Gets the item at an address.

        Returns:
            int: The item at `address`, ``None`` if empty.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.peek(3)  # -> ord('C') = 67
            67
            >>> memory.peek(6)  # -> ord('$') = 36
            36
            >>> memory.peek(10)  # -> ord('z') = 122
            122
            >>> memory.peek(0)
            None
            >>> memory.peek(7)
            None
            >>> memory.peek(11)
            None
        """

        return Memory_Peek(self._, address)

    def poke(
        self: Memory,
        address: Address,
        item: Optional[Union[AnyBytes, Value]],
    ) -> None:
        r"""Sets the item at an address.

        Arguments:
            address (int):
                Address of the target item.

            item (int or byte):
                Item to set, ``None`` to clear the cell.

        See Also:
            :meth:`poke_backup`
            :meth:`poke_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.poke(3, b'@')
            >>> memory.peek(3)  # -> ord('@') = 64
            64
            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.poke(5, b'@')
            >>> memory.peek(5)  # -> ord('@') = 64
            64
        """

        Memory_Poke(self._, address, item)

    def poke_backup(
        self: Memory,
        address: Address,
    ) -> Tuple[Address, Optional[Value]]:
        r"""Backups a `poke()` operation.

        Arguments:
            address (int):
                Address of the target item.

        Returns:
            (int, int): `address`, item at `address` (``None`` if empty).

        See Also:
            :meth:`poke`
            :meth:`poke_restore`
        """

        return address, Memory_Peek(self._, address)

    def poke_restore(
        self: Memory,
        address: Address,
        item: Optional[Value],
    ) -> None:
        r"""Restores a `poke()` operation.

        Arguments:
            address (int):
                Address of the target item.

            item (int or byte):
                Item to restore.

        See Also:
            :meth:`poke`
            :meth:`poke_backup`
        """

        if item is None:
            Memory_PokeNone_(self._, <addr_t>address)
        else:
            Memory_Poke_(self._, <addr_t>address, <byte_t>item)

    def pop(
        self: Memory,
        address: Optional[Address] = None,
        default: Optional[Value] = None,
    ) -> Optional[Value]:
        r"""Takes a value away.

        Arguments:
            address (int):
                Address of the byte to pop.
                If ``None``, the very last byte is popped.

            default (int):
                Value to return if `address` is within emptiness.

        Return:
            int: Value at `address`; `default` within emptiness.

        See Also:
            :meth:`pop_backup`
            :meth:`pop_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | D]|   |[$]|   |[x | y]|   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | D]|   |[$]|   |[x | y]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.pop()  # -> ord('z') = 122
            122
            >>> memory.pop(3)  # -> ord('C') = 67
            67
            >>> memory.pop(6, 63)  # -> ord('?') = 67
            63
        """

        return Memory_Pop(self._, address, default)

    def pop_backup(
        self: Memory,
        address: Optional[Address] = None,
    ) -> Tuple[Address, Optional[Value]]:
        r"""Backups a `pop()` operation.

        Arguments:
            address (int):
                Address of the byte to pop.
                If ``None``, the very last byte is popped.

        Returns:
            (int, int): `address`, item at `address` (``None`` if empty).

        See Also:
            :meth:`pop`
            :meth:`pop_restore`
        """

        if address is None:
            address = self.endex - 1
        return address, Memory_Peek(self._, address)

    def pop_restore(
        self: Memory,
        address: Address,
        item: Optional[Value],
    ) -> None:
        r"""Restores a `pop()` operation.

        Arguments:
            address (int):
                Address of the target item.

            item (int or byte):
                Item to restore, ``None`` if empty.

        See Also:
            :meth:`pop`
            :meth:`pop_backup`
        """
        cdef:
            byte_t value

        if item is None:
            Memory_Reserve_(self._, <addr_t>address, 1)
        else:
            value = <byte_t>item
            Memory_InsertRaw_(self._, address, 1, &value)

    def popitem(
        self: Memory,
    ) -> Tuple[Address, Value]:
        r"""Pops the last item.

        Return:
            (int, int): Address and value of the last item.

        See Also:
            :meth:`popitem_backup`
            :meth:`popitem_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A]|   |   |   |   |   |   |   |[y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'A'], [9, b'yz']])
            >>> memory.popitem()  # -> ord('z') = 122
            (10, 122)
            >>> memory.popitem()  # -> ord('y') = 121
            (9, 121)
            >>> memory.popitem()  # -> ord('A') = 65
            (1, 65)
            >>> memory.popitem()
            Traceback (most recent call last):
                ...
            KeyError: empty
        """
        cdef:
            addr_t address
            int value

        address, value = Memory_PopItem(self._)
        return address, value

    def popitem_backup(
        self: Memory,
    ) -> Tuple[Address, Value]:
        r"""Backups a `popitem()` operation.

        Returns:
            (int, int): Address and value of the last item.

        See Also:
            :meth:`popitem`
            :meth:`popitem_restore`
        """
        cdef:
            const Memory_* memory = self._
            const Rack_* blocks = memory.blocks
            const Block_* block
            size_t offset
            byte_t backup

        if Rack_Bool(blocks):
            block = Rack_Last_(blocks)
            offset = Block_Length(block)
            if offset:
                offset -= 1
                address = Block_Start(block) + offset
                backup = Block_Get__(block, offset)
                return address, backup
            else:
                raise RuntimeError('empty block')
        else:
            raise KeyError('empty')

    def popitem_restore(
        self: Memory,
        address: Address,
        item: Value,
    ) -> None:
        r"""Restores a `popitem()` operation.

        Arguments:
            address (int):
                Address of the target item.

            item (int):
                Item to restore.

        See Also:
            :meth:`popitem`
            :meth:`popitem_backup`
        """
        cdef:
            addr_t address_ = <addr_t>address
            byte_t item_ = <byte_t>item
            Memory_* memory = self._

        if address_ == Memory_ContentEndex(memory):
            Memory_Append_(memory, item_)
        else:
            Memory_InsertRaw_(memory, address_, 1, &item_)

    def remove(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:
        r"""Removes an item.

        Searches and deletes the first occurrence of an item.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Raises:
            :obj:`ValueError`: Item not found.

        See Also:
            :meth:`remove_backup`
            :meth:`remove_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | D]|   |[$]|   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | D]|   |   |[x | y | z]|   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.remove(b'BC')
            >>> memory.to_blocks()
            [[1, b'AD'], [4, b'$'], [6, b'xyz']]
            >>> memory.remove(ord('$'))
            >>> memory.to_blocks()
            [[1, b'AD'], [5, b'xyz']]
            >>> memory.remove(b'?')
            Traceback (most recent call last):
                ...
            ValueError: subsection not found
        """
        cdef:
            Memory_* memory = self._
            size_t size = 1 if isinstance(item, int) else <size_t>len(item)
            addr_t address = <addr_t>Memory_Index(memory, item, start, endex)

        CheckAddAddrU(address, size)
        Memory_Erase__(memory, address, address + size, True)  # delete

    def remove_backup(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:
        r"""Backups a `remove()` operation.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            :obj:`Memory`: Backup memory region.

        See Also:
            :meth:`remove`
            :meth:`remove_restore`
        """
        cdef:
            Memory_* memory = self._
            size_t size = 1 if isinstance(item, int) else <size_t>len(item)
            addr_t address = <addr_t>Memory_Index(memory, item, start, endex)

        CheckAddAddrU(address, size)
        memory = Memory_Extract__(memory, address, address + size, 0, NULL, 1, True)
        return Memory_AsObject(memory)

    def remove_restore(
        self: Memory,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores a `remove()` operation.

        Arguments:
            backup (:obj:`Memory`):
                Backup memory region.

        See Also:
            :meth:`remove`
            :meth:`remove_backup`
        """
        cdef:
            Memory_* memory = self._
            const Memory_* backup_

        if isinstance(backup, Memory):
            backup_ = (<Memory>backup)._
            Memory_Reserve_(memory, Memory_Start(backup_), Memory_Length(backup_))
            Memory_WriteSame_(memory, 0, backup_, True)
        else:
            Memory_Reserve(memory, backup.start, len(backup))
            Memory_Write(memory, 0, backup, True)

    def reserve(
        self: Memory,
        address: Address,
        size: Address,
    ) -> None:
        r"""Inserts emptiness.

        Reserves emptiness at the provided address.

        Arguments:
            address (int):
                Start address of the emptiness to insert.

            size (int):
                Size of the emptiness to insert.

        See Also:
            :meth:`reserve_backup`
            :meth:`reserve_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |[A]|   |   | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[3, b'ABC'], [7, b'xyz']])
            >>> memory.reserve(4, 2)
            >>> memory.to_blocks()
            [[3, b'A'], [6, b'BC'], [9, b'xyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |   |   |[A | B | C]|   |[x | y | z]|)))|
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |   |   |   |   |   |   |   |[A | B]|)))|
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']], endex=12)
            >>> memory.reserve(5, 5)
            >>> memory.to_blocks()
            [[10, b'AB']]
        """

        Memory_Reserve(self._, address, size)

    def reserve_backup(
        self: Memory,
        address: Address,
        size: Address,
    ) -> Tuple[Address, ImmutableMemory]:
        r"""Backups a `reserve()` operation.

        Arguments:
            address (int):
                Start address of the emptiness to insert.

            size (int):
                Size of the emptiness to insert.

        Returns:
            (int, :obj:`ImmutableMemory`): Reservation address, backup memory region.

        See Also:
            :meth:`reserve`
            :meth:`reserve_restore`
        """

        backup = self._pretrim_endex_backup(address, size)
        return address, backup

    def reserve_restore(
        self: Memory,
        address: Address,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores a `reserve()` operation.

        Arguments:
            address (int):
                Address of the reservation point.

            backup (:obj:`ImmutableMemory`):
                Backup memory region to restore.

        See Also:
            :meth:`reserve`
            :meth:`reserve_backup`
        """
        cdef:
            addr_t address_ = <addr_t>address
            addr_t size
            Memory_* memory = self._
            const Memory_* backup_

        if isinstance(backup, Memory):
            backup_ = (<Memory>backup)._
            size = Memory_Length(backup_)
            CheckAddAddrU(address_, size)
            Memory_Delete_(memory, address_, address_ + size)
            Memory_WriteSame_(memory, 0, backup_, True)
        else:
            size = <addr_t>len(backup)
            CheckAddAddrU(address_, size)
            Memory_Delete_(memory, address_, address_ + size)
            Memory_Write(memory, 0, backup, True)

    def reverse(
        self: Memory,
    ) -> None:
        r"""Reverses the memory in-place.

        Data is reversed within the memory :attr:`span`.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[z | y | x]|   |[$]|   |[D | C | B | A]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.reverse()
            >>> memory.to_blocks()
            [[1, b'zyx'], [5, b'$'], [7, b'DCBA']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |   |[[[|   |[A | B | C]|   |   |   |)))|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |   |[[[|   |   |[C | B | A]|   |   |)))|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_bytes(b'ABCD', 3, start=2, endex=10)
            >>> memory.reverse()
            >>> memory.to_blocks()
            [[5, b'CBA']]
        """

        Memory_Reverse(self._)

    def rfind(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:
        r"""Index of an item, reversed search.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The index of the last item equal to `value`, or -1.
        """

        return Memory_RevFind(self._, item, start, endex)

    def rindex(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:
        r"""Index of an item, reversed search.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The index of the last item equal to `value`.

        Raises:
            :obj:`ValueError`: Item not found.
        """

        return Memory_RevIndex(self._, item, start, endex)

    def rofind(
        self: Memory,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Optional[Address]:
        r"""Index of an item, reversed search.

        Arguments:
            item (items):
                Value to find. Can be either some byte string or an integer.

            start (int):
                Inclusive start of the searched range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the searched range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            int: The index of the last item equal to `value`, or ``None``.
        """

        return Memory_RevObjFind(self._, item, start, endex)

    def rvalues(
        self: Memory,
        start: Optional[Union[Address, EllipsisType]] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Optional[Value]]:
        r"""Iterates over values, reversed order.

        Iterates over values, from `endex` to `start`.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.
                If ``Ellipsis``, the iterator is infinite.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

            pattern (items):
                Pattern of values to fill emptiness.

        Yields:
            int: Range values.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |   |   | A | B | C | D | A |   |   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |   |   | 65| 66| 67| 68| 65|   |   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> from itertools import islice
            >>> memory = Memory()
            >>> list(memory.rvalues(endex=8))
            [None, None, None, None, None, None, None, None]
            >>> list(memory.rvalues(3, 8))
            [None, None, None, None, None]
            >>> list(islice(memory.rvalues(..., 8), 7))
            [None, None, None, None, None, None, None]
            >>> list(memory.rvalues(3, 8, b'ABCD'))
            [65, 68, 67, 66, 65]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|<1 | 2>|[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67|   |   |120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67| 49| 50|120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.rvalues())
            [122, 121, 120, None, None, 67, 66, 65]
            >>> list(memory.rvalues(3, 8))
            [121, 120, None, None, 67]
            >>> list(islice(memory.rvalues(..., 8), 7))
            [121, 120, None, None, 67, 66, 65]
            >>> list(memory.rvalues(3, 8, b'0123'))
            [121, 120, 50, 49, 67]
        """
        cdef:
            addr_t start_
            addr_t endex_
            Rover_* rover = NULL
            byte_t pattern_value
            const byte_t[:] pattern_view
            size_t pattern_size = 0
            const byte_t* pattern_data = NULL

        if start is None:
            start_ = Memory_Start(self._)
        elif start is Ellipsis:
            start_ = ADDR_MIN
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = Memory_Endex(self._)
        else:
            endex_ = <addr_t>endex

        if pattern is not None:
            if isinstance(pattern, int):
                pattern_value = <byte_t>pattern
                pattern_size = 1
                pattern_data = &pattern_value
            else:
                try:
                    pattern_view = pattern
                except TypeError:
                    pattern_view = bytes(pattern)
                with cython.boundscheck(False):
                    pattern_size = len(pattern_view)
                    pattern_data = &pattern_view[0]

        rover = Rover_Create(self._, start_, endex_, pattern_size, pattern_data, False, start is Ellipsis)
        try:
            while True:
                yield Rover_Next(rover)
        except StopIteration:
            pass
        finally:
            Rover_Free(rover)
            if pattern is not None:  # keep
                pattern = None  # release

    def setdefault(
        self: Memory,
        address: Address,
        default: Optional[Union[AnyBytes, Value]] = None,
    ) -> Optional[Value]:
        r"""Defaults a value.

        Arguments:
            address (int):
                Address of the byte to set.

            default (int):
                Value to set if `address` is within emptiness.

        Return:
            int: Value at `address`; `default` within emptiness.

        See Also:
            :meth:`setdefault_backup`
            :meth:`setdefault_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.setdefault(3, b'@')  # -> ord('C') = 67
            67
            >>> memory.peek(3)  # -> ord('C') = 67
            67
            >>> memory.setdefault(5, 64)  # -> ord('@') = 64
            64
            >>> memory.peek(5)  # -> ord('@') = 64
            64
            >>> memory.setdefault(9) is None
            False
            >>> memory.peek(9) is None
            False
            >>> memory.setdefault(7) is None
            True
            >>> memory.peek(7) is None
            True
        """
        cdef:
            addr_t address_ = <addr_t>address
            Memory_* memory = self._
            int backup = Memory_Peek_(memory, address_)
            const byte_t[:] view
            byte_t value

        if backup < 0:
            if default is not None:
                if isinstance(default, int):
                    value = <byte_t>default
                else:
                    view = default
                    if len(view) != 1:
                        raise ValueError('expecting single item')
                    with cython.boundscheck(False):
                        value = view[0]
                Memory_Poke_(memory, address_, value)
                return value
            else:
                return None
        else:
            return backup

    def setdefault_backup(
        self: Memory,
        address: Address,
    ) -> Tuple[Address, Optional[Value]]:
        r"""Backups a `setdefault()` operation.

        Arguments:
            address (int):
                Address of the byte to set.

        Returns:
            (int, int): `address`, item at `address` (``None`` if empty).

        See Also:
            :meth:`setdefault`
            :meth:`setdefault_restore`
        """

        return address, Memory_Peek(self._, address)

    def setdefault_restore(
        self: Memory,
        address: Address,
        item: Optional[Value],
    ) -> None:
        r"""Restores a `setdefault()` operation.

        Arguments:
            address (int):
                Address of the target item.

            item (int or byte):
                Item to restore, ``None`` if empty.

        See Also:
            :meth:`setdefault`
            :meth:`setdefault_backup`
        """

        Memory_Poke(self._, address, item)

    def shift(
        self: Memory,
        offset: Address,
    ) -> None:
        r"""Shifts the items.

        Arguments:
            offset (int):
                Signed amount of address shifting.

        See Also:
            :meth:`shift_backup`
            :meth:`shift_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |   |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
            >>> memory.shift(-2)
            >>> memory.to_blocks()
            [[3, b'ABC'], [7, b'xyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[[[|   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |[y | z]|   |   |   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']], start=3)
            >>> memory.shift(-8)
            >>> memory.to_blocks()
            [[2, b'yz']]
        """

        Memory_Shift(self._, offset)

    def shift_backup(
        self: Memory,
        offset: Address,
    ) -> Tuple[Address, ImmutableMemory]:
        r"""Backups a `shift()` operation.

        Arguments:
            offset (int):
                Signed amount of address shifting.

        Returns:
            (int, :obj:`ImmutableMemory`): Shifting, backup memory region.

        See Also:
            :meth:`shift`
            :meth:`shift_restore`
        """
        cdef:
            Memory backup

        if offset < 0:
            backup = self._pretrim_start_backup(None, -offset)
        else:
            backup = self._pretrim_endex_backup(None, +offset)
        return offset, backup

    def shift_restore(
        self: Memory,
        offset: Address,
        backup: ImmutableMemory,
    ) -> None:
        r"""Restores an `shift()` operation.

        Arguments:
            offset (int):
                Signed amount of address shifting.

            backup (:obj:`ImmutableMemory`):
                Backup memory region to restore.

        See Also:
            :meth:`shift`
            :meth:`shift_backup`
        """
        cdef:
            Memory_* memory = self._

        Memory_Shift(memory, -offset)
        Memory_Write(memory, 0, backup, True)

    @property
    def span(
        self: Memory,
    ) -> ClosedInterval:
        r"""tuple of int: Memory address span.

        A :obj:`tuple` holding both :attr:`start` and :attr:`endex`.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().span
            (0, 0)
            >>> Memory(start=1, endex=8).span
            (1, 8)

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.span
            (1, 8)
        """

        return Memory_Span(self._)

    @property
    def start(
        self: Memory,
    ) -> Address:
        r"""int: Inclusive start address.

        This property holds the inclusive start address of the virtual space.
        By default, it is the current minimum inclusive start address of
        the first stored block.

        If :attr:`trim_start` not ``None``, that is returned.

        If the memory has no data and no trimming, 0 is returned.

        Examples:
            >>> from cbytesparse.c import Memory

            >>> Memory().start
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
            >>> memory.start
            1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[[[|   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[5, b'xyz']], start=1)
            >>> memory.start
            1
        """

        return Memory_Start(self._)

    def to_blocks(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> BlockList:
        r"""Exports into blocks.

        Exports data blocks within an address range, converting them into
        standalone :obj:`bytes` objects.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Returns:
            list of blocks: Exported data blocks.

        See Also:
            :meth:`blocks`
            :meth:`from_blocks`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> memory.to_blocks()
            [[1, b'AB'], [5, b'x'], [7, b'123']]
            >>> memory.to_blocks(2, 9)
            [[2, b'B'], [5, b'x'], [7, b'12']]
            >>> memory.to_blocks(3, 5)]
            []
        """
        cdef:
            addr_t start_
            addr_t endex_
            const Rack_* blocks = self._.blocks
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            const Block_* block
            addr_t block_start
            addr_t block_endex
            addr_t slice_start
            addr_t slice_endex
            bytes slice_data
            list result = []

        if block_count:
            if start is None and endex is None:  # faster
                for block_index in range(block_count):
                    block = Rack_Get__(blocks, block_index)
                    result.append([Block_Start(block), Block_Bytes(block)])
            else:
                block_index_start = 0 if start is None else Rack_IndexStart(blocks, start)
                block_index_endex = block_count if endex is None else Rack_IndexEndex(blocks, endex)
                start_, endex_ = Memory_Bound(self._, start, endex)

                for block_index in range(block_index_start, block_index_endex):
                    block = Rack_Get__(blocks, block_index)
                    block_start = Block_Start(block)
                    block_endex = Block_Endex(block)
                    slice_start = block_start if start_ < block_start else start_
                    slice_endex = endex_ if endex_ < block_endex else block_endex
                    if slice_start < slice_endex:
                        slice_start -= block_start
                        slice_endex -= block_start
                        slice_data = PyBytes_FromStringAndSize(<char*><void*>Block_At__(block, <size_t>slice_start),
                                                               <ssize_t>(slice_endex - slice_start))
                        result.append([block_start + slice_start, slice_data])
        return result

    def to_bytes(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> bytes:
        r"""Exports into bytes.

        Exports data within an address range, converting into a standalone
        :obj:`bytes` object.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.

        Returns:
            bytes: Exported data bytes.

        See Also:
            :meth:`from_bytes`
            :meth:`view`

        Examples:
            >>> from cbytesparse.c import Memory

            >>> memory = Memory.from_bytes(b'')
            >>> memory.to_bytes()
            b''

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |   |[A | B | C | x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_bytes(b'ABCxyz', 2)
            >>> memory.to_bytes()
            b'ABCxyz'
            >>> memory.to_bytes(start=4)
            b'Cxyz'
            >>> memory.to_bytes(endex=6)
            b'ABCx'
            >>> memory.to_bytes(4, 6)
            b'Cx'
        """
        cdef:
            BlockView view = Memory_View(self._, start, endex)
            bytes data = view.__bytes__()

        view.release_()
        return data

    @property
    def trim_endex(
        self: Memory,
    ) -> Optional[Address]:
        r"""int: Trimming exclusive end address.

        Any data at or after this address is automatically discarded.
        Disabled if ``None``.
        """

        return Memory_GetTrimEndex(self._)

    @trim_endex.setter
    def trim_endex(
        self: Memory,
        trim_endex: Address,
    ) -> None:

        Memory_SetTrimEndex(self._, trim_endex)

    @property
    def trim_span(
        self: Memory,
    ) -> OpenInterval:
        r"""tuple of int: Trimming span addresses.

        A :obj:`tuple` holding :attr:`trim_start` and :attr:`trim_endex`.
        """

        return Memory_GetTrimSpan(self._)

    @trim_span.setter
    def trim_span(
        self: Memory,
        trim_span: OpenInterval,
    ) -> None:

        Memory_SetTrimSpan(self._, trim_span)

    @property
    def trim_start(
        self: Memory,
    ) -> Optional[Address]:
        r"""int: Trimming start address.

        Any data before this address is automatically discarded.
        Disabled if ``None``.
        """

        return Memory_GetTrimStart(self._)

    @trim_start.setter
    def trim_start(
        self: Memory,
        trim_start: Address,
    ) -> None:

        Memory_SetTrimStart(self._, trim_start)

    def update(
        self: Memory,
        data: Union[AddressValueMapping,
                    Iterable[Tuple[Address, Value]],
                    Mapping[Address, Union[Value, AnyBytes]],
                    ImmutableMemory],
        clear: bool = False,
        **kwargs: Any,  # string keys cannot become addresses
    ) -> None:
        r"""Updates data.

        Arguments:
            data (iterable):
                Data to update with.
                Can be either another memory, an (address, value)
                mapping, or an iterable of (address, value) pairs.

            clear (bool):
                Clears the target range before writing data.
                Useful only if `data` is a :obj:`Memory` with empty spaces.

        See Also:
            :meth:`update_backup`
            :meth:`update_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |   |   |   |   |[A | B | C]|   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[x | y]|   |   |[A | B | C]|   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[x | y | @]|   |[A | ? | C]|   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory()
            >>> memory.update(Memory.from_bytes(b'ABC', 5))
            >>> memory.to_blocks()
            [[5, b'ABC']]
            >>> memory.update({1: b'x', 2: ord('y')})
            >>> memory.to_blocks()
            [[1, b'xy'], [5, b'ABC']]
            >>> memory.update([(6, b'?'), (3, ord('@'))])
            >>> memory.to_blocks()
            [[1, b'xy@'], [5, b'A?C']]
        """
        cdef:
            Memory_* memory = self._

        if kwargs:
            raise KeyError('cannot convert kwargs.keys() into addresses')

        if isinstance(data, ImmutableMemory):
            Memory_Write(memory, 0, data, clear)
        else:
            if isinstance(data, Mapping):
                data = data.items()
            for address, value in data:
                Memory_Poke(memory, address, value)

    def update_backup(
        self: Memory,
        data: Union[AddressValueMapping, Iterable[Tuple[Address, Value]], ImmutableMemory],
        clear: bool = False,
        **kwargs: Any,  # string keys cannot become addresses
    ) -> Union[AddressValueMapping, ImmutableMemory]:
        r"""Backups an `update()` operation.

        Arguments:
            data (iterable):
                Data to update with.
                Can be either another memory, an (address, value)
                mapping, or an iterable of (address, value) pairs.

            clear (bool):
                Clears the target range before writing data.
                Useful only if `data` is a :obj:`Memory` with empty spaces.

        Returns:
            list of :obj:`ImmutableMemory`: Backup memory regions.

        See Also:
            :meth:`update`
            :meth:`update_restore`
        """
        cdef:
            Memory_* memory = self._

        if kwargs:
            raise KeyError('cannot convert kwargs.keys() into addresses')

        if isinstance(data, ImmutableMemory):
            return self.write_backup(0, data, clear=clear)
        else:
            if isinstance(data, Mapping):
                backups = {address: Memory_Peek(memory, address) for address in data.keys()}
            else:
                backups = {address: Memory_Peek(memory, address) for address, _ in data}
            return backups

    def update_restore(
        self: Memory,
        backups: Union[AddressValueMapping, List[ImmutableMemory]],
    ) -> None:
        r"""Restores an `update()` operation.

        Arguments:
            backups (list of :obj:`ImmutableMemory`):
                Backup memory regions to restore.

        See Also:
            :meth:`update`
            :meth:`update_backup`
        """
        cdef:
            Memory_* memory = self._

        if isinstance(backups, list):
            for backup in backups:
                Memory_Write(memory, 0, backup, True)
        else:
            self.update(backups)

    def validate(
        self: Memory,
    ) -> None:
        r"""Validates internal structure.

        It makes sure that all the allocated blocks are sorted by block start
        address, and that all the blocks are non-overlapping.

        Raises:
            :obj:`ValueError`: Invalid data detected (see exception message).
        """

        Memory_Validate(self._)

    def values(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Optional[Value]]:
        r"""Iterates over values.

        Iterates over values, from `start` to `endex`.
        Implemets the interface of :obj:`dict`.

        Arguments:
            start (int):
                Inclusive start address.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address.
                If ``None``, :attr:`endex` is considered.
                If ``Ellipsis``, the iterator is infinite.

            pattern (items):
                Pattern of values to fill emptiness.

        Yields:
            int: Range values.

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |   |   | A | B | C | D | A |   |   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |   |   | 65| 66| 67| 68| 65|   |   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> from itertools import islice
            >>> memory = Memory()
            >>> list(memory.values(endex=8))
            [None, None, None, None, None, None, None, None]
            >>> list(memory.values(3, 8))
            [None, None, None, None, None]
            >>> list(islice(memory.values(3, ...), 7))
            [None, None, None, None, None, None, None]
            >>> list(memory.values(3, 8, b'ABCD'))
            [65, 66, 67, 68, 65]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|<1 | 2>|[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67|   |   |120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67| 49| 50|120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.values())
            [65, 66, 67, None, None, 120, 121, 122]
            >>> list(memory.values(3, 8))
            [67, None, None, 120, 121]
            >>> list(islice(memory.values(3, ...), 7))
            [67, None, None, 120, 121, 122, None]
            >>> list(memory.values(3, 8, b'0123'))
            [67, 49, 50, 120, 121]
        """
        cdef:
            addr_t start_
            addr_t endex_
            Rover_* rover = NULL
            byte_t pattern_value
            const byte_t[:] pattern_view
            size_t pattern_size = 0
            const byte_t* pattern_data = NULL

        if start is None:
            start_ = Memory_Start(self._)
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = Memory_Endex(self._)
        elif endex is Ellipsis:
            endex_ = ADDR_MAX
        else:
            endex_ = <addr_t>endex

        if pattern is not None:
            if isinstance(pattern, int):
                pattern_value = <byte_t>pattern
                pattern_size = 1
                pattern_data = &pattern_value
            else:
                try:
                    pattern_view = pattern
                except TypeError:
                    pattern_view = bytes(pattern)
                with cython.boundscheck(False):
                    pattern_size = len(pattern_view)
                    pattern_data = &pattern_view[0]

        try:
            rover = Rover_Create(self._, start_, endex_, pattern_size, pattern_data, True, endex is Ellipsis)
            while Rover_HasNext(rover):
                yield Rover_Next(rover)
        finally:
            rover = Rover_Free(rover)

    def view(
        self: Memory,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> BlockView:
        r"""Creates a view over a range.

        Creates a memory view over the selected address range.
        Data within the range is required to be contiguous.

        Arguments:
            start (int):
                Inclusive start of the viewed range.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end of the viewed range.
                If ``None``, :attr:`endex` is considered.

        Returns:
            :obj:`memoryview`: A view of the selected address range.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> bytes(memory.view(2, 5))
            b'BCD'
            >>> bytes(memory.view(9, 10))
            b'y'
            >>> memory.view()
            Traceback (most recent call last):
                ...
            ValueError: non-contiguous data within range
            >>> memory.view(0, 6)
            Traceback (most recent call last):
                ...
            ValueError: non-contiguous data within range
        """
        cdef:
            const Memory_* memory = self._
            addr_t start_ = Memory_Start(memory) if start is None else <addr_t>start
            addr_t endex_ = Memory_Endex(memory) if endex is None else <addr_t>endex

        return Memory_View_(memory, start_, endex_)

    def write(
        self: Memory,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
        clear: bool = False,
    ) -> None:
        r"""Writes data.

        Arguments:
            address (int):
                Address where to start writing data.

            data (bytes):
                Data to write.

            clear (bool):
                Clears the target range before writing data.
                Useful only if `data` is a :obj:`ImmutableMemory` with empty spaces.

        See Also:
            :meth:`write_backup`
            :meth:`write_restore`

        Examples:
            >>> from cbytesparse.c import Memory

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |[1 | 2 | 3 | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
            >>> memory.write(5, b'123')
            >>> memory.to_blocks()
            [[1, b'ABC'], [5, b'123z']]
        """

        Memory_Write(self._, address, data, clear)

    def write_backup(
        self: Memory,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
        clear: bool = False,
    ) -> List[ImmutableMemory]:
        r"""Backups a `write()` operation.

        Arguments:
            address (int):
                Address where to start writing data.

            data (bytes):
                Data to write.

            clear (bool):
                Clears the target range before writing data.
                Useful only if `data` is a :obj:`Memory` with empty spaces.

        Returns:
            list of :obj:`ImmutableMemory`: Backup memory regions.

        See Also:
            :meth:`write`
            :meth:`write_restore`
        """
        cdef:
            addr_t address_ = <addr_t>address
            Memory memory
            addr_t start
            addr_t endex
            list backups

        if isinstance(data, Memory):
            memory = <Memory>data
            start = Memory_Start(memory._)
            endex = Memory_Endex(memory._)
            CheckAddAddrU(start, address_)
            CheckAddAddrU(endex, address_)
            start += address
            endex += address
            if endex <= start:
                backups = []
            elif clear:
                backups = [Memory_Extract_(self._, start, endex, 0, NULL, 1, True)]
            else:
                backups = [Memory_Extract_(self._, <addr_t>block_start, <addr_t>block_endex, 0, NULL, 1, True)
                           for block_start, block_endex in memory.intervals(start=start, endex=endex)]

        elif isinstance(data, ImmutableMemory):
            start = <addr_t>data.start
            endex = <addr_t>data.endex
            CheckAddAddrU(start, address_)
            CheckAddAddrU(endex, address_)
            start += address
            endex += address
            if endex <= start:
                backups = []
            elif clear:
                backups = [Memory_Extract_(self._, start, endex, 0, NULL, 1, True)]
            else:
                backups = [Memory_Extract_(self._, <addr_t>block_start, <addr_t>block_endex, 0, NULL, 1, True)
                           for block_start, block_endex in data.intervals(start=start, endex=endex)]

        else:
            if isinstance(data, int):
                start = address_
                endex = start + 1
                backups = [Memory_Extract_(self._, start, endex, 0, NULL, 1, True)]
            else:
                start = address_
                endex = start + <size_t>len(data)
                if start < endex:
                    backups = [Memory_Extract_(self._, start, endex, 0, NULL, 1, True)]
                else:
                    backups = []

        return backups

    def write_restore(
        self: Memory,
        backups: Sequence[ImmutableMemory],
    ) -> None:
        r"""Restores a `write()` operation.

        Arguments:
            backups (list of :obj:`ImmutableMemory`):
                Backup memory regions to restore.

        See Also:
            :meth:`write`
            :meth:`write_backup`
        """
        cdef:
            Memory_* memory = self._

        for backup in backups:
            Memory_Write(memory, 0, backup, True)


ImmutableMemory.register(Memory)
MutableMemory.register(Memory)


# =====================================================================================================================

cdef class bytesparse(Memory):
    r"""Wrapper for more `bytearray` compatibility.

    This wrapper class can make :class:`Memory` closer to the actual
    :class:`bytearray` API.

    For instantiation, please refer to :meth:`bytearray.__init__`.

    With respect to :class:`Memory`, negative addresses are not allowed.
    Instead, negative addresses are to consider as referred to :attr:`endex`.

    Arguments:
        source:
            The optional `source` parameter can be used to initialize the
            array in a few different ways:

            * If it is a string, you must also give the `encoding` (and
              optionally, `errors`) parameters; it then converts the string to
              bytes using :meth:`str.encode`.

            * If it is an integer, the array will have that size and will be
              initialized with null bytes.

            * If it is an object conforming to the buffer interface, a
              read-only buffer of the object will be used to initialize the byte
              array.

            * If it is an iterable, it must be an iterable of integers in the
              range 0 <= x < 256, which are used as the initial contents of the
              array.

        encoding (str):
            Optional string encoding.

        errors (str):
            Optional string error management.

        start (int):
            Optional memory start address.
            Anything before will be trimmed away.
            If `source` is provided, its data start at this address
            (0 if `start` is ``None``).

        endex (int):
            Optional memory exclusive end address.
            Anything at or after it will be trimmed away.
    """

    def __init__(
        self,
        *args: Any,  # see bytearray.__init__()
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ):
        cdef:
            const byte_t[:] view
            addr_t address
            size_t size
            const byte_t* ptr

        super().__init__(start, endex)

        data = bytearray(*args)
        if data:
            if start is None:
                start = 0

            if endex is not None:
                if endex <= start:
                    return

                del data[(endex - start):]

            view = data
            address = <addr_t>start
            with cython.boundscheck(False):
                size = <size_t>len(view)
                ptr = &view[0]
            Memory_Place__(self._, address, size, ptr, False)

    def __delitem__(
        self,
        key: Union[Address, slice],
    ) -> None:

        if isinstance(key, slice):
            start, endex = self._rectify_span(key.start, key.stop)
            key = slice(start, endex, key.step)
        else:
            key = self._rectify_address(key)

        super().__delitem__(key)

    def __getitem__(
        self,
        key: Union[Address, slice],
    ) -> Any:

        if isinstance(key, slice):
            start, endex = self._rectify_span(key.start, key.stop)
            key = slice(start, endex, key.step)
        else:
            key = self._rectify_address(key)

        return super().__getitem__(key)

    def __setitem__(
        self,
        key: Union[Address, slice],
        value: Optional[Union[AnyBytes, Value]],
    ) -> None:

        if isinstance(key, slice):
            start, endex = self._rectify_span(key.start, key.stop)
            key = slice(start, endex, key.step)
        else:
            key = self._rectify_address(key)

        super().__setitem__(key, value)

    def _rectify_address(
        self,
        address: Address,
    ) -> Address:
        r"""Rectifies an address.

        In case the provided `address` is negative, it is recomputed as
        referred to :attr:`endex`.

        In case the rectified address would still be negative, an
        exception is raised.

        Arguments:
            address:
                Address to be rectified.

        Returns:
            int: Rectified address.

        Raises:
            IndexError: The rectified address would still be negative.
        """

        try:
            try:
                return <addr_t>address

            except OverflowError:
                return <addr_t>(<object>Memory_Endex(self._) + address)

        except OverflowError:
            raise IndexError('index out of range')

    def _rectify_span(
        self,
        start: Optional[Address],
        endex: Optional[Address],
    ) -> OpenInterval:
        r"""Rectifies an address span.

        In case a provided address is negative, it is recomputed as
        referred to :attr:`endex`.

        In case the rectified address would still be negative, it is
        clamped to address zero.

        Arguments:
            start (int):
                Inclusive start address for rectification.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for rectification.
                If ``None``, :attr:`endex` is considered.

        Returns:
            pair of int: Rectified address span.
        """

        endex_ = None

        if start is not None and start < 0:
            endex_ = Memory_Endex(self._)
            start = endex_ + start
            if start < 0:
                start = 0

        if endex is not None and endex < 0:
            if endex_ is None:
                endex_ = Memory_Endex(self._)
            endex = endex_ + endex
            if endex < 0:
                endex = 0

        return start, endex

    def block_span(
        self,
        address: Address,
    ) -> Tuple[Optional[Address], Optional[Address], Optional[Value]]:

        address = self._rectify_address(address)
        return super().block_span(address)

    def blocks(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Tuple[Address, memoryview]]:

        start, endex = self._rectify_span(start, endex)
        yield from super().blocks(start=start, endex=endex)

    def bound(
        self,
        start: Optional[Address],
        endex: Optional[Address],
    ) -> ClosedInterval:

        start, endex = self._rectify_span(start, endex)
        return super().bound(start, endex)

    def clear(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        start, endex = self._rectify_span(start=start, endex=endex)
        super().clear(start, endex)

    def clear_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        start, endex = self._rectify_span(start, endex)
        return super().clear_backup(start=start, endex=endex)

    def count(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> int:

        start, endex = self._rectify_span(start, endex)
        return super().count(item, start=start, endex=endex)

    def crop(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        start, endex = self._rectify_span(start, endex)
        super().crop(start=start, endex=endex)

    def crop_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Tuple[Optional[ImmutableMemory], Optional[ImmutableMemory]]:

        start, endex = self._rectify_span(start, endex)
        return super().crop_backup(start=start, endex=endex)

    def cut(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        bound: bool = True,
    ) -> ImmutableMemory:

        start, endex = self._rectify_span(start, endex)
        return super().cut(start=start, endex=endex)

    def delete(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        start, endex = self._rectify_span(start, endex)
        super().delete(start=start, endex=endex)

    def delete_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        start, endex = self._rectify_span(start, endex)
        return super().delete_backup(start=start, endex=endex)

    def equal_span(
        self,
        address: Address,
    ) -> Tuple[Optional[Address], Optional[Address], Optional[Value]]:

        address = self._rectify_address(address)
        return super().equal_span(address)

    def extract(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
        step: Optional[Address] = None,
        bound: bool = True,
    ) -> ImmutableMemory:

        start, endex = self._rectify_span(start, endex)
        return super().extract(start=start, endex=endex, pattern=pattern, step=step, bound=bound)

    def fill(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
    ) -> None:

        start, endex = self._rectify_span(start, endex)
        super().fill(start=start, endex=endex, pattern=pattern)

    def fill_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        start, endex = self._rectify_span(start, endex)
        return super().fill_backup(start=start, endex=endex)

    def find(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        start, endex = self._rectify_span(start, endex)
        return super().find(item, start=start, endex=endex)

    def flood(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
    ) -> None:

        start, endex = self._rectify_span(start, endex)
        super().flood(start=start, endex=endex, pattern=pattern)

    def flood_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> List[OpenInterval]:

        start, endex = self._rectify_span(start, endex)
        return super().flood_backup(start=start, endex=endex)

    @classmethod
    def from_blocks(
        cls,
        blocks: BlockSequence,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> bytesparse:
        cdef:
            Memory memory1
            bytesparse memory2
            Memory_* memory1_
            Memory_* memory2_

        if blocks:
            block_start = blocks[0][0]
            if block_start + offset < 0:
                raise ValueError('negative offseted start')

        if start is not None and start < 0:
            raise ValueError('negative start')
        if endex is not None and endex < 0:
            raise ValueError('negative endex')

        memory1 = super().from_blocks(blocks, offset=offset, start=start, endex=endex, copy=copy, validate=validate)
        memory2 = cls()
        memory1_ = memory1._
        memory2_ = memory2._

        memory2_.blocks = Rack_Free(memory2_.blocks)
        memory2_.blocks = memory1_.blocks
        memory1_.blocks = NULL

        memory2_.trim_start = memory1_.trim_start
        memory2_.trim_endex = memory1_.trim_endex
        memory2_.trim_start_ = memory1_.trim_start_
        memory2_.trim_endex_ = memory1_.trim_endex_
        return memory2

    @classmethod
    def from_bytes(
        cls,
        data: AnyBytes,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> bytesparse:
        cdef:
            Memory memory1
            bytesparse memory2
            Memory_* memory1_
            Memory_* memory2_

        if offset < 0:
            raise ValueError('negative offset')
        if start is not None and start < 0:
            raise ValueError('negative start')
        if endex is not None and endex < 0:
            raise ValueError('negative endex')

        memory1 = super().from_bytes(data, offset=offset, start=start, endex=endex, copy=copy, validate=validate)
        memory2 = cls()
        memory1_ = memory1._
        memory2_ = memory2._

        memory2_.blocks = Rack_Free(memory2_.blocks)
        memory2_.blocks = memory1_.blocks
        memory1_.blocks = NULL

        memory2_.trim_start = memory1_.trim_start
        memory2_.trim_endex = memory1_.trim_endex
        memory2_.trim_start_ = memory1_.trim_start_
        memory2_.trim_endex_ = memory1_.trim_endex_
        return memory2

    @classmethod
    def from_memory(
        cls,
        memory: ImmutableMemory,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> bytesparse:
        cdef:
            const Memory_* memory_
            Memory memory1
            bytesparse memory2
            Memory_* memory1_
            Memory_* memory2_
            addr_t block_start

        if isinstance(memory, Memory):
            memory_ = (<Memory>memory)._
            if Memory_Bool(memory_):
                if Memory_Start(memory_) + offset < 0:
                    raise ValueError('negative offseted start')
        else:
            if memory:
                if memory.start + offset < 0:
                    raise ValueError('negative offseted start')

        if start is not None and start < 0:
            raise ValueError('negative start')
        if endex is not None and endex < 0:
            raise ValueError('negative endex')

        memory1 = super().from_memory(memory, offset=offset, start=start, endex=endex, copy=copy, validate=validate)
        memory2 = cls()
        memory1_ = memory1._
        memory2_ = memory2._

        memory2_.blocks = Rack_Free(memory2_.blocks)
        memory2_.blocks = memory1_.blocks
        memory1_.blocks = NULL

        memory2_.trim_start = memory1_.trim_start
        memory2_.trim_endex = memory1_.trim_endex
        memory2_.trim_start_ = memory1_.trim_start_
        memory2_.trim_endex_ = memory1_.trim_endex_
        return memory2

    def gaps(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[OpenInterval]:

        start, endex = self._rectify_span(start, endex)
        yield from super().gaps(start=start, endex=endex)

    def get(
        self,
        address: Address,
        default: Optional[Value] = None,
    ) -> Optional[Value]:

        address = self._rectify_address(address)
        return super().get(address, default=default)

    def index(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        start, endex = self._rectify_span(start, endex)
        return super().index(item, start=start, endex=endex)

    def insert(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
    ) -> None:

        address = self._rectify_address(address)
        super().insert(address, data)

    def insert_backup(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
    ) -> Tuple[Address, ImmutableMemory]:

        address = self._rectify_address(address)
        return super().insert_backup(address, data)

    def intervals(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[ClosedInterval]:

        start, endex = self._rectify_span(start, endex)
        yield from super().intervals(start=start, endex=endex)

    def items(
        self,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Tuple[Address, Optional[Value]]]:

        endex_ = endex  # backup
        if endex is Ellipsis:
            endex = None
        start, endex = self._rectify_span(start, endex)
        if endex_ is Ellipsis:
            endex = endex_  # restore
        yield from super().items(start=start, endex=endex, pattern=pattern)

    def keys(
        self,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
    ) -> Iterator[Address]:

        endex_ = endex  # backup
        if endex is Ellipsis:
            endex = None
        start, endex = self._rectify_span(start, endex)
        if endex_ is Ellipsis:
            endex = endex_  # restore
        yield from super().keys(start=start, endex=endex)

    def ofind(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Optional[Address]:

        start, endex = self._rectify_span(start, endex)
        return super().ofind(item, start=start, endex=endex)

    def peek(
        self,
        address: Address,
    ) -> Optional[Value]:

        address = self._rectify_address(address)
        return super().peek(address)

    def poke(
        self,
        address: Address,
        item: Optional[Union[AnyBytes, Value]],
    ) -> None:

        address = self._rectify_address(address)
        super().poke(address, item)

    def poke_backup(
        self,
        address: Address,
    ) -> Tuple[Address, Optional[Value]]:

        address = self._rectify_address(address)
        return super().poke_backup(address)

    def pop(
        self,
        address: Optional[Address] = None,
        default: Optional[Value] = None,
    ) -> Optional[Value]:

        if address is not None:
            address = self._rectify_address(address)
        return super().pop(address=address, default=default)

    def pop_backup(
        self,
        address: Optional[Address] = None,
    ) -> Tuple[Address, Optional[Value]]:

        if address is not None:
            address = self._rectify_address(address)
        return super().pop_backup(address=address)

    def remove(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        start, endex = self._rectify_span(start, endex)
        super().remove(item, start=start, endex=endex)

    def remove_backup(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        start, endex = self._rectify_span(start, endex)
        return super().remove_backup(item, start=start, endex=endex)

    def reserve(
        self,
        address: Address,
        size: Address,
    ) -> None:

        address = self._rectify_address(address)
        super().reserve(address, size)

    def reserve_backup(
        self,
        address: Address,
        size: Address,
    ) -> Tuple[Address, ImmutableMemory]:

        address = self._rectify_address(address)
        return super().reserve_backup(address, size)

    def rfind(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        start, endex = self._rectify_span(start, endex)
        return super().rfind(item, start=start, endex=endex)

    def rindex(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        start, endex = self._rectify_span(start, endex)
        return super().rindex(item, start=start, endex=endex)

    def rofind(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Optional[Address]:

        start, endex = self._rectify_span(start, endex)
        return super().rofind(item, start=start, endex=endex)

    def rvalues(
        self,
        start: Optional[Union[Address, EllipsisType]] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Optional[Value]]:

        start_ = start  # backup
        if start is Ellipsis:
            start = None
        start, endex = self._rectify_span(start, endex)
        if start_ is Ellipsis:
            start = start_  # restore
        yield from super().rvalues(start=start, endex=endex, pattern=pattern)

    def setdefault(
        self,
        address: Address,
        default: Optional[Value] = None,
    ) -> Optional[Value]:

        address = self._rectify_address(address)
        return super().setdefault(address, default=default)

    def setdefault_backup(
        self,
        address: Address,
    ) -> Tuple[Address, Optional[Value]]:

        address = self._rectify_address(address)
        return super().setdefault_backup(address)

    def shift(
        self,
        offset: Address,
    ) -> None:
        cdef:
            const Memory_* memory = self._

        if not memory.trim_start_ and offset < 0:
            if Memory_Bool(memory):
                if Memory_Start(memory) + offset < 0:
                    raise ValueError('negative offseted start')

        super().shift(offset)

    def shift_backup(
        self,
        offset: Address,
    ) -> Tuple[Address, ImmutableMemory]:
        cdef:
            const Memory_* memory = self._

        if not memory.trim_start_ and offset < 0:
            if Memory_Bool(memory):
                if Memory_Start(memory) + offset < 0:
                    raise ValueError('negative offseted start')

        return super().shift_backup(offset)

    @property
    def trim_endex(
        self,
    ) -> Optional[Address]:

        # Copy-pasted from Memory, because I cannot figure out how to override properties
        return Memory_GetTrimEndex(self._)

    @trim_endex.setter
    def trim_endex(
        self,
        trim_endex: Optional[Address],
    ) -> None:

        if trim_endex is not None and trim_endex < 0:
            raise ValueError('negative endex')

        # Copy-pasted from Memory, because I cannot figure out how to override properties
        Memory_SetTrimEndex(self._, trim_endex)

    @property
    def trim_span(
        self,
    ) -> OpenInterval:

        # Copy-pasted from Memory, because I cannot figure out how to override properties
        return Memory_GetTrimSpan(self._)

    @trim_span.setter
    def trim_span(
        self,
        trim_span: OpenInterval,
    ) -> None:

        trim_start, trim_endex = trim_span
        if trim_start is not None and trim_start < 0:
            raise ValueError('negative start')
        if trim_endex is not None and trim_endex < 0:
            raise ValueError('negative endex')

        # Copy-pasted from Memory, because I cannot figure out how to override properties
        Memory_SetTrimSpan(self._, trim_span)

    @property
    def trim_start(
        self,
    ) -> Optional[Address]:

        # Copy-pasted from Memory, because I cannot figure out how to override properties
        return Memory_GetTrimStart(self._)

    @trim_start.setter
    def trim_start(
        self,
        trim_start: Optional[Address],
    ) -> None:

        if trim_start is not None and trim_start < 0:
            raise ValueError('negative start')

        # Copy-pasted from Memory, because I cannot figure out how to override properties
        Memory_SetTrimStart(self._, trim_start)

    def values(
        self,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Optional[Value]]:

        endex_ = endex  # backup
        if endex is Ellipsis:
            endex = None
        start, endex = self._rectify_span(start, endex)
        if endex_ is Ellipsis:
            endex = endex_  # restore
        yield from super().values(start=start, endex=endex, pattern=pattern)

    def view(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> memoryview:

        start, endex = self._rectify_span(start, endex)
        return super().view(start=start, endex=endex)

    def write(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
        clear: bool = False,
    ) -> None:

        address = self._rectify_address(address)
        super().write(address, data, clear=clear)

    def write_backup(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
        clear: bool = False,
    ) -> List[ImmutableMemory]:

        address = self._rectify_address(address)
        return super().write_backup(address, data, clear=clear)
