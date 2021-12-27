# cython: language_level = 3
# cython: embedsignature = True

# Copyright (c) 2020-2021, Andrea Zoppi.
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

from itertools import islice as _islice
from itertools import zip_longest as _zip_longest

STR_MAX_CONTENT_SIZE: Address = 1000


# =====================================================================================================================

cdef void* PyMem_Calloc(size_t nelem, size_t elsize, bint zero):
    cdef:
        void* ptr
        size_t total = nelem * elsize

    if CannotMulSizeU(nelem, elsize):
        return NULL  # overflow

    ptr = PyMem_Malloc(total)
    if ptr and zero:
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

    if size > SIZE_HMAX:
        raise OverflowError('size overflow')

    # Allocate as per request
    allocated = Upsize(0, size)
    if allocated > SIZE_HMAX:
        raise MemoryError()

    that = <Block_*>PyMem_Calloc(Block_HEADING + (allocated * sizeof(byte_t)), 1, zero)
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
                ptr = <Block_*>PyMem_Calloc(Block_HEADING + (allocated * sizeof(byte_t)), 1, zero)
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

    Accessing a block makes it read-only. Please ensure any views get
    disposed before trying to write to the memory block again.
    """

    def __cinit__(self):
        self._block = NULL

    def __dealloc__(self):
        if self._block:
            self._block = Block_Release(self._block)

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef:
            int CONTIGUOUS = PyBUF_C_CONTIGUOUS | PyBUF_F_CONTIGUOUS | PyBUF_ANY_CONTIGUOUS

        if flags & PyBUF_WRITABLE:
            raise ValueError('read only access')

        self.check_()

        # self._block = Block_Acquire(self._block)

        buffer.buf = &self._block.data[self._start]
        buffer.obj = self
        buffer.len = self._endex - self._start
        buffer.itemsize = 1
        buffer.readonly = 1
        buffer.ndim = 1
        buffer.format = <char*>'B' if flags & (PyBUF_FORMAT | CONTIGUOUS) else NULL
        buffer.shape = &buffer.len if flags & (PyBUF_ND | CONTIGUOUS) else NULL
        buffer.strides = &buffer.itemsize if flags & (PyBUF_STRIDES | CONTIGUOUS) else NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        # if self._block:
        #     self._block = Block_Release(self._block)
        pass

    def __repr__(
        self: 'BlockView',
    ) -> str:

        return repr(str(self))

    def __str__(
        self: 'BlockView',
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
            return self.memview.tobytes().decode('ascii')

    def __bool__(
        self: 'BlockView',
    ) -> bool:
        r"""Has any data.

        Returns:
            bool: Non-null slice length.
        """

        self.check_()
        return self._start < self._endex

    def __bytes__(
        self: 'BlockView',
    ) -> bytes:
        r"""Converts into bytes.

        Returns:
            bytes: :class:`bytes` clone of the viewed slice.
        """

        return bytes(self.memview)

    @property
    def memview(
        self: 'BlockView',
    ) -> memoryview:
        r"""memoryview: Python :class:`memoryview` wrapper."""

        self.check_()
        if self._memview is None:
            self._memview = memoryview(self)
        return self._memview

    def __len__(
        self: 'BlockView',
    ) -> Address:
        r"""int: Slice length."""

        self.check_()
        return self._endex - self._start

    def __getattr__(
        self: 'BlockView',
        attr: str,
    ) -> Any:

        return getattr(self.memview, attr)

    def __getitem__(
        self: 'BlockView',
        item: Any,
    ) -> Any:

        self.check_()
        return self.memview[item]

    @property
    def start(
        self: 'BlockView',
    ) -> Address:
        r"""int: Slice inclusive start address."""

        self.check()
        return self._block.address

    @property
    def endex(
        self: 'BlockView',
    ) -> Address:
        r"""int: Slice exclusive end address."""

        self.check()
        return self._block.address + self._endex - self._start

    @property
    def endin(
        self: 'BlockView',
    ) -> Address:
        r"""int: Slice inclusive end address."""

        return self.endex - 1

    @property
    def acquired(
        self: 'BlockView',
    ) -> bool:
        r"""bool: Underlying block currently acquired."""

        return self._block != NULL

    cdef bint check_(self) except -1:
        if self._block == NULL:
            raise RuntimeError('null internal data pointer')

    def check(
        self: 'BlockView',
    ) -> None:
        r"""Checks for data consistency."""

        self.check_()

    def dispose(
        self: 'BlockView',
    ) -> None:
        r"""Forces object disposal.

        Useful to make sure that any memory blocks are unreferenced before automatic
        garbage collection.

        Any access to the object after calling this function could raise exceptions.
        """

        if self._block:
            self._block = Block_Release(self._block)


# =====================================================================================================================

cdef Rack_* Rack_Alloc(size_t size) except NULL:
    cdef:
        Rack_* that = NULL
        size_t allocated

    if size > SIZE_HMAX:
        raise OverflowError('size overflow')

    # Allocate as per request
    allocated = Upsize(0, size)
    if allocated > SIZE_HMAX:
        raise MemoryError()

    that = <Rack_*>PyMem_Calloc(Rack_HEADING + (allocated * sizeof(Block_*)), 1, True)
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


cdef Rack_* Rack_ShallowCopy(const Rack_* other) except NULL:
    cdef:
        Rack_* that = Rack_Alloc(other.endex - other.start)
        size_t start1 = that.start
        size_t start2 = other.start
        size_t offset

    try:
        for offset in range(that.endex - that.start):
            that.blocks[start1 + offset] = Block_Acquire(other.blocks[start2 + offset])
        return that

    except:
        that = Rack_Free(that)
        raise


cdef Rack_* Rack_Copy(const Rack_* other) except NULL:
    cdef:
        Rack_* that = Rack_Alloc(other.endex - other.start)
        size_t start1 = that.start
        size_t start2 = other.start
        size_t offset

    try:
        for offset in range(that.endex - that.start):
            that.blocks[start1 + offset] = Block_Copy(other.blocks[start2 + offset])
        return that

    except:
        that = Rack_Free(that)
        raise


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
        size_t block_length
        const Block_* block1
        const Block_* block2

    if block_count != other.endex - other.start:
        return False

    for block_index in range(block_count):
        block1 = Rack_Get__(that, block_index)
        block2 = Rack_Get__(other, block_index)
        block_length = Block_Length(block1)

        if block1.address != block2.address:
            return False

        if not Block_Eq(block1, block2):
            return False

    return True


cdef Rack_* Rack_Reserve_(Rack_* that, size_t offset, size_t size) except NULL:
    cdef:
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
                ptr = <Rack_*>PyMem_Calloc(Rack_HEADING + (allocated * sizeof(Block_*)), 1, True)
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


cdef ssize_t Rack_IndexAt(const Rack_* that, addr_t address) except -2:
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


cdef ssize_t Rack_IndexStart(const Rack_* that, addr_t address) except -2:
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


cdef ssize_t Rack_IndexEndex(const Rack_* that, addr_t address) except -2:
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

cdef class Rover:
    r"""Memory iterator.

    Iterates over values stored within a :class:`Memory`.

    Arguments:
        memory (:class:`Memory`):
            Memory to iterate.

        start (int):
            Inclusive start address of the iterated range.

        endex (int):
            Exclusive end address of the iterated range.

        pattern (bytes):
            Pattern to fill emptiness.

        forward (bool):
            Forward iterator.

        infinite (bool):
            Infinite iterator.
    """

    def __cinit__(self):
        pass

    def __dealloc__(self):
        self.dispose_()

    def __init__(
        self,
        Memory memory not None,
        addr_t start,
        addr_t endex,
        object pattern,
        bint forward,
        bint infinite,
    ):
        cdef:
            Block_* block = NULL
            const byte_t[:] view
            addr_t offset

        if forward:
            if endex < start:
                endex = start
        else:
            if start > endex:
                start = endex

        if pattern is not None:
            if isinstance(pattern, int):
                self._pattern_value = <byte_t>pattern
                self._pattern_data = &self._pattern_value
                self._pattern_size = 1
            else:
                try:
                    view = pattern
                except TypeError:
                    view = bytes(pattern)
                self._pattern_view = view  # save references
                self._pattern_size = len(view)
                if self._pattern_size:
                    with cython.boundscheck(False):
                        self._pattern_data = &view[0]
                    if not forward:
                        self._pattern_offset = self._pattern_size - 1
                else:
                    raise ValueError('non-empty pattern required')

        self._forward = forward
        self._infinite = infinite
        self._start = start
        self._endex = endex
        self._address = start if forward else endex

        self._memory = memory  # keep reference
        self._blocks = memory._
        self._block_count = Rack_Length(self._blocks)

        if self._block_count:
            if forward:
                self._block_index = Rack_IndexStart(self._blocks, start)
                if self._block_index < self._block_count:
                    block = Rack_Get_(self._blocks, self._block_index)
                    self._block_start = Block_Start(block)
                    self._block_endex = Block_Endex(block)

                    offset = start if start >= self._block_start else self._block_start
                    if offset > self._block_endex:
                        offset = self._block_endex
                    offset -= self._block_start
                    CheckAddrToSizeU(offset)

                    block = Block_Acquire(block)
                    self._block = block
                    self._block_ptr = Block_At__(block, <size_t>offset)

            else:
                self._block_index = Rack_IndexEndex(self._blocks, endex)
                if self._block_index:
                    block = Rack_Get_(self._blocks, self._block_index - 1)
                    self._block_start = Block_Start(block)
                    self._block_endex = Block_Endex(block)

                    offset = endex if endex >= self._block_start else self._block_start
                    if offset > self._block_endex:
                        offset = self._block_endex
                    offset -= self._block_start
                    CheckAddrToSizeU(offset)

                    block = Block_Acquire(block)
                    self._block = block
                    self._block_ptr = Block_At__(block, <size_t>offset)

    def __len__(self):
        r"""Address range length.

        Returns:
            int: Address range length.
        """
        return self._endex - self._start

    cdef int next_(self) except -2:
        cdef:
            Block_* block
            int value = -1

        try:
            if self._forward:
                while True:  # loop to move to the next block when necessary
                    if self._address < self._endex:
                        if self._block_index < self._block_count:
                            if self._address < self._block_start:
                                self._address += 1
                                if self._pattern_size:
                                    value = <int><unsigned>self._pattern_data[self._pattern_offset]
                                else:
                                    value = -1
                                break

                            elif self._address < self._block_endex:
                                self._address += 1
                                value = self._block_ptr[0]
                                self._block_ptr += 1
                                break

                            else:
                                self._block_index += 1
                                if self._block_index < self._block_count:
                                    self._block = Block_Release(self._block)
                                    self._block = NULL
                                    block = Rack_Get_(self._blocks, self._block_index)
                                    block = Block_Acquire(block)
                                    self._block = block
                                    self._block_start = Block_Start(block)
                                    self._block_endex = Block_Endex(block)
                                    self._block_ptr = Block_At_(block, 0)
                                continue
                        else:
                            self._address += 1
                            if self._pattern_size:
                                value = <int><unsigned>self._pattern_data[self._pattern_offset]
                            else:
                                value = -1
                            break

                    elif self._infinite:
                        if self._pattern_size:
                            value = <int><unsigned>self._pattern_data[self._pattern_offset]
                        else:
                            value = -1

                    else:
                        raise StopIteration()
            else:
                while True:  # loop to move to the next block when necessary
                    if self._address > self._start:
                        if self._block_index:
                            if self._address > self._block_endex:
                                self._address -= 1
                                if self._pattern_size:
                                    value = <int><unsigned>self._pattern_data[self._pattern_offset]
                                else:
                                    value = -1
                                break

                            elif self._address > self._block_start:
                                self._address -= 1
                                self._block_ptr -= 1
                                value = self._block_ptr[0]
                                break

                            else:
                                self._block_index -= 1
                                if self._block_index:
                                    self._block = Block_Release(self._block)
                                    self._block = NULL
                                    block = Rack_Get_(self._blocks, self._block_index - 1)
                                    block = Block_Acquire(block)
                                    self._block = block
                                    self._block_start = Block_Start(block)
                                    self._block_endex = Block_Endex(block)
                                    self._block_ptr = Block_At__(block, Block_Length(block))
                                value = -1
                                continue
                        else:
                            self._address -= 1
                            if self._pattern_size:
                                value = <int><unsigned>self._pattern_data[self._pattern_offset]
                            else:
                                value = -1
                            break

                    elif self._infinite:
                        if self._pattern_size:
                            value = <int><unsigned>self._pattern_data[self._pattern_offset]
                        else:
                            value = -1

                    else:
                        raise StopIteration()

            if self._pattern_size:
                if self._forward:
                    if self._pattern_offset < self._pattern_size - 1:
                        self._pattern_offset += 1
                    else:
                        self._pattern_offset = 0
                else:
                    if self._pattern_offset > 0:
                        self._pattern_offset -= 1
                    else:
                        self._pattern_offset = self._pattern_size - 1

            return value

        except:
            self._block = Block_Release(self._block)  # preempt
            raise

    def __next__(self):
        r"""Next iterated value.

        Returns:
            int: Byte value at the current address; ``None`` within emptiness.
        """
        cdef:
            int value

        value = self.next_()
        return None if value < 0 else value

    def __iter__(self):
        r"""Values iterator.

        Yields:
            int: Byte value at the current address; ``None`` within emptiness.
        """
        cdef:
            int value

        while True:
            value = self.next_()
            yield None if value < 0 else value

    cdef vint dispose_(self) except -1:
        self._address = self._endex if self._forward else self._start
        self._block = Block_Release(self._block)
        self._memory = None

    def dispose(self):
        r"""Forces object disposal.

        Useful to make sure that any memory blocks are unreferenced before automatic
        garbage collection.

        Any access to the object after calling this function could raise exceptions.
        """
        self.dispose_()

    @property
    def forward(self) -> bool:
        r"""bool: Forward iterator."""
        return self._forward

    @property
    def infinite(self) -> bool:
        r"""bool: Infinite iterator."""
        return self._infinite

    @property
    def address(self) -> Address:
        r"""int: Current address being iterated."""
        return self._address

    @property
    def start(self) -> Address:
        r"""int: Inclusive start address of the iterated range."""
        return self._start

    @property
    def endex(self) -> Address:
        r"""int: Exclusive end address of the iterated range."""
        return self._endex


# ---------------------------------------------------------------------------------------------------------------------

cdef class Memory:
    r"""Virtual memory.

    This class is a handy wrapper around `blocks`, so that it can behave mostly
    like a :obj:`bytearray`, but on sparse chunks of data.

    Please look at examples of each method to get a glimpse of the features of
    this class.

    On creation, at most one of `memory`, `blocks`, or `data` can be specified.

    The Cython implementation limits the address range to that of the integral
    type ``uint_fast64_t``.

    Arguments:
        memory (Memory):
            An optional :obj:`Memory` to copy data from.

        data (bytes):
            An optional :obj:`bytes` string to create a single block of data.

        offset (int):
            Start address of the initial block, built if `data` is given.

        blocks (list of blocks):
            A sequence of non-overlapping blocks, sorted by address.

        start (int):
            Optional memory start address.
            Anything before will be trimmed away.

        endex (int):
            Optional memory exclusive end address.
            Anything at or after it will be trimmed away.

        copy (bool):
            Forces copy of provided input data.

        validate (bool):
            Validates the resulting :obj:`Memory` object.

    Raises:
        :obj:`ValueError`: More than one of `memory`, `data`, and `blocks`.

    Examples:
        >>> memory = Memory()
        >>> memory._blocks
        []

        >>> memory = Memory(data=b'Hello, World!', offset=5)
        >>> memory._blocks
        [[5, b'Hello, World!']]
    """

    def __cinit__(self):
        r"""Cython constructor."""
        self._ = NULL
        self._trim_start = 0
        self._trim_endex = ADDR_MAX

    def __dealloc__(self):
        r"""Cython deallocation method."""
        self._ = Rack_Free(self._)

    def __init__(
        self: 'Memory',
        memory: Optional['Memory'] = None,
        data: Optional[AnyBytes] = None,
        offset: Address = 0,
        blocks: Optional[BlockList] = None,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ):
        cdef:
            Memory memory_
            addr_t start_
            addr_t endex_
            addr_t address
            size_t size
            const byte_t[:] view
            const byte_t* ptr = NULL
            Block_* block = NULL

        if (memory is not None) + (data is not None) + (blocks is not None) > 1:
            raise ValueError('only one of [memory, data, blocks] is allowed')

        start_ = 0 if start is None else <addr_t>start
        endex_ = ADDR_MAX if endex is None else <addr_t>endex
        if endex_ < start_:
            endex_ = start_  # clamp negative length

        if memory is not None:
            memory_ = <Memory>memory

            if copy or offset:
                self._ = Rack_Copy(memory_._)
                self._ = Rack_Shift(self._, offset)
            else:
                self._ = Rack_ShallowCopy(memory_._)

        elif data is not None:
            if offset < 0:
                raise ValueError('negative offset')

            address = <addr_t>offset
            size = <size_t>len(data)
            self._ = Rack_Alloc(0)

            if size:
                view = data
                with cython.boundscheck(False):
                    ptr = &view[0]
                block = Block_Create(address, size, ptr)
                try:
                    self._ = Rack_Append(self._, block)
                except:
                    block = Block_Free(block)
                    raise

        elif blocks:
            self._ = Rack_FromObject(blocks, offset)

        else:
            self._ = Rack_Alloc(0)

        self._trim_start = start_
        self._trim_endex = endex_
        self._trim_start_ = start is not None
        self._trim_endex_ = endex is not None

        self._crop_(start_, endex_, None)

        if validate:
            self.validate()

    def __repr__(
        self: 'Memory',
    ) -> str:
        cdef:
            addr_t start = self.start_()
            addr_t endex = self.endex_()

        return f'<{type(self).__name__}[0x{start:X}:0x{endex:X}]@0x{id(self):X}>'

    def __str__(
        self: 'Memory',
    ) -> str:
        r"""String representation.

        If :attr:`content_size` is lesser than ``STR_MAX_CONTENT_SIZE``, then
        the memory is represented as a list of blocks.

        If exceeding, it is equivalent to :meth:`__repr__`.


        Returns:
            str: String representation.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [7, b'xyz']])
            >>> memory._blocks
            'ABCxyz'
        """
        cdef:
            addr_t size = self.content_size_()
            addr_t start
            addr_t endex

        if size > STR_MAX_CONTENT_SIZE:
            start = self.start_()
            endex = self.endex_()
            return f'<{type(self).__name__}[0x{start:X}:0x{endex:X}]@0x{id(self):X}>'

        else:
            return str(self._to_blocks())

    def __bool__(
        self: 'Memory',
    ) -> bool:
        r"""Has any items.

        Returns:
            bool: Has any items.

        Examples:
            >>> memory = Memory()
            >>> bool(memory)
            False

            >>> memory = Memory(data=b'Hello, World!', offset=5)
            >>> bool(memory)
            True
        """

        return Rack_Length(self._) > 0

    cdef bint __eq__same_(self, Memory other) except -1:
        return Rack_Eq(self._, (<Memory>other)._)

    cdef bint __eq__raw_(self, size_t data_size, const byte_t* data_ptr) except -1:
        cdef:
            const Rack_* blocks = self._
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

    cdef bint __eq__view_(self, const byte_t[:] view) except -1:
        with cython.boundscheck(False):
            return self.__eq__raw_(len(view), &view[0])

    cdef bint __eq__iter_(self, iterable) except -1:
        iter_self = _islice(self, len(self))  # avoid infinite loop
        iter_other = iter(iterable)
        return all(a == b for a, b in _zip_longest(iter_self, iter_other, fillvalue=None))

    def __eq__(
        self: 'Memory',
        other: Any,
    ) -> bool:
        r"""Equality comparison.

        Arguments:
            other (Memory):
                Data to compare with `self`.

                If it is a :obj:`Memory`, all of its blocks must match.

                If it is a :obj:`list`, it is expected that it contains the
                same blocks as `self`.

                Otherwise, it must match the first stored block, considered
                equal if also starts at 0.

        Returns:
            bool: `self` is equal to `other`.

        Examples:
            >>> data = b'Hello, World!'
            >>> memory = Memory(data=data)
            >>> memory == data
            True
            >>> memory.shift(1)
            >>> memory == data
            False

            >>> data = b'Hello, World!'
            >>> memory = Memory(data=data)
            >>> memory == [[0, data]]
            True
            >>> memory == list(data)
            False
            >>> memory.shift(1)
            >>> memory == [[0, data]]
            False
        """
        cdef:
            const byte_t[:] view

        if isinstance(other, Memory):
            return self.__eq__same_(other)
        else:
            try:
                view = other
            except TypeError:
                return self.__eq__iter_(other)
            else:
                return self.__eq__view_(other)

    def __iter__(
        self: 'Memory',
    ) -> Iterator[Optional[Value]]:
        r"""Iterates over values.

        Iterates over values between :attr:`start` and :attr:`endex`.

        Yields:
            int: Value as byte integer, or ``None``.
        """

        yield from self.values()

    def __reversed__(
        self: 'Memory',
    ) -> Iterator[Optional[Value]]:
        r"""Iterates over values, reversed order.

        Iterates over values between :attr:`start` and :attr:`endex`, in
        reversed order.

        Yields:
            int: Value as byte integer, or ``None``.
        """

        yield from self.rvalues()

    def __add__(
        self: 'Memory',
        value: Union[AnyBytes, 'Memory'],
    ) -> 'Memory':

        memory = self.copy_()
        memory.extend(value)
        return memory

    def __iadd__(
        self: 'Memory',
        value: Union[AnyBytes, 'Memory'],
    ) -> 'Memory':

        self.extend(value)
        return self

    def __mul__(
        self: 'Memory',
        times: int,
    ) -> 'Memory':
        cdef:
            Memory memory
            addr_t offset

        times = int(times)
        if times < 0:
            times = 0

        if times and Rack_Length(self._):
            start = self.start
            size = self.endex - start
            offset = size  # adjust first write
            memory = self.__deepcopy__()

            for time in range(times - 1):
                memory.write_same_(offset, self, False, None)
                offset += size

            return memory
        else:
            return Memory()

    def __imul__(
        self: 'Memory',
        times: int,
    ) -> 'Memory':
        cdef:
            Memory memory
            addr_t offset

        times = int(times)
        if times < 0:
            times = 0

        if times and Rack_Length(self._):
            start = self.start
            size = self.endex - start
            offset = size
            memory = self.__deepcopy__()

            for time in range(times - 1):
                self.write_same_(offset, memory, False, None)
                offset += size
        else:
            self._ = Rack_Clear(self._)
        return self

    def __len__(
        self: 'Memory',
    ) -> Address:
        r"""Actual length.

        Computes the actual length of the stored items, i.e.
        (:attr:`endex` - :attr:`start`).
        This will consider any trimmings being active.

        Returns:
            int: Memory length.
        """

        return self.endex_() - self.start_()

    def ofind(
        self: 'Memory',
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

        offset = self.find(item, start, endex)
        if offset >= 0:
            return offset
        else:
            return None

    def rofind(
        self: 'Memory',
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

        offset = self.rfind(item, start, endex)
        if offset >= 0:
            return offset
        else:
            return None

    cdef saddr_t find_unbounded_(self, size_t size, const byte_t* buffer) except -2:
        cdef:
            const Rack_* blocks = self._
            size_t block_index
            const Block_* block
            ssize_t offset

        if size:
            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                offset = Block_Find_(block, 0, SIZE_MAX, size, buffer)
                if offset >= 0:
                    return Block_Start(block) + <size_t>offset
        return -1

    cdef saddr_t find_bounded_(self, size_t size, const byte_t* buffer, addr_t start, addr_t endex) except -2:
        cdef:
            const Rack_* blocks = self._
            size_t block_index
            const Block_* block
            ssize_t offset
            size_t block_index_start
            size_t block_index_endex
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
                offset = Block_Find_(block, slice_start, slice_endex, size, buffer)
                if offset >= 0:
                    return Block_Start(block) + <size_t>offset
        return -1

    def find(
        self: 'Memory',
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
            return self.find_unbounded_(item_size, item_ptr)

        # Bounded slice
        start_, endex_ = self.bound_(start, endex)
        return self.find_bounded_(item_size, item_ptr, start_, endex_)

    cdef saddr_t rfind_unbounded_(self, size_t size, const byte_t* buffer) except -2:
        cdef:
            const Rack_* blocks = self._
            size_t block_index
            const Block_* block
            ssize_t offset

        if size:
            for block_index in range(Rack_Length(blocks), 0, -1):
                block = Rack_Get__(blocks, block_index - 1)
                offset = Block_ReverseFind_(block, 0, SIZE_MAX, size, buffer)
                if offset >= 0:
                    return Block_Start(block) + <size_t>offset
        return -1

    cdef saddr_t rfind_bounded_(self, size_t size, const byte_t* buffer, addr_t start, addr_t endex) except -2:
        cdef:
            const Rack_* blocks = self._
            size_t block_index
            const Block_* block
            ssize_t offset
            size_t block_index_start
            size_t block_index_endex
            size_t slice_start
            size_t slice_endex

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
                    return Block_Start(block) + <size_t>offset
        return -1

    def rfind(
        self: 'Memory',
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
            return self.rfind_unbounded_(item_size, item_ptr)

        # Bounded slice
        start_, endex_ = self.bound_(start, endex)
        return self.rfind_bounded_(item_size, item_ptr, start_, endex_)

    def index(
        self: 'Memory',
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

        offset = self.find(item, start, endex)
        if offset >= 0:
            return offset
        else:
            raise ValueError('subsection not found')

    def rindex(
        self: 'Memory',
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

        offset = self.rfind(item, start, endex)
        if offset >= 0:
            return offset
        else:
            raise ValueError('subsection not found')

    def __contains__(
        self: 'Memory',
        item: Union[AnyBytes, Value],
    ) -> bool:
        r"""Checks if some items are contained.

        Arguments:
            item (items):
                Items to find. Can be either some byte string or an integer.

        Returns:
            bool: Item is contained.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[1 | 2 | 3]|   |[x | y | z]|
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'123'], [9, b'xyz']])
            >>> b'23' in memory
            True
            >>> ord('y') in memory
            True
            >>> b'$' in memory
            False
        """

        return self.find(item) >= 0

    cdef addr_t count_unbounded_(self, size_t size, const byte_t* buffer) except -1:
        cdef:
            const Rack_* blocks = self._
            size_t block_index
            const Block_* block
            addr_t count = 0

        if size:
            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                count += Block_Count_(block, 0, SIZE_MAX, size, buffer)
        return count

    cdef addr_t count_bounded_(self, size_t size, const byte_t* buffer, addr_t start, addr_t endex) except -1:
        cdef:
            const Rack_* blocks = self._
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

    def count(
        self: 'Memory',
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

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[B | a | t]|   |[t | a | b]|
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'Bat'], [9, b'tab']])
            >>> memory.count(b'a')
            2
        """
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
            return self.count_unbounded_(item_size, item_ptr)

        # Bounded slice
        start_, endex_ = self.bound_(start, endex)
        return self.count_bounded_(item_size, item_ptr, start_, endex_)

    def __getitem__(
        self: 'Memory',
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
            This method is not optimized for a :class:`slice` where its `step`
            is an integer greater than 1.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|
            +---+---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67| 68|   | 36|   |120|121|122|
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory[9]  # -> ord('y') = 121
            121
            >>> memory[:3]._blocks
            [[1, b'AB']]
            >>> memory[3:10]._blocks
            [[3, b'CD'], [6, b'$'], [8, b'xy']]
            >>> bytes(memory[3:10:b'.'])
            b'CD.$.xy'
            >>> memory[memory.endex]
            None
            >>> bytes(memory[3:10:3])
            b'C$y'
            >>> memory[3:10:2]._blocks
            [[3, b'C'], [6, b'y']]
            >>> bytes(memory[3:10:2])
            Traceback (most recent call last):
                ...
            ValueError: non-contiguous data within range
        """
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
            start = self.start_() if key_start is None else <addr_t>key_start
            endex = self.endex_() if key_endex is None else <addr_t>key_endex
            key_step = key_.step

            if key_step is None or key_step is 1 or key_step == 1:
                return self.extract_(start, endex, 0, NULL, 1, True)

            elif isinstance(key_step, int):
                if key_step > 1:
                    return self.extract_(start, endex, 0, NULL, <saddr_t>key_step, True)
                else:
                    return Memory()  # empty

            else:
                pattern = Block_FromObject(0, key_step, True)
                try:
                    memory = self.extract_(start, endex, Block_Length(pattern), Block_At__(pattern, 0), 1, True)
                finally:
                    Block_Free(pattern)  # orphan
                return memory
        else:
            value = self.peek_(<addr_t>key)
            return None if value < 0 else value

    def __setitem__(
        self: 'Memory',
        key: Union[Address, slice],
        value: Optional[Union[AnyBytes, Value]],
    ) -> None:
        r"""Sets data.

        Arguments:
            key (slice or int):
                Selection range or address.

            value (items):
                Items to write at the selection address.
                If `value` is null, the range is cleared.

        Note:
            This method is not optimized for a :class:`slice` where its `step`
            is an integer greater than 1.

        Examples:
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

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']])
            >>> memory[7:10] = None
            >>> memory._blocks
            [[5, b'AB'], [10, b'yz']]
            >>> memory[7] = b'C'
            >>> memory[9] = b'x'
            >>> memory._blocks == [[5, b'ABC'], [9, b'xyz']]
            True
            >>> memory[6:12:3] = None
            >>> memory._blocks
            [[5, b'A'], [7, b'C'], [10, b'yz']]
            >>> memory[6:13:3] = b'123'
            >>> memory._blocks
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

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']])
            >>> memory[0:4] = b'$'
            >>> memory._blocks
            [[0, b'$'], [2, b'ABC'], [6, b'xyz']]
            >>> memory[4:7] = b'45678'
            >>> memory._blocks
            [[0, b'$'], [2, b'AB45678yz']]
            >>> memory[6:8] = b'<>'
            >>> memory._blocks
            [[0, b'$'], [2, b'AB45<>8yz']]
        """
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
            start = self.start_() if key_start is None else <addr_t>key_start
            endex = self.endex_() if key_endex is None else <addr_t>key_endex
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
                    self._erase_(start, endex, False, False)  # clear
                else:
                    address = start
                    while address < endex:
                        self._erase_(address, address + 1, False, False)  # clear
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
                    if not step or not value_size:
                        if CannotAddAddrU(start, value_size):
                            del_start = ADDR_MAX
                        else:
                            del_start = start + value_size
                        if CannotAddAddrU(del_start, (slice_size - value_size)):
                            del_endex = ADDR_MAX
                        else:
                            del_endex = del_start + (slice_size - value_size)
                        self._erase_(del_start, del_endex, True, True)  # delete
                        if value_size:
                            self.write_raw_(start, value_size, Block_At__(value_, 0), None)
                    else:
                        raise ValueError(f'attempt to assign bytes of size {value_size}'
                                         f' to extended slice of size {slice_size}')
                elif slice_size < value_size:
                    # Enlarge: insert excess, overwrite existing
                    if not step:
                        self.insert_raw_(endex, value_size - slice_size, Block_At__(value_, slice_size), None)
                        self.write_raw_(start, slice_size, Block_At__(value_, 0), None)
                    else:
                        raise ValueError(f'attempt to assign bytes of size {value_size}'
                                         f' to extended slice of size {slice_size}')
                else:
                    # Same size: overwrite existing
                    if not step:
                        self.write_raw_(start, value_size, Block_At__(value_, 0), None)
                    else:
                        CheckMulAddrU(step, value_size)
                        CheckAddAddrU(start, step * value_size)
                        for offset in range(value_size):
                            self.poke_(start + (step * offset), Block_Get__(value_, offset))
            finally:
                Block_Free(value_)  # orphan
        else:
            # below: self.poke(key, value)
            address = <addr_t>key
            if value is None:
                self.poke_none__(address)
            else:
                if isinstance(value, int):
                    self.poke_(address, <byte_t>value)
                else:
                    if len(value) != 1:
                        raise ValueError('expecting single item')
                    self.poke_(address, <byte_t>value[0])

    def __delitem__(
        self: 'Memory',
        key: Union[Address, slice],
    ) -> None:
        r"""Deletes data.

        Arguments:
            key (slice or int):
                Deletion range or address.

        Note:
            This method is not optimized for a :class:`slice` where its `step`
            is an integer greater than 1.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | y | z]|   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> del memory[4:9]
            >>> memory._blocks
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

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> del memory[9]
            >>> memory._blocks
            [[1, b'ABCD'], [6, b'$'], [8, b'xz']]
            >>> del memory[3]
            >>> memory._blocks
            [[1, b'ABD'], [5, b'$'], [7, b'xz']]
            >>> del memory[2:10:3]
            >>> memory._blocks
            [[1, b'AD'], [5, b'x']]
        """
        cdef:
            slice key_
            addr_t start
            addr_t endex
            addr_t step
            addr_t address

        if Rack_Length(self._):
            if isinstance(key, slice):
                key_ = <slice>key
                key_start = key_.start
                key_endex = key_.stop
                start = self.start_() if key_start is None else <addr_t>key_start
                endex = self.endex_() if key_endex is None else <addr_t>key_endex

                if start < endex:
                    key_step = key_.step
                    if key_step is None or key_step is 1 or key_step == 1:
                        self._erase_(start, endex, True, True)  # delete

                    elif key_step > 1:
                        step = <addr_t>key_step - 1
                        address = start
                        while address < endex:
                            self._erase_(address, address + 1, True, True)  # delete
                            address += step
                            endex -= 1
            else:
                address = <addr_t>key
                self._erase_(address, address + 1, True, True)  # delete

    cdef vint append_(self, byte_t value) except -1:
        cdef:
            Rack_* blocks = self._
            size_t block_count
            Block_* block

        block_count = Rack_Length(blocks)
        if block_count:
            block = Block_Append(Rack_Last_(blocks), value)
            Rack_Set__(blocks, block_count - 1, block)  # update pointer
        else:
            block = Block_Create(0, 1, &value)
            try:
                self._ = blocks = Rack_Append(blocks, block)
            except:
                Block_Free(block)  # orphan
                raise

    def append(
        self: 'Memory',
        item: Union[AnyBytes, Value],
    ) -> None:
        r"""Appends a single item.

        Arguments:
            item (int):
                Value to append. Can be a single byte string or integer.

        Examples:
            >>> memory = Memory()
            >>> memory.append(b'$')
            >>> memory._blocks
            [[0, b'$']]

            ~~~

            >>> memory = Memory()
            >>> memory.append(3)
            >>> memory._blocks
            [[0, b'\x03']]
        """

        if isinstance(item, int):
            self.append_(<byte_t>item)
        else:
            if len(item) != 1:
                raise ValueError('expecting single item')
            self.append_(<byte_t>item[0])

    cdef vint extend_same_(self, Memory items, addr_t offset) except -1:
        cdef:
            addr_t content_endex = self.content_endex_()

        CheckAddAddrU(content_endex, offset)
        offset += content_endex
        self.write_same_(offset, items, False, None)

    cdef vint extend_raw_(self, size_t items_size, const byte_t* items_ptr, addr_t offset) except -1:
        cdef:
            addr_t content_endex = self.content_endex_()

        CheckAddAddrU(content_endex, offset)
        offset += content_endex
        CheckAddAddrU(offset, items_size)
        self.write_raw_(offset, items_size, items_ptr, None)

    def extend(
        self: 'Memory',
        items: Union[AnyBytes, 'Memory'],
        offset: Address = 0,
    ) -> None:
        r"""Concatenates items.

        Equivalent to ``self += items``.

        Arguments:
            items (items):
                Items to append at the end of the current virtual space.

                If a :obj:`list`, it is interpreted as a sequence of
                non-overlapping blocks, sorted by start address.

            offset (int):
                Optional offset w.r.t. :attr:`content_endex`.
        """
        cdef:
            const byte_t[:] items_view
            byte_t items_value
            size_t items_size
            const byte_t* items_ptr

        if offset < 0:
            raise ValueError('negative extension offset')

        if isinstance(items, Memory):
            self.extend_same_(items, <addr_t>offset)
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

            self.extend_raw_(items_size, items_ptr, <addr_t>offset)

    cdef int pop_last_(self) except -2:
        cdef:
            Rack_* blocks = self._
            size_t block_count = Rack_Length(blocks)
            Block_* block
            byte_t backup

        if block_count:
            block = Rack_Last_(blocks)
            if Block_Length(block) > 1:
                block = Block_Pop__(block, &backup)
                Rack_Set__(blocks, block_count - 1, block)  # update pointer
            else:
                backup = Block_Get__(block, 0)
                self._ = blocks = Rack_Pop__(blocks, NULL)
            return backup
        else:
            return -1

    cdef int pop_at_(self, addr_t address) except -2:
        cdef:
            int backup

        backup = self.peek_(address)
        self._erase_(address, address + 1, True, True)  # delete
        return backup

    def pop(
        self: 'Memory',
        address: Optional[Address] = None,
    ) -> Optional[Value]:
        r"""Takes a value away.

        Arguments:
            address (int):
                Address of the byte to pop.
                If ``None``, the very last byte is popped.

        Return:
            int: Value at `address`; ``None`` within emptiness.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | D]|   |[$]|   |[x | y]|   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | D]|   |[$]|   |[x | y]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.pop()  # -> ord('z') = 122
            122
            >>> memory.pop(3)  # -> ord('C') = 67
            67
        """
        cdef:
            int value

        if address is None:
            value = self.pop_last_()
        else:
            value = self.pop_at_(<addr_t>address)
        return None if value < 0 else value

    cdef BlockView _memview(self):
        cdef:
            Rack_* blocks = self._
            size_t block_count = Rack_Length(blocks)
            addr_t start
            addr_t endex

        if not block_count:
            start = self._trim_start
            endex = self._trim_endex
            if self._trim_start_ and self._trim_endex_ and start < endex - 1:
                raise ValueError('non-contiguous data within range')
            return Block_View(Block_Alloc(start, 0, False))

        elif block_count == 1:
            start = self._trim_start
            if self._trim_start_:
                if start != Block_Start(Rack_First__(blocks)):
                    raise ValueError('non-contiguous data within range')

            endex = self._trim_endex
            if self._trim_endex_:
                if endex != Block_Endex(Rack_Last__(blocks)):
                    raise ValueError('non-contiguous data within range')

            return Block_View(Rack_First_(blocks))

        else:
            raise ValueError('non-contiguous data within range')

    def __bytes__(
        self: 'Memory',
    ) -> bytes:
        r"""Creates a bytes clone.

        Returns:
            :obj:`bytes`: Cloned data.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).
        """

        return bytes(self._memview())

    def to_bytes(
        self: 'Memory',
    ) -> bytes:
        r"""Creates a bytes clone.

        Returns:
            :obj:`bytes`: Cloned data.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).
        """

        return bytes(self._memview())

    def to_bytearray(
        self: 'Memory',
    ) -> bytearray:
        r"""Creates a bytearray clone.

        Arguments:
            copy (bool):
                Creates a clone of the underlying :obj:`bytearray` data
                structure.

        Returns:
            :obj:`bytearray`: Cloned data.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).
        """

        return bytearray(self._memview())

    def to_memoryview(
        self: 'Memory',
    ) -> memoryview:
        r"""Creates a memory view.

        Returns:
            :obj:`memoryview`: View over data.

        Raises:
            :obj:`ValueError`: Data not contiguous (see :attr:`contiguous`).
        """

        return self._memview()

    cdef Memory copy_(self):
        cdef:
            Memory memory = Memory()

        memory._ = Rack_Free(memory._)
        memory._ = Rack_Copy(self._)

        memory._trim_start = self._trim_start
        memory._trim_endex = self._trim_endex
        memory._trim_start_ = self._trim_start_
        memory._trim_endex_ = self._trim_endex_

        return memory

    def __copy__(
        self: 'Memory',
    ) -> 'Memory':
        r"""Creates a shallow copy.

        Note:
            The Cython implementation actually creates a deep copy.

        Returns:
            :obj:`Memory`: Shallow copy.
        """

        return self.copy_()

    def __deepcopy__(
        self: 'Memory',
    ) -> 'Memory':
        r"""Creates a deep copy.

        Returns:
            :obj:`Memory`: Deep copy.
        """

        return self.copy_()

    @property
    def contiguous(
        self: 'Memory',
    ) -> bool:
        r"""bool: Contains contiguous data.

        The memory is considered to have contiguous data if there is no empty
        space between blocks.

        If trimming is defined, there must be no empty space also towards it.
        """

        try:
            self._memview()
            return True
        except ValueError:
            return False

    @property
    def trim_start(
        self: 'Memory',
    ) -> Optional[Address]:
        r"""int: Trimming start address.

        Any data before this address is automatically discarded.
        Disabled if ``None``.
        """

        return self._trim_start if self._trim_start_ else None

    @trim_start.setter
    def trim_start(
        self: 'Memory',
        trim_start: Address,
    ) -> None:
        cdef:
            addr_t trim_start_
            addr_t trim_endex_

        if trim_start is None:
            trim_start_ = 0
            self._trim_start_ = False
        else:
            trim_start_ = <addr_t>trim_start
            self._trim_start_ = True

        trim_endex_ = self._trim_endex
        if self._trim_start_ and self._trim_endex_ and trim_endex_ < trim_start_:
            self._trim_endex = trim_endex_ = trim_start_

        self._trim_start = trim_start_
        if self._trim_start_:
            self._crop_(trim_start_, trim_endex_, None)

    @property
    def trim_endex(
        self: 'Memory',
    ) -> Optional[Address]:
        r"""int: Trimming exclusive end address.

        Any data at or after this address is automatically discarded.
        Disabled if ``None``.
        """

        return self._trim_endex if self._trim_endex_ else None

    @trim_endex.setter
    def trim_endex(
        self: 'Memory',
        trim_endex: Address,
    ) -> None:
        cdef:
            addr_t trim_start_
            addr_t trim_endex_

        if trim_endex is None:
            trim_endex_ = ADDR_MAX
            self._trim_endex_ = False
        else:
            trim_endex_ = <addr_t>trim_endex
            self._trim_endex_ = True

        trim_start_ = self._trim_start
        if self._trim_start_ and self._trim_endex_ and trim_endex_ < trim_start_:
            self._trim_start = trim_start_ = trim_endex_

        self._trim_endex = trim_endex_
        if self._trim_endex_:
            self._crop_(trim_start_, trim_endex_, None)

    @property
    def trim_span(
        self: 'Memory',
    ) -> OpenInterval:
        r"""tuple of int: Trimming span addresses.

        A :obj:`tuple` holding :attr:`trim_start` and :attr:`trim_endex`.
        """

        return (self._trim_start if self._trim_start_ else None,
                self._trim_endex if self._trim_endex_ else None)

    @trim_span.setter
    def trim_span(
        self: 'Memory',
        span: OpenInterval,
    ) -> None:

        trim_start, trim_endex = span

        if trim_start is None:
            trim_start_ = 0
            self._trim_start_ = False
        else:
            trim_start_ = <addr_t>trim_start
            self._trim_start_ = True

        if trim_endex is None:
            trim_endex_ = ADDR_MAX
            self._trim_endex_ = False
        else:
            trim_endex_ = <addr_t>trim_endex
            self._trim_endex_ = True

        if self._trim_start_ and self._trim_endex_ and trim_endex_ < trim_start_:
            trim_endex_ = trim_start_

        self._trim_start = trim_start_
        self._trim_endex = trim_endex_
        if self._trim_start_ or self._trim_endex_:
            self._crop_(trim_start_, trim_endex_, None)

    cdef addr_t start_(self):
        cdef:
            const Rack_* blocks

        if not self._trim_start_:
            # Return actual
            blocks = self._
            if Rack_Length(blocks):
                return Block_Start(Rack_First__(blocks))
            else:
                return 0
        else:
            return self._trim_start

    @property
    def start(
        self: 'Memory',
    ) -> Address:
        r"""int: Inclusive start address.

        This property holds the inclusive start address of the virtual space.
        By default, it is the current minimum inclusive start address of
        the first stored block.

        If :attr:`trim_start` not ``None``, that is returned.

        If the memory has no data and no trimming, 0 is returned.

        Examples:
            >>> Memory().start
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.start
            1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[[[|   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'xyz']], start=1)
            >>> memory.start
            1
        """

        return self.start_()

    cdef addr_t endex_(self):
        cdef:
            const Rack_* blocks

        if not self._trim_endex_:
            # Return actual
            blocks = self._
            if Rack_Length(blocks):
                return Block_Endex(Rack_Last__(blocks))
            else:
                return self.start_()
        else:
            return self._trim_endex

    @property
    def endex(
        self: 'Memory',
    ) -> Address:
        r"""int: Exclusive end address.

        This property holds the exclusive end address of the virtual space.
        By default, it is the current maximmum exclusive end address of
        the last stored block.

        If  :attr:`trim_endex` not ``None``, that is returned.

        If the memory has no data and no trimming, :attr:`start` is returned.

        Examples:
            >>> Memory().endex
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.endex
            8

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC']], endex=8)
            >>> memory.endex
            8
        """

        return self.endex_()

    cdef (addr_t, addr_t) span_(self):
        return self.start_(), self.endex_()

    @property
    def span(
        self: 'Memory',
    ) -> ClosedInterval:
        r"""tuple of int: Memory address span.

        A :obj:`tuple` holding both :attr:`start` and :attr:`endex`.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.span
            (1, 8)
        """

        return self.span_()

    @property
    def endin(
        self: 'Memory',
    ) -> Address:
        r"""int: Inclusive end address.

        This property holds the inclusive end address of the virtual space.
        By default, it is the current maximmum inclusive end address of
        the last stored block.

        If  :attr:`trim_endex` not ``None``, that minus one is returned.

        If the memory has no data and no trimming, :attr:`start` is returned.

        Examples:
            >>> Memory().endin
            -1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.endin
            7

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC']], endex=8)
            >>> memory.endin
            7
        """
        cdef:
            const Rack_* blocks

        if not self._trim_endex_:
            # Return actual
            blocks = self._
            if Rack_Length(blocks):
                return <object>Block_Endex(Rack_Last__(blocks)) - 1
            else:
                return self.start - 1
        else:
            return <object>self._trim_endex - 1

    cdef addr_t content_start_(self):
        cdef:
            const Rack_* blocks = self._

        if Rack_Length(blocks):
            return Block_Start(Rack_First__(blocks))
        elif not self._trim_start_:
            return 0
        else:
            return self._trim_start

    @property
    def content_start(
        self: 'Memory',
    ) -> Address:
        r"""int: Inclusive content start address.

        This property holds the inclusive start address of the memory content.
        By default, it is the current minimum inclusive start address of
        the first stored block.

        If the memory has no data and no trimming, 0 is returned.

        Trimming is considered only for an empty memory.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_start
            1

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[[[|   |   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'xyz']], start=1)
            >>> memory.content_start
            5
        """

        return self.content_start_()

    cdef addr_t content_endex_(self):
        cdef:
            const Rack_* blocks = self._

        if Rack_Length(blocks):
            return Block_Endex(Rack_Last__(blocks))
        elif not self._trim_start_:
            return 0  # default to start
        else:
            return self._trim_start  # default to start

    @property
    def content_endex(
        self: 'Memory',
    ) -> Address:
        r"""int: Exclusive content end address.

        This property holds the exclusive end address of the memory content.
        By default, it is the current maximmum exclusive end address of
        the last stored block.

        If the memory has no data and no trimming, :attr:`start` is returned.

        Trimming is considered only for an empty memory.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_endex
            8

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC']], endex=8)
            >>> memory.content_endex
            4
        """

        return self.content_endex_()

    cdef (addr_t, addr_t) content_span_(self):
        return self.content_start_(), self.content_endex_()

    @property
    def content_span(
        self: 'Memory',
    ) -> ClosedInterval:
        r"""tuple of int: Memory content address span.

        A :attr:`tuple` holding both :attr:`content_start` and
        :attr:`content_endex`.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_span
            (1, 8)
        """

        return self.content_span_()

    @property
    def content_endin(
        self: 'Memory',
    ) -> Address:
        r"""int: Inclusive content end address.

        This property holds the inclusive end address of the memory content.
        By default, it is the current maximmum inclusive end address of
        the last stored block.

        If the memory has no data and no trimming, :attr:`start` minus one is
        returned.

        Trimming is considered only for an empty memory.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_endin
            7

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC']], endex=8)
            >>> memory.content_endin
            3
        """
        cdef:
            const Rack_* blocks = self._

        if Rack_Length(blocks):
            return <object>Block_Endex(Rack_Last__(blocks)) - 1
        elif not self._trim_start_:  # default to start-1
            return -1
        else:
            return <object>self._trim_start - 1  # default to start-1

    cdef addr_t content_size_(self):
        cdef:
            const Rack_* blocks = self._
            size_t block_index
            const Block_* block
            addr_t content_size = 0

        for block_index in range(Rack_Length(blocks)):
            block = Rack_Get__(blocks, block_index)
            content_size += Block_Length(block)
        return content_size

    @property
    def content_size(
        self: 'Memory',
    ) -> Address:
        r"""Actual content size.

        Returns:
            int: The sum of all block lengths.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_size
            6

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC']], endex=8)
            >>> memory.content_size
            3
        """

        return self.content_size_()

    cdef size_t content_parts_(self):
        return Rack_Length(self._)

    @property
    def content_parts(
        self: 'Memory',
    ) -> int:
        r"""Number of blocks.

        Returns:
            int: The number of blocks.

        Examples:
            >>> Memory().content_parts
            0

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.content_parts
            2

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC']], endex=8)
            >>> memory.content_parts
            1
        """

        return self.content_parts_()

    cdef vint validate_(self) except -1:
        cdef:
            const Rack_* blocks = self._
            size_t block_count = Rack_Length(blocks)

            addr_t start
            addr_t endex
            addr_t previous_endex = 0

            size_t block_index
            const Block_* block
            addr_t block_start
            addr_t block_endex

        start, endex = self.bound_(None, None)
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

    def validate(
        self: 'Memory',
    ) -> None:
        r"""Validates internal structure.

        It makes sure that all the allocated blocks are sorted by block start
        address, and that all the blocks are non-overlapping.

        Raises:
            :obj:`ValueError`: Invalid data detected (see exception message).
        """

        self.validate_()

    cdef (addr_t, addr_t) bound_(self, object start, object endex):
        cdef:
            addr_t trim_start
            addr_t trim_endex
            addr_t start_ = 0 if start is None else <addr_t>start
            addr_t endex_ = start_ if endex is None else <addr_t>endex

        trim_start = self._trim_start
        trim_endex = self._trim_endex

        if start is None:
            if not self._trim_start_:
                if Rack_Length(self._):
                    start_ = Block_Start(Rack_First__(self._))
                else:
                    start_ = 0
            else:
                start_ = trim_start
        else:
            if self._trim_start_:
                if start_ < trim_start:
                    start_ = trim_start
            if endex is not None:
                if endex_ < start_:
                    endex_ = start_

        if endex is None:
            if not self._trim_endex_:
                if Rack_Length(self._):
                    endex_ = Block_Endex(Rack_Last__(self._))
                else:
                    endex_ = start_
            else:
                endex_ = trim_endex
        else:
            if self._trim_endex_:
                if endex_ > trim_endex:
                    endex_ = trim_endex
            if start is not None:
                if start_ > endex_:
                    start_ = endex_

        return start_, endex_

    def bound(
        self: 'Memory',
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
            >>> Memory().bound()
            (0, 0)
            >>> Memory().bound(endex=100)
            (0, 0)

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [5, b'xyz']])
            >>> memory.bound(0, 30)
            (1, 8)
            >>> memory.bound(2, 6)
            (2, 6)
            >>> memory.bound(endex=6)
            (1, 6)
            >>> memory.bound(start=2)
            (2, 8)

            ~~~

            +---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            +===+===+===+===+===+===+===+===+===+
            |   |[[[|   |[A | B | C]|   |   |)))|
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[3, b'ABC']], start=1, endex=8)
            >>> memory.bound()
            (1, 8)
            >>> memory.bound(0, 30)
            (1, 8)
            >>> memory.bound(2, 6)
            (2, 6)
            >>> memory.bound(start=2)
            (2, 8)
            >>> memory.bound(endex=6)
            (1, 6)
        """

        return self.bound_(start, endex)

    def _block_index_at(
        self: 'Memory',
        address: Address,
    ) -> Optional[BlockIndex]:
        r"""Locates the block enclosing an address.

        Returns the index of the block enclosing the given address.

        Arguments:
            address (int):
                Address of the target item.

        Returns:
            int: Block index if found, ``None`` otherwise.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   | 0 | 0 | 0 | 0 |   | 1 |   | 2 | 2 | 2 |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> [memory._block_index_at(i) for i in range(12)]
            [None, 0, 0, 0, 0, None, 1, None, 2, 2, 2, None]
        """
        cdef:
            ssize_t block_index

        block_index = Rack_IndexAt(self._, address)
        return None if block_index < 0 else block_index

    def _block_index_start(
        self: 'Memory',
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

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 2 | 2 | 2 | 3 |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> [memory._block_index_start(i) for i in range(12)]
            [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
        """

        return Rack_IndexStart(self._, address)

    def _block_index_endex(
        self: 'Memory',
        address: Address,
    ) -> BlockIndex:
        r"""Locates the first block after an address range.

        Returns the index of the first block whose end address is lesser than or
        equal to `address`.

        Useful to find the termination block index in a ranged search.

        Arguments:
            address (int):
                Exclusive end address of the scanned range.

        Returns:
            int: First block index after `address`.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> [memory._block_index_endex(i) for i in range(12)]
            [0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
        """

        return Rack_IndexEndex(self._, address)

    cdef int peek_(self, addr_t address) except -2:
        cdef:
            addr_t address_ = address
            ssize_t block_index
            const Block_* block

        block_index = Rack_IndexAt(self._, address_)
        if block_index < 0:
            return -1
        else:
            block = Rack_Get__(self._, <size_t>block_index)
            return Block_Get__(block, address_ - Block_Start(block))

    def peek(
        self: 'Memory',
        address: Address,
    ) -> Optional[Value]:
        r"""Gets the item at an address.

        Returns:
            int: The item at `address`, ``None`` if empty.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
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
        cdef:
            int value

        value = self.peek_(<addr_t>address)
        return None if value < 0 else value

    cdef int poke_none_(self, addr_t address) except -2:
        cdef:
            int value

        # Standard clear method
        value = self.peek_(address)
        self._erase_(address, address + 1, False, False)  # clear
        return value

    cdef vint poke_none__(self, addr_t address) except -1:
        # Standard clear method
        self._erase_(address, address + 1, False, False)  # clear

    cdef int poke_(self, addr_t address, byte_t item) except -2:
        cdef:
            Rack_* blocks = self._
            size_t block_count = Rack_Length(blocks)
            size_t block_index
            Block_* block
            addr_t block_start
            addr_t block_endex
            Block_* block2
            addr_t block_start2
            int value

        block_index = Rack_IndexEndex(blocks, address) - 1

        if block_index < block_count:
            block = Rack_Get__(blocks, block_index)
            block_start = Block_Start(block)
            block_endex = Block_Endex(block)

            if block_start <= address < block_endex:
                # Address within existing block, update directly
                address -= block_start
                value = Block_Get__(block, <size_t>address)
                Block_Set__(block, <size_t>address, item)
                return value

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
                        self._ = blocks = Rack_Pop_(blocks, block_index, NULL)
                return -1

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
                        return -1

        # There is no faster way than the standard block writing method
        self._erase_(address, address + 1, False, True)  # insert
        self._insert_(address, 1, &item, False)

        self._crop_(self._trim_start, self._trim_endex, None)
        return -1

    def poke(
        self: 'Memory',
        address: Address,
        item: Optional[Union[AnyBytes, Value]],
    ) -> Optional[Value]:
        r"""Sets the item at an address.

        Arguments:
            address (int):
                Address of the target item.

            item (int or byte):
                Item to set, ``None`` to clear the cell.

        Returns:
            int: The previous item at `address`, ``None`` if empty.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.poke(3, b'@')  # -> ord('C') = 67
            67
            >>> memory.peek(3)  # -> ord('@') = 64
            64
            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.poke(5, '@')
            None
            >>> memory.peek(5)  # -> ord('@') = 64
            64
        """
        cdef:
            addr_t address_ = <addr_t>address
            int value

        if item is None:
            value = self.poke_none_(address_)
        else:
            if isinstance(item, int):
                value = self.poke_(address_, <byte_t>item)
            else:
                if len(item) != 1:
                    raise ValueError('expecting single item')
                value = self.poke_(address_, <byte_t>item[0])

        return None if value < 0 else value

    cdef Memory extract_(self, addr_t start, addr_t endex,
                         size_t pattern_size, const byte_t* pattern_ptr,
                         saddr_t step, bint bound):
        cdef:
            const Rack_* blocks1 = self._
            size_t block_count = Rack_Length(blocks1)
            size_t block_index
            size_t block_index_start
            size_t block_index_endex
            Memory memory
            Rack_* blocks2
            Block_* block2
            addr_t offset
            Block_* pattern = NULL
            int value
            saddr_t skip

        memory = Memory()

        if step == 1:
            if start < endex and block_count:
                block_index_start = Rack_IndexStart(blocks1, start)
                block_index_endex = Rack_IndexEndex(blocks1, endex)
            else:
                block_index_start = 0
                block_index_endex = 0

            # Reserve slots to clone blocks
            blocks2 = memory._
            block_count = block_index_endex - block_index_start
            memory._ = blocks2 = Rack_Reserve_(blocks2, 0, block_count)
            try:
                # Clone blocks into the new memory
                for block_index in range(block_count):
                    block1 = Rack_Get__(blocks1, block_index_start + block_index)
                    block2 = Block_Copy(block1)
                    Rack_Set__(blocks2, block_index, block2)
            except:
                memory._ = blocks2 = Rack_Clear(blocks2)  # orphan
                raise

            # Trim data in excess
            memory._crop_(start, endex, None)

            if pattern_size and pattern_ptr:
                pattern = Block_Create(0, pattern_size, pattern_ptr)
                try:
                    memory.flood_(start, endex, &pattern, None)
                except:
                    Block_Free(pattern)  # orphan
                    raise
        else:
            if step > 1:
                block2 = NULL
                offset = start
                pattern_obj = <const byte_t[:pattern_size]>pattern_ptr if pattern_ptr else None
                rover = Rover(self, start, endex, pattern_obj, True, False)
                try:
                    while True:
                        value = rover.next_()
                        if value < 0:
                            if block2:
                                memory._ = Rack_Append(memory._, block2)
                                block2 = NULL
                        else:
                            if not block2:
                                block2 = Block_Alloc(offset, 0, False)
                            block2 = Block_Append(block2, <byte_t>value)

                        offset += 1
                        for skip in range(step - 1):
                            rover.next_()
                except StopIteration:
                    if block2:
                        memory._ = Rack_Append(memory._, block2)
                        block2 = NULL
                finally:
                    block2 = Block_Free(block2)  # orphan

                if bound:
                    endex = offset
        if bound:
            memory._trim_start_ = True
            memory._trim_endex_ = True
            memory._trim_start = start
            memory._trim_endex = endex

        return memory

    def extract(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
        step: Optional[Address] = None,
        bound: bool = True,
    ) -> 'Memory':
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
            :obj:`Memory`: A copy of the memory from the selected range.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C | D]|   |[$]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
            >>> memory.extract()._blocks
            [[1, b'ABCD'], [6, b'$'], [8, b'xyz']]
            >>> memory.extract(2, 9)._blocks
            [[2, b'BCD'], [6, b'$'], [8, b'x']]
            >>> memory.extract(start=2)._blocks
            [[2, b'BCD'], [6, b'$'], [8, b'xyz']]
            >>> memory.extract(endex=9)._blocks
            [[1, b'ABCD'], [6, b'$'], [8, b'x']]
            >>> memory.extract(5, 8).span
            (5, 8)
            >>> memory.extract(pattern='.')._blocks
            [[1, b'ABCD.$.xyz']]
            >>> memory.extract(pattern='.', step=3)._blocks
            [[1, b'AD.z']]
        """
        cdef:
            addr_t start_
            addr_t endex_
            const byte_t[:] pattern_view
            byte_t pattern_value
            size_t pattern_size
            const byte_t* pattern_ptr
            saddr_t step_ = <saddr_t>1 if step is None else <saddr_t>step
            bint bound_ = <bint>bound

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

        start_, endex_ = self.bound_(start, endex)
        return self.extract_(start_, endex_, pattern_size, pattern_ptr, step_, bound_)

    cdef vint shift_left_(self, addr_t offset, list backups) except -1:
        cdef:
            Rack_* blocks = self._
            size_t block_index
            Block_* block

        if offset and Rack_Length(blocks):
            self._pretrim_start_(ADDR_MAX, offset, backups)
            blocks = self._

            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                block.address -= offset

    cdef vint shift_right_(self, addr_t offset, list backups) except -1:
        cdef:
            Rack_* blocks = self._
            size_t block_index
            Block_* block

        if offset and Rack_Length(blocks):
            self._pretrim_endex_(ADDR_MIN, offset, backups)
            blocks = self._

            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                block.address += offset

    def shift(
        self: 'Memory',
        offset: Address,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Shifts the items.

        Arguments:
            offset (int):
                Signed amount of address shifting.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items, before trimming.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |   |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']])
            >>> memory.shift(-2)
            >>> memory._blocks
            [[3, b'ABC'], [7, b'xyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[[[|   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |[y | z]|   |   |   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']], start=2)
            >>> backups = []
            >>> memory.shift(-7, backups=backups)
            >>> memory._blocks
            [[2, b'yz']]
            >>> len(backups)
            1
            >>> backups[0]._blocks
            [[5, b'ABC'], [9, b'x']]
        """

        if offset < 0:
            return self.shift_left_(<addr_t>-offset, backups)
        else:
            return self.shift_right_(<addr_t>offset, backups)

    cdef vint reserve_(self, addr_t address, addr_t size, list backups) except -1:
        cdef:
            addr_t offset
            Rack_* blocks = self._
            size_t block_count
            size_t block_index
            Block_* block
            addr_t block_start
            Block_* block2

        if size and Rack_Length(blocks):
            self._pretrim_endex_(address, size, backups)

            blocks = self._
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
                        self._ = blocks = Rack_Insert(blocks, block_index, block2)
                    except:
                        block2 = Block_Free(block2)  # orphan
                        raise
                    block_index += 1

                for block_index in range(block_index, Rack_Length(blocks)):
                    block = Rack_Get_(blocks, block_index)
                    block.address += size

    def reserve(
        self: 'Memory',
        address: Address,
        size: Address,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Inserts emptiness.

        Reserves emptiness at the provided address.

        Arguments:
            address (int):
                Start address of the emptiness to insert.

            size (int):
                Size of the emptiness to insert.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items, before trimming.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |[A]|   |   | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[3, b'ABC'], [7, b'xyz']])
            >>> memory.reserve(4, 2)
            >>> memory._blocks
            [[2, b'A'], [6, b'BC'], [9, b'xyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+---+
            | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |   |   |[A | B | C]|   |[x | y | z]|)))|
            +---+---+---+---+---+---+---+---+---+---+---+
            |   |   |   |   |   |   |   |   |[A | B]|)))|
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']], endex=12)
            >>> backups = []
            >>> memory.reserve(5, 5, backups=backups)
            >>> memory._blocks
            [[10, b'AB']]
            >>> len(backups)
            1
            >>> backups[0]._blocks
            [[7, b'C'], [9, b'xyz']]
        """

        self.reserve_(<addr_t>address, <addr_t>size, backups)

    cdef vint _insert_(self, addr_t address, size_t size, const byte_t* buffer, bint shift_after) except -1:
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
            blocks = self._
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
                        block_index += 1
                        if block_index < Rack_Length(blocks):
                            CheckAddAddrU(block_endex, size)
                            block_endex += size

                            block2 = Rack_Get_(blocks, block_index)
                            block_start2 = Block_Start(block2)

                            # Merge with next block
                            if block_endex == block_start2:
                                block = Block_Extend(block, block2)
                                Rack_Set__(blocks, block_index - 1, block)  # update pointer
                                self._ = blocks = Rack_Pop_(blocks, block_index, NULL)
                    return 0

            if block_index < Rack_Length(blocks):
                block = Rack_Get_(blocks, block_index)
                block_start = Block_Start(block)

                if address < block_start:
                    if shift_after:
                        # Insert a standalone block before
                        block = Block_Create(address, size, buffer)
                        try:
                            self._ = blocks = Rack_Insert(blocks, block_index, block)
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
                                self._ = blocks = Rack_Insert(blocks, block_index, block)
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
                    self._ = blocks = Rack_Append(blocks, block)
                except:
                    Block_Free(block)  # orphan
                    raise

    def _insert(
        self: 'Memory',
        address: Address,
        data: bytearray,
        shift_after: bool,
    ) -> None:
        r"""Inserts data.

        Low-level method to insert data into the underlying data structure.

        Arguments:
            address (int):
                Address of the insertion point.

            data (:obj:`bytearray`):
                Data to insert.

            shift_after (bool):
                Shifts the addresses of blocks after the insertion point,
                adding the size of the inserted data.
        """
        cdef:
            size_t size
            const byte_t[:] view

        view = data
        size = len(view)
        if size > SIZE_HMAX:
            raise OverflowError('data size')

        self._insert_(<addr_t>address, size, &view[0], <bint>shift_after)

    cdef vint _erase_(self, addr_t start, addr_t endex, bint shift_after, bint merge_deletion) except -1:
        cdef:
            addr_t size
            addr_t offset

            Rack_* blocks = self._
            size_t block_index
            size_t inner_start
            size_t inner_endex

            Block_* block = NULL
            addr_t block_start
            addr_t block_endex

            Block_* block2 = NULL
            addr_t block_start2

        if endex > start:
            size = endex - start
            block_index = Rack_IndexStart(blocks, start)

            # Delete final/inner part of deletion start block
            for block_index in range(block_index, Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)

                block_start = Block_Start(block)
                if start <= block_start:
                    break  # inner starts here

                block_endex = Block_Endex(block)
                if start < block_endex:
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
                            self._ = blocks = Rack_Insert_(blocks, block_index, block)
                        except:
                            block = Block_Free(block)  # orphan
                            raise
                    block_index += 1  # skip this from inner part
                    break
            else:
                block_index = Rack_Length(blocks)

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

            if merge_deletion:
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

            if shift_after:
                # Shift blocks after deletion
                for block_index in range(block_index, Rack_Length(blocks)):
                    block = Rack_Get__(blocks, block_index)
                    CheckSubAddrU(block.address, size)
                    block.address -= size  # update address

            # Delete inner full blocks
            if inner_start < inner_endex:
                self._ = blocks = Rack_DelSlice_(blocks, inner_start, inner_endex)

    def _erase(
        self: 'Memory',
        start: Address,
        endex: Address,
        shift_after: bool,
        merge_deletion: bool,
    ) -> None:
        r"""Erases an address range.

        Low-level method to erase data within the underlying data structure.

        Arguments:
            start (int):
                Start address of the erasure range.

            endex (int):
                Exclusive end address of the erasure range.

            shift_after (bool):
                Shifts addresses of blocks after the end of the range,
                subtracting the size of the range itself.

            merge_deletion (bool):
                If data blocks before and after the address range are
                contiguous after erasure, merge the two blocks together.
        """

        self._erase_(<addr_t>start, <addr_t>endex, <bint>shift_after, <bint>merge_deletion)

    cdef vint insert_same_(self, addr_t address, Memory data, list backups) except -1:
        cdef:
            addr_t data_start
            addr_t data_endex

        data_start = data.start_()
        data_endex = data.endex_()

        if data_start < data_endex:
            self.reserve_(data_start, data_endex, backups)
            self.write_same_(data_start, data, False, backups)

    cdef vint insert_raw_(self, addr_t address, size_t data_size, const byte_t* data_ptr, list backups) except -1:

        self._insert_(address, data_size, data_ptr, True)  # TODO: backups

        if data_size:
            self._crop(self._trim_start, self._trim_endex, None)  # TODO: pre-trimming

    def insert(
        self: 'Memory',
        address: Address,
        data: Union[AnyBytes, Value, 'Memory'],
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Inserts data.

        Inserts data, moving existing items after the insertion address by the
        size of the inserted data.

        Arguments::
            address (int):
                Address of the insertion point.

            data (bytes):
                Data to insert.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items, before trimming.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11|
            +===+===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |   |[x | y | z]|   |[$]|   |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |   |[x | y | 1 | z]|   |[$]|
            +---+---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> memory.insert(10, b'$')
            >>> memory._blocks
            [[1, b'ABC'], [6, b'xyz'], [10, b'$']]
            >>> memory.insert(8, b'1')
            >>> memory._blocks
            [[1, b'ABC'], [6, b'xy1z'], [11, b'$']]
        """
        cdef:
            addr_t address_ = <addr_t>address
            const byte_t[:] data_view
            byte_t data_value
            size_t data_size
            const byte_t* data_ptr

        if isinstance(data, Memory):
            self.insert_same_(address_, data, backups)

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

            self.insert_raw_(address_, data_size, data_ptr, backups)

    cdef vint delete_(self, addr_t start, addr_t endex, list backups) except -1:
        if start < endex:
            if backups is not None:
                backups.append(self.extract_(start, endex, 0, NULL, 1, True))

            self._erase_(start, endex, True, True)  # delete

    def delete(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Deletes an address range.

        Arguments:
            start (int):
                Inclusive start address for deletion.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for deletion.
                If ``None``, :attr:`endex` is considered.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items.

        Example:
            +---+---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12| 13|
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | y | z]|   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']])
            >>> memory.delete(6, 10)
            >>> memory._blocks
            [[5, b'Ayz']]
        """
        cdef:
            addr_t start_
            addr_t endex_

        start_, endex_ = self.bound_(start, endex)
        self.delete_(start_, endex_, backups)

    cdef vint clear_(self, addr_t start, addr_t endex, list backups) except -1:
        if start < endex:
            if backups is not None:
                backups.append(self.extract_(start, endex, 0, NULL, 1, True))

            self._erase_(start, endex, False, False)  # clear

    def clear(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Clears an address range.

        Arguments:
            start (int):
                Inclusive start address for clearing.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for clearing.
                If ``None``, :attr:`endex` is considered.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the cleared
                items.

        Example:
            +---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |[A]|   |   |   |   |[y | z]|   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']])
            >>> memory.clear(6, 10)
            >>> memory._blocks
            [[5, b'A'], [10, b'yz']]
        """
        cdef:
            addr_t start_
            addr_t endex_

        start_, endex_ = self.bound_(start, endex)
        self.clear_(start_, endex_, backups)

    cdef vint _pretrim_start_(self, addr_t endex_max, addr_t size, list backups) except -1:
        cdef:
            addr_t trim_start
            addr_t endex

        if size:
            trim_start = self._trim_start if self._trim_start_ else ADDR_MIN
            if CannotAddAddrU(trim_start, size):
                endex = ADDR_MAX
            else:
                endex = trim_start + size

            if endex > endex_max:
                endex = endex_max

            if backups is not None:
                backups.append(self.extract(endex=endex))

            self._erase_(ADDR_MIN, endex, False, False)  # clear

    def _pretrim_start(
        self: 'Memory',
        endex_max: Optional[Address],
        size: Address,
        backups: Optional[MemoryList],
    ) -> None:
        r"""Trims initial data.

        Low-level method to manage trimming of data starting from an address.

        Arguments:
            endex_max (int):
                Exclusive end address of the erasure range.
                If ``None``, :attr:`trim_start` plus `size` is considered.

            size (int):
                Size of the erasure range.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the cleared
                items.
        """
        cdef:
            addr_t endex_max_ = ADDR_MAX if endex_max is None else <addr_t>endex_max

        self._pretrim_start_(endex_max_, <addr_t>size, backups)

    cdef vint _pretrim_endex_(self, addr_t start_min, addr_t size, list backups) except -1:
        cdef:
            addr_t trim_endex
            addr_t start

        if size:
            trim_endex = self._trim_endex if self._trim_endex_ else ADDR_MAX
            if CannotSubAddrU(trim_endex, size):
                start = ADDR_MIN
            else:
                start = trim_endex - size

            if start < start_min:
                start = start_min

            if backups is not None:
                backups.append(self.extract(start=start))

            self._erase_(start, ADDR_MAX, False, False)  # clear

    def _pretrim_endex(
        self: 'Memory',
        start_min: Optional[Address],
        size: Address,
        backups: Optional[MemoryList],
    ) -> None:
        r"""Trims final data.

        Low-level method to manage trimming of data starting from an address.

        Arguments:
            start_min (int):
                Starting address of the erasure range.
                If ``None``, :attr:`trim_endex` minus `size` is considered.

            size (int):
                Size of the erasure range.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the cleared
                items.
        """
        cdef:
            addr_t start_min_ = ADDR_MIN if start_min is None else <addr_t>start_min

        self._pretrim_endex_(start_min_, <addr_t>size, backups)

    cdef vint _crop_(self, addr_t start, addr_t endex, list backups) except -1:
        cdef:
            addr_t block_start
            addr_t block_endex

        # Trim blocks exceeding before memory start
        if Rack_Length(self._):
            block_start = Block_Start(Rack_First_(self._))

            if block_start < start:
                if backups is not None:
                    backups.append(self.extract_(block_start, start, 0, NULL, 1, True))

                self._erase_(block_start, start, False, False)  # clear

        # Trim blocks exceeding after memory end
        if Rack_Length(self._):
            block_endex = Block_Endex(Rack_Last_(self._))

            if endex < block_endex:
                if backups is not None:
                    backups.append(self.extract_(endex, block_endex, 0, NULL, 1, True))

                self._erase_(endex, block_endex, False, False)  # clear

    def _crop(
        self: 'Memory',
        start: Address,
        endex: Address,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Keeps data within an address range.

        Low-level method to crop the underlying data structure.

        Arguments:
            start (int):
                Inclusive start address for cropping.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for cropping.
                If ``None``, :attr:`endex` is considered.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the cleared
                items.
        """
        cdef:
            addr_t start_
            addr_t endex_

        start_, endex_ = self.bound_(start, endex)
        self._crop_(start_, endex_, backups)

    def crop(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Keeps data within an address range.

        Arguments:
            start (int):
                Inclusive start address for cropping.
                If ``None``, :attr:`start` is considered.

            endex (int):
                Exclusive end address for cropping.
                If ``None``, :attr:`endex` is considered.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the cleared
                items.

        Example:
            +---+---+---+---+---+---+---+---+---+
            | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12|
            +===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+
            |   |   |[B | C]|   |[x]|   |   |   |
            +---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[5, b'ABC'], [9, b'xyz']])
            >>> memory.crop(6, 10)
            >>> memory._blocks
            [[6, b'BC'], [9, b'x']]
        """
        cdef:
            addr_t start_
            addr_t endex_

        start_, endex_ = self.bound_(start, endex)
        self._crop_(start_, endex_, backups)

    cdef vint write_same_(self, addr_t address, Memory data, bint clear, list backups) except -1:
        cdef:
            addr_t data_start
            addr_t data_endex
            addr_t size
            const Rack_* blocks
            size_t block_index
            const Block_* block
            addr_t block_start
            addr_t block_endex

        data_start = data.start_()
        data_endex = data.endex_()
        size = data_endex - data_start
        blocks = data._

        if size:
            if clear:
                # Clear anything between source data boundaries
                if backups is not None:
                    backups.append(self.extract_(data_start, data_endex, 0, NULL, 1, True))

                self._erase_(data_start, data_endex, False, True)  # insert

            else:
                # Clear only overwritten ranges
                for block_index in range(Rack_Length(blocks)):
                    block = Rack_Get__(blocks, block_index)

                    block_start = Block_Start(block)
                    CheckAddAddrU(block_start, address)
                    block_start += address

                    block_endex = Block_Endex(block)
                    CheckAddAddrU(block_endex, address)
                    block_endex += address

                    if backups is not None:
                        backups.append(self.extract_(block_start, block_endex, 0, NULL, 1, True))

                    self._erase_(block_start, block_endex, False, True)  # insert

            for block_index in range(Rack_Length(blocks)):
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                CheckAddAddrU(block_start, address)
                self._insert_(block_start + address, Block_Length(block), Block_At__(block, 0), False)

            self._crop_(self._trim_start, self._trim_endex, None)  # FIXME: prevent after-cropping; trim while writing

    cdef vint write_raw_(self, addr_t address, size_t data_size, const byte_t* data_ptr, list backups) except -1:
        cdef:
            addr_t size = data_size
            addr_t start
            addr_t endex
            addr_t trim_start
            addr_t trim_endex
            addr_t offset

        if CannotAddAddrU(address, size):
            size = ADDR_MAX - address

        if size:
            start = address
            endex = start + size

            trim_endex = self._trim_endex if self._trim_endex_ else ADDR_MAX
            if start >= trim_endex:
                return 0
            elif endex > trim_endex:
                size -= endex - trim_endex
                endex = start + size

            trim_start = self._trim_start if self._trim_start_ else ADDR_MIN
            if endex <= trim_start:
                return 0
            elif trim_start > start:
                offset = trim_start - start
                size -= offset
                start += offset
                endex = start + size
                data_ptr += offset

            CheckAddrToSizeU(size)
            if backups is not None:
                backups.append(self.extract_(start, endex, 0, NULL, 1, True))

            if size == 1:
                self.poke_(start, data_ptr[0])  # might be faster
            else:
                self._erase_(start, endex, False, True)  # insert
                self._insert_(start, <size_t>size, data_ptr, False)

    def write(
        self: 'Memory',
        address: Address,
        data: Union[AnyBytes, Value, 'Memory'],
        clear: bool = False,
        backups: Optional[MemoryList] = None,
    ) -> None:
        r"""Writes data.

        Arguments:
            address (int):
                Address where to start writing data.

            data (bytes):
                Data to write.

            clear (bool):
                Clears the target range before writing data.
                Useful only if `data` is a :obj:`Memory` with empty spaces.

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items.

        Example:
            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C]|   |[1 | 2 | 3 | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> memory.write(5, b'123')
            >>> memory._blocks
            [[1, b'ABC'], [5, b'123z']]
        """
        cdef:
            addr_t address_ = <addr_t>address
            const byte_t[:] data_view
            byte_t data_value
            size_t data_size
            const byte_t* data_ptr

        if isinstance(data, Memory):
            self.write_same_(address_, data, <bint>clear, backups)

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

            self.write_raw_(address_, data_size, data_ptr, backups)

    cdef vint fill_(self, addr_t start, addr_t endex, Block_** pattern, list backups, addr_t start_) except -1:
        cdef:
            size_t offset
            size_t size

        if start < endex:
            CheckAddrToSizeU(endex - start)
            if not Block_Length(pattern[0]):
                raise ValueError('non-empty pattern required')

            if start > start_:
                offset = start - start_
                CheckAddrToSizeU(offset)
                Block_RotateLeft_(pattern[0], <size_t>offset)

            # Resize the pattern to the target range
            size = <size_t>(endex - start)
            pattern[0] = Block_RepeatToSize(pattern[0], size)

            if backups is not None:
                backups.append(self.extract_(start, endex, 0, NULL, 1, True))

            # Standard write method
            self._erase_(start, endex, False, True)  # insert
            self._insert_(start, size, Block_At__(pattern[0], 0), False)

    def fill(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
        backups: Optional[MemoryList] = None,
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

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items, before trimming.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[1 | 2 | 3 | 1 | 2 | 3 | 1 | 2]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> memory.fill(pattern=b'123')
            >>> memory._blocks
            [[1, b'12312312']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | 1 | 2 | 3 | 1 | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> memory.fill(3, 7, b'123')
            >>> memory._blocks
            [[1, b'AB1231yz']]
        """
        cdef:
            addr_t start__
            addr_t start_
            addr_t endex_
            Block_* pattern_ = NULL

        start_, endex_ = self.bound_(start, endex)
        if start_ < endex_:
            pattern_ = Block_FromObject(0, pattern, False)  # size checked later on
            try:
                start__ = self.start_() if start is None else <addr_t>start
                self.fill_(start_, endex_, &pattern_, backups, start__)
            finally:
                Block_Free(pattern_)  # orphan

    cdef vint flood_(self, addr_t start, addr_t endex, Block_** pattern, list backups) except -1:
        cdef:
            Rack_* blocks
            const Block_* block
            addr_t block_start
            addr_t block_endex
            size_t block_index_start
            size_t block_index_endex
            addr_t offset

        if start < endex:
            blocks = self._
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
            if not Block_Length(pattern[0]):
                raise ValueError('non-empty pattern required')

            if backups is not None:
                for gap_start, gap_endex in self.gaps(start, endex):
                    backups.append(Memory(start=gap_start, endex=gap_endex, validate=False))

            size = <size_t>(endex - start)
            pattern[0] = Block_RepeatToSize(pattern[0], size)
            pattern[0].address = start

            for block_index in range(block_index_start, block_index_endex):
                block = Rack_Get__(blocks, block_index)
                offset = Block_Start(block) - start
                # CheckAddrToSizeU(offset)  # implied
                pattern[0] = Block_Write_(pattern[0], <size_t>offset, Block_Length(block), Block_At__(block, 0))

            self._ = blocks = Rack_DelSlice_(blocks, block_index_start, block_index_endex)
            self._ = blocks = Rack_Insert_(blocks, block_index_start, pattern[0])

    def flood(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
        backups: Optional[MemoryList] = None,
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

            backups (list of :obj:`Memory`):
                Optional output list holding backup copies of the deleted
                items, before trimming.

        Examples:
            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | 1 | 2 | x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> memory.flood(pattern=b'123')
            >>> memory._blocks
            [[1, b'ABC12xyz']]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   |[A | B | C | 2 | 3 | x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> memory.flood(3, 7, b'123')
            >>> memory._blocks
            [[1, b'ABC23xyz']]
        """
        cdef:
            addr_t start_
            addr_t endex_
            Block_* pattern_ = NULL

        start_, endex_ = self.bound_(start, endex)
        if start_ < endex_:
            pattern_ = Block_FromObject(0, pattern, False)  # size checked later on
            try:
                self.flood_(start_, endex_, &pattern_, backups)
            except:
                Block_Free(pattern_)  # orphan
                raise

    def keys(
        self: 'Memory',
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.keys())
            [1, 2, 3, 4, 5, 6, 7, 8]
            >>> list(memory.keys(endex=8))
            [0, 1, 2, 3, 4, 5, 6, 7]
            >>> list(memory.keys(3, 8))
            [3, 4, 5, 6, 7]
            >>> list(islice(memory.keys(3, ...), 7))
            [3, 4, 5, 6, 7, 8, 9]
        """
        cdef:
            addr_t start_
            addr_t endex_

        if start is None:
            start_ = self.start_()
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = self.endex_()
        elif endex is Ellipsis:
            endex_ = ADDR_MAX
        else:
            endex_ = <addr_t>endex

        while start_ < endex_:
            yield start_
            start_ += 1

    def values(
        self: 'Memory',
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
            >>> from itertools import islice
            >>> memory = Memory()
            >>> list(memory.values(endex=8))
            [None, None, None, None, None, None, None]
            >>> list(memory.values(3, 8))
            [None, None, None, None, None]
            >>> list(islice(memory.values(3, ...), 7))
            [None, None, None, None, None, None, None]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67|   |   |120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.values())
            [65, 66, 67, None, None, 120, 121, 122]
            >>> list(memory.values(3, 8))
            [67, None, None, 120, 121]
            >>> list(islice(memory.values(3, ...), 7))
            [67, None, None, 120, 121, 122, None]
        """
        cdef:
            addr_t start_
            addr_t endex_

        if start is None:
            start_ = self.start_()
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = self.endex_()
        elif endex is Ellipsis:
            endex_ = ADDR_MAX
        else:
            endex_ = <addr_t>endex

        yield from Rover(self, start_, endex_, pattern, True, endex is Ellipsis)

    def rvalues(
        self: 'Memory',
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
            >>> from itertools import islice
            >>> memory = Memory()
            >>> list(memory.values(endex=8))
            [None, None, None, None, None, None, None]
            >>> list(memory.values(3, 8))
            [None, None, None, None, None]
            >>> list(islice(memory.values(3, ...), 7))
            [None, None, None, None, None, None, None]

            ~~~

            +---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
            +===+===+===+===+===+===+===+===+===+===+
            |   |[A | B | C]|   |   |[x | y | z]|   |
            +---+---+---+---+---+---+---+---+---+---+
            |   | 65| 66| 67|   |   |120|121|122|   |
            +---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.values())
            [65, 66, 67, None, None, 120, 121, 122]
            >>> list(memory.values(3, 8))
            [67, None, None, 120, 121]
            >>> list(islice(memory.values(3, ...), 7))
            [67, None, None, 120, 121, 122, None]
        """
        cdef:
            addr_t start_
            addr_t endex_

        if start is None:
            start_ = self.start_()
        elif start is Ellipsis:
            start_ = ADDR_MIN
        else:
            start_ = <addr_t>start

        if endex is None:
            endex_ = self.endex_()
        else:
            endex_ = <addr_t>endex

        yield from Rover(self, start_, endex_, pattern, False, start is Ellipsis)

    def items(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Tuple[Address, Value]]:
        r"""Iterates over address and value couples.

        Iterates over address and value couples, from `start` to `endex`.
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
            int: Range address and value couples.

        Examples:
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

            >>> memory = Memory(blocks=[[1, b'ABC'], [6, b'xyz']])
            >>> list(memory.items())
            [(1, 65), (2, 66), (3, 67), (4, None), (5, None), (6, 120), (7, 121), (8, 122)]
            >>> list(memory.items(3, 8))
            [(3, 67), (4, None), (5, None), (6, 120), (7, 121)]
            >>> list(islice(memory.items(3, ...), 7))
            [(3, 67), (4, None), (5, None), (6, 120), (7, 121), (8, 122), (9, None)]
        """

        yield from zip(self.keys(start, endex), self.values(start, endex, pattern))

    def intervals(
        self: 'Memory',
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
            couple of addresses: Block data interval boundaries.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'AB'], [5, b'x'], [7, b'123']])
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
            const Rack_* blocks = self._
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
            start_, endex_ = self.bound_(start, endex)

            for block_index in range(block_index_start, block_index_endex):
                block = Rack_Get__(blocks, block_index)
                block_start = Block_Start(block)
                block_endex = Block_Endex(block)
                slice_start = block_start if start_ < block_start else start_
                slice_endex = endex_ if endex_ < block_endex else block_endex
                if slice_start < slice_endex:
                    yield slice_start, slice_endex

    def gaps(
        self: 'Memory',
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        bound: bool = False,
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

            bound (bool):
                Only gaps within blocks are considered; emptiness before and
                after global data bounds are ignored.

        Yields:
            couple of addresses: Block data interval boundaries.

        Example:
            +---+---+---+---+---+---+---+---+---+---+---+
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
            +===+===+===+===+===+===+===+===+===+===+===+
            |   |[A | B]|   |   |[x]|   |[1 | 2 | 3]|   |
            +---+---+---+---+---+---+---+---+---+---+---+

            >>> memory = Memory(blocks=[[1, b'AB'], [5, b'x'], [7, b'123']])
            >>> list(memory.gaps())
            [(None, 1), (3, 5), (6, 7), (10, None)]
            >>> list(memory.gaps(bound=True))
            [(3, 5), (6, 7)]
            >>> list(memory.gaps(2, 6))
            [(3, 5)]
        """
        cdef:
            addr_t start_
            addr_t endex_
            bint bound_ = <bint>bound
            const Rack_* blocks = self._
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
            start_, endex_ = self.bound(start, endex)

            if start__ is None:
                if not bound_:
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

            if endex__ is None and not bound_:
                yield start_, None
            elif start_ < endex_:
                yield start_, endex_

        elif not bound_:
            yield None, None

    def equal_span(
        self: 'Memory',
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

            >>> memory = Memory(blocks=[[0, b'ABBBC'], [7, b'CCD']])
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
            const Rack_* blocks = self._
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

    def block_span(
        self: 'Memory',
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

            >>> memory = Memory(blocks=[[0, b'ABBBC'], [7, b'CCD']])
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
            const Rack_* blocks = self._
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

    def _to_blocks(
        self: 'Memory',
        size_max: Optional[Address] = STR_MAX_CONTENT_SIZE,
    ) -> BlockList:
        r"""Converts into a list of blocks."""
        cdef:
            const Rack_* blocks1 = self._
            size_t block_count = Rack_Length(blocks1)
            size_t block_index
            Block_* block = NULL
            size_t size
            const byte_t[:] view
            list blocks2 = []

        for block_index in range(block_count):
            block = Rack_Get__(blocks1, block_index)
            size = Block_Length(block)
            view = <const byte_t[:size]><const byte_t*>Block_At__(block, 0)
            data = bytearray(view) if size_max is None or size < size_max else view
            blocks2.append([Block_Start(block), data])
        return blocks2

    @property
    def _blocks(
        self: 'Memory',
    ) -> BlockList:
        r"""list of blocks: A sequence of spaced blocks, sorted by address."""

        return self._to_blocks(size_max=None)
