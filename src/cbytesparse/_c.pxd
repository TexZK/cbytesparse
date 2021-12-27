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

from itertools import count as _count
from itertools import islice as _islice
from itertools import repeat as _repeat
from itertools import zip_longest as _zip_longest
from typing import Any
from typing import ByteString
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

Address = int
Value = int
AnyBytes = Union[ByteString, bytes, bytearray, memoryview, Sequence[Value]]
Data = bytearray

Block = List[Union[Address, Data]]  # typed as Tuple[Address, Data]
BlockIndex = int
BlockIterable = Iterable[Block]
BlockSequence = Sequence[Block]
BlockList = List[Block]
MemoryList = List['Memory']

OpenInterval = Tuple[Optional[Address], Optional[Address]]
ClosedInterval = Tuple[Address, Address]

EllipsisType = Type['Ellipsis']

# =====================================================================================================================

cimport cython
from cpython cimport Py_buffer
from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS
from cpython.buffer cimport PyBUF_C_CONTIGUOUS
from cpython.buffer cimport PyBUF_F_CONTIGUOUS
from cpython.buffer cimport PyBUF_FORMAT
from cpython.buffer cimport PyBUF_ND
from cpython.buffer cimport PyBUF_SIMPLE
from cpython.buffer cimport PyBUF_STRIDES
from cpython.buffer cimport PyBUF_WRITABLE
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Free
from cpython.mem cimport PyMem_Malloc
from cpython.mem cimport PyMem_Realloc
from cpython.ref cimport PyObject
from libc.stdint cimport int_fast64_t
from libc.stdint cimport uint_fast64_t
from libc.stdint cimport uintptr_t
from libc.string cimport memcmp
from libc.string cimport memcpy
from libc.string cimport memmove
from libc.string cimport memset
from libc.string cimport strstr


cdef extern from *:
    r"""
    #include <stddef.h>
    #include <stdint.h>

    #define SIZE_MIN  ((size_t)0)
    #define SIZE_HMAX ((size_t)(SIZE_MAX >> 1))
    #ifndef SSIZE_MAX
        #define SSIZE_MAX ((ssize_t)SIZE_HMAX)
    #endif
    #ifndef SSIZE_MIN
        #define SSIZE_MIN (-(ssize_t)SIZE_HMAX - (ssize_t)1)
    #endif
    #define MARGIN    (sizeof(size_t) >> 1)

    typedef uint_fast64_t addr_t;
    typedef int_fast64_t saddr_t;
    #define ADDR_MIN  ((addr_t)0)
    #define ADDR_MAX  (UINT_FAST64_MAX)
    #define SADDR_MAX (INT_FAST64_MAX)
    #define SADDR_MIN (INT_FAST64_MIN)

    typedef unsigned char byte_t;
    """
    size_t SIZE_MIN
    size_t SIZE_MAX  # declare here instead of importing, to avoid strange compilation in tests
    size_t SIZE_HMAX
    ssize_t SSIZE_MAX
    ssize_t SSIZE_MIN
    size_t MARGIN

    ctypedef uint_fast64_t addr_t
    ctypedef int_fast64_t saddr_t
    addr_t ADDR_MIN
    addr_t ADDR_MAX
    saddr_t SADDR_MAX
    saddr_t SADDR_MIN

    ctypedef unsigned char byte_t

ctypedef bint vint  # "void int", to allow exceptions for functions returning void

# Forward class declarations
cdef class BlockView
cdef class Memory


cdef void* PyMem_Calloc(size_t nelem, size_t elsize, bint zero)


# =====================================================================================================================

cdef size_t Downsize(size_t allocated, size_t requested) nogil
cdef size_t Upsize(size_t allocated, size_t requested) nogil


# ---------------------------------------------------------------------------------------------------------------------

cdef void Reverse(byte_t* buffer, size_t start, size_t endin) nogil
cdef bint IsSequence(object obj) except -1

# =====================================================================================================================

cdef bint CannotAddSizeU(size_t a, size_t b) nogil
cdef vint CheckAddSizeU(size_t a, size_t b) except -1
cdef size_t AddSizeU(size_t a, size_t b) except? 0xDEAD

cdef bint CannotSubSizeU(size_t a, size_t b) nogil
cdef vint CheckSubSizeU(size_t a, size_t b) except -1
cdef size_t SubSizeU(size_t a, size_t b) except? 0xDEAD

cdef bint CannotMulSizeU(size_t a, size_t b) nogil
cdef vint CheckMulSizeU(size_t a, size_t b) except -1
cdef size_t MulSizeU(size_t a, size_t b) except? 0xDEAD


# ---------------------------------------------------------------------------------------------------------------------

cdef bint CannotAddSizeS(ssize_t a, ssize_t b) nogil
cdef vint CheckAddSizeS(ssize_t a, ssize_t b) except -1
cdef ssize_t AddSizeS(ssize_t a, ssize_t b) except? 0xDEAD

cdef bint CannotSubSizeS(ssize_t a, ssize_t b) nogil
cdef vint CheckSubSizeS(ssize_t a, ssize_t b) except -1
cdef ssize_t SubSizeS(ssize_t a, ssize_t b) except? 0xDEAD

cdef bint CannotMulSizeS(ssize_t a, ssize_t b) nogil
cdef vint CheckMulSizeS(ssize_t a, ssize_t b) except -1
cdef ssize_t MulSizeS(ssize_t a, ssize_t b) except? 0xDEAD


# =====================================================================================================================

cdef bint CannotAddAddrU(addr_t a, addr_t b) nogil
cdef vint CheckAddAddrU(addr_t a, addr_t b) except -1
cdef addr_t AddAddrU(addr_t a, addr_t b) except? 0xDEAD

cdef bint CannotSubAddrU(addr_t a, addr_t b) nogil
cdef vint CheckSubAddrU(addr_t a, addr_t b) except -1
cdef addr_t SubAddrU(addr_t a, addr_t b) except? 0xDEAD

cdef bint CannotMulAddrU(addr_t a, addr_t b) nogil
cdef vint CheckMulAddrU(addr_t a, addr_t b) except -1
cdef addr_t MulAddrU(addr_t a, addr_t b) except? 0xDEAD

cdef bint CannotAddrToSizeU(addr_t a) nogil
cdef vint CheckAddrToSizeU(addr_t a) except -1
cdef size_t AddrToSizeU(addr_t a) except? 0xDEAD


# ---------------------------------------------------------------------------------------------------------------------

cdef bint CannotAddAddrS(saddr_t a, saddr_t b) nogil
cdef vint CheckAddAddrS(saddr_t a, saddr_t b) except -1
cdef saddr_t AddAddrS(saddr_t a, saddr_t b) except? 0xDEAD

cdef bint CannotSubAddrS(saddr_t a, saddr_t b) nogil
cdef vint CheckSubAddrS(saddr_t a, saddr_t b) except -1
cdef saddr_t SubAddrS(saddr_t a, saddr_t b) except? 0xDEAD

cdef bint CannotMulAddrS(saddr_t a, saddr_t b) nogil
cdef vint CheckMulAddrS(saddr_t a, saddr_t b) except -1
cdef saddr_t MulAddrS(saddr_t a, saddr_t b) except? 0xDEAD

cdef bint CannotAddrToSizeS(saddr_t a) nogil
cdef vint CheckAddrToSizeS(saddr_t a) except -1
cdef ssize_t AddrToSizeS(saddr_t a) except? 0xDEAD


# =====================================================================================================================

cdef extern from *:
    r"""
    typedef struct Block_ {
        addr_t address;
        size_t references;
        size_t allocated;
        size_t start;
        size_t endex;
        byte_t data[1];
    } Block_;

    #define Block_HEADING (offsetof(Block_, data))
    """

    ctypedef struct Block_:
        # Block address; must be updated externally
        addr_t address

        # Reference counter
        size_t references

        # Buffer size in bytes; always positive
        size_t allocated

        # Start element index
        size_t start

        # Exclusive end element index
        size_t endex

        # Buffer data bytes (more can be present for an actual allocation)
        byte_t data[1]

    size_t Block_HEADING


cdef Block_* Block_Alloc(addr_t address, size_t size, bint zero) except NULL
cdef Block_* Block_Free(Block_* that)

cdef Block_* Block_Create(addr_t address, size_t size, const byte_t* buffer) except NULL
cdef Block_* Block_Copy(const Block_* that) except NULL
cdef Block_* Block_FromObject(addr_t address, object obj, bint nonnull) except NULL

cdef Block_* Block_Acquire(Block_* that) except NULL
cdef Block_* Block_Release_(Block_* that)
cdef Block_* Block_Release(Block_* that)

cdef size_t Block_Length(const Block_* that) nogil
cdef addr_t Block_Start(const Block_* that) nogil
cdef addr_t Block_Endex(const Block_* that) nogil
cdef addr_t Block_Endin(const Block_* that) nogil

cdef addr_t Block_BoundAddress(const Block_* that, addr_t address) nogil
cdef size_t Block_BoundAddressToOffset(const Block_* that, addr_t address) nogil
cdef size_t Block_BoundOffset(const Block_* that, size_t offset) nogil

cdef (addr_t, addr_t) Block_BoundAddressSlice(const Block_* that, addr_t start, addr_t endex) nogil
cdef (size_t, size_t) Block_BoundAddressSliceToOffset(const Block_* that, addr_t start, addr_t endex) nogil
cdef (size_t, size_t) Block_BoundOffsetSlice(const Block_* that, size_t start, size_t endex) nogil

cdef vint Block_CheckMutable(const Block_* that) except -1

cdef bint Block_Eq_(const Block_* that, size_t size, const byte_t* buffer) nogil
cdef bint Block_Eq(const Block_* that, const Block_* other) nogil

cdef int Block_Cmp_(const Block_* that, size_t size, const byte_t* buffer) nogil
cdef int Block_Cmp(const Block_* that, const Block_* other) nogil

cdef ssize_t Block_Find__(const Block_* that, size_t start, size_t endex, byte_t value) nogil
cdef ssize_t Block_Find_(const Block_* that, size_t start, size_t endex,
                         size_t size, const byte_t* buffer) nogil
cdef ssize_t Block_Find(const Block_* that, ssize_t start, ssize_t endex,
                        size_t size, const byte_t* buffer) nogil

cdef ssize_t Block_ReverseFind__(const Block_* that, size_t start, size_t endex, byte_t value) nogil
cdef ssize_t Block_ReverseFind_(const Block_* that, size_t start, size_t endex,
                                size_t size, const byte_t* buffer) nogil
cdef ssize_t Block_ReverseFind(const Block_* that, ssize_t start, ssize_t endex,
                               size_t size, const byte_t* buffer) nogil

cdef size_t Block_Count__(const Block_* that, size_t start, size_t endex, byte_t value) nogil
cdef size_t Block_Count_(const Block_* that, size_t start, size_t endex,
                         size_t size, const byte_t* buffer) nogil
cdef size_t Block_Count(const Block_* that, ssize_t start, ssize_t endex,
                        size_t size, const byte_t* buffer) nogil

cdef Block_* Block_Reserve_(Block_* that, size_t offset, size_t size, bint zero) except NULL
cdef Block_* Block_Delete_(Block_* that, size_t offset, size_t size) except NULL
cdef Block_* Block_Clear(Block_* that) except NULL

cdef byte_t* Block_At_(Block_* that, size_t offset) nogil
cdef const byte_t* Block_At__(const Block_* that, size_t offset) nogil

cdef byte_t Block_Get__(const Block_* that, size_t offset) nogil
cdef int Block_Get_(const Block_* that, size_t offset) except -1
cdef int Block_Get(const Block_* that, ssize_t offset) except -1

cdef byte_t Block_Set__(Block_* that, size_t offset, byte_t value) nogil
cdef int Block_Set_(Block_* that, size_t offset, byte_t value) except -1
cdef int Block_Set(Block_* that, ssize_t offset, byte_t value) except -1

cdef Block_* Block_Pop__(Block_* that, byte_t* value) except NULL
cdef Block_* Block_Pop_(Block_* that, size_t offset, byte_t* value) except NULL
cdef Block_* Block_Pop(Block_* that, ssize_t offset, byte_t* value) except NULL
cdef Block_* Block_PopLeft(Block_* that, byte_t* value) except NULL

cdef Block_* Block_Insert_(Block_* that, size_t offset, byte_t value) except NULL
cdef Block_* Block_Insert(Block_* that, ssize_t offset, byte_t value) except NULL

cdef Block_* Block_Append(Block_* that, byte_t value) except NULL
cdef Block_* Block_AppendLeft(Block_* that, byte_t value) except NULL

cdef Block_* Block_Extend_(Block_* that, size_t size, const byte_t* buffer) except NULL
cdef Block_* Block_Extend(Block_* that, const Block_* more) except NULL

cdef Block_* Block_ExtendLeft_(Block_* that, size_t size, const byte_t* buffer) except NULL
cdef Block_* Block_ExtendLeft(Block_* that, const Block_* more) except NULL

cdef void Block_RotateLeft__(Block_* that, size_t offset) nogil
cdef void Block_RotateLeft_(Block_* that, size_t offset) nogil
cdef void Block_RotateRight__(Block_* that, size_t offset) nogil
cdef void Block_RotateRight_(Block_* that, size_t offset) nogil
cdef void Block_Rotate(Block_* that, ssize_t offset) nogil

cdef Block_* Block_Repeat(Block_* that, size_t times) except NULL
cdef Block_* Block_RepeatToSize(Block_* that, size_t size) except NULL

cdef vint Block_Read_(const Block_* that, size_t offset, size_t size, byte_t* buffer) except -1
cdef Block_* Block_Write_(Block_* that, size_t offset, size_t size, const byte_t* buffer) except NULL

cdef vint Block_ReadSlice_(const Block_* that, size_t start, size_t endex,
                           size_t* size_, byte_t* buffer) except -1
cdef vint Block_ReadSlice(const Block_* that, ssize_t start, ssize_t endex,
                          size_t* size_, byte_t* buffer) except -1

cdef Block_* Block_GetSlice_(const Block_* that, size_t start, size_t endex) except NULL
cdef Block_* Block_GetSlice(const Block_* that, ssize_t start, ssize_t endex) except NULL

cdef Block_* Block_WriteSlice_(Block_* that, size_t start, size_t endex,
                               size_t size, const byte_t* buffer) except NULL
cdef Block_* Block_WriteSlice(Block_* that, ssize_t start, ssize_t endex,
                              size_t size, const byte_t* buffer) except NULL

cdef Block_* Block_SetSlice_(Block_* that, size_t start, size_t endex,
                             const Block_* src, size_t start2, size_t endex2) except NULL
cdef Block_* Block_SetSlice(Block_* that, ssize_t start, ssize_t endex,
                            const Block_* src, ssize_t start2, ssize_t endex2) except NULL

cdef Block_* Block_DelSlice_(Block_* that, size_t start, size_t endex) except NULL
cdef Block_* Block_DelSlice(Block_* that, ssize_t start, ssize_t endex) except NULL

cdef bytes Block_Bytes(const Block_* that)
cdef bytearray Block_Bytearray(const Block_* that)

cdef BlockView Block_View(Block_* that)
cdef BlockView Block_ViewSlice_(Block_* that, size_t start, size_t endex)
cdef BlockView Block_ViewSlice(Block_* that, ssize_t start, ssize_t endex)


# ---------------------------------------------------------------------------------------------------------------------

cdef class BlockView:
    cdef:
        Block_* _block  # wrapped C implementation
        size_t _start  # data slice start
        size_t _endex  # data slice endex
        object _memview  # shadow memoryview

    cdef bint check_(self) except -1


# =====================================================================================================================

cdef extern from *:
    r"""
    typedef struct Rack_ {
        size_t references;
        size_t allocated;
        size_t start;
        size_t endex;
        Block_* blocks[1];
    } Rack_;

    #define Rack_HEADING (offsetof(Rack_, blocks))
    """

    ctypedef struct Rack_:
        # Allocated list length; always positive
        size_t allocated

        # Start element index
        size_t start

        # Exclusive end element index
        size_t endex

        # Rack blocks (more can be present for an actual allocation)
        Block_* blocks[1]

    size_t Rack_HEADING


cdef Rack_* Rack_Alloc(size_t size) except NULL
cdef Rack_* Rack_Free(Rack_* that)

cdef Rack_* Rack_ShallowCopy(const Rack_* other) except NULL
cdef Rack_* Rack_Copy(const Rack_* other) except NULL
cdef Rack_* Rack_FromObject(object obj, saddr_t offset) except NULL

cdef size_t Rack_Length(const Rack_* that) nogil
cdef (addr_t, addr_t) Rack_BoundSlice(const Rack_* that, addr_t start, addr_t endex) nogil

cdef Rack_* Rack_Shift_(Rack_* that, addr_t offset) except NULL
cdef Rack_* Rack_Shift(Rack_* that, saddr_t offset) except NULL

cdef bint Rack_Eq(const Rack_* that, const Rack_* other) except -1

cdef Rack_* Rack_Reserve_(Rack_* that, size_t offset, size_t size) except NULL
cdef Rack_* Rack_Delete_(Rack_* that, size_t offset, size_t size) except NULL
cdef Rack_* Rack_Clear(Rack_* that) except NULL
cdef Rack_* Rack_Consolidate(Rack_* that) except NULL

cdef Block_** Rack_At_(Rack_* that, size_t offset) nogil
cdef const Block_** Rack_At__(const Rack_* that, size_t offset) nogil

cdef Block_* Rack_First_(Rack_* that) nogil
cdef const Block_* Rack_First__(const Rack_* that) nogil

cdef Block_* Rack_Last_(Rack_* that) nogil
cdef const Block_* Rack_Last__(const Rack_* that) nogil

cdef Block_* Rack_Get__(const Rack_* that, size_t offset) nogil
cdef Block_* Rack_Get_(const Rack_* that, size_t offset) except? NULL
cdef Block_* Rack_Get(const Rack_* that, ssize_t offset) except? NULL

cdef Block_* Rack_Set__(Rack_* that, size_t offset, Block_* value) nogil
cdef vint Rack_Set_(Rack_* that, size_t offset, Block_* value, Block_** backup) except -1
cdef vint Rack_Set(Rack_* that, ssize_t offset, Block_* value, Block_** backup) except -1

cdef Rack_* Rack_Pop__(Rack_* that, Block_** value) except NULL
cdef Rack_* Rack_Pop_(Rack_* that, size_t offset, Block_** value) except NULL
cdef Rack_* Rack_Pop(Rack_* that, ssize_t offset, Block_** value) except NULL
cdef Rack_* Rack_PopLeft(Rack_* that, Block_** value) except NULL

cdef Rack_* Rack_Insert_(Rack_* that, size_t offset, Block_* value) except NULL
cdef Rack_* Rack_Insert(Rack_* that, ssize_t offset, Block_* value) except NULL

cdef Rack_* Rack_Append(Rack_* that, Block_* value) except NULL
cdef Rack_* Rack_AppendLeft(Rack_* that, Block_* value) except NULL

cdef Rack_* Rack_Extend_(Rack_* that, size_t size, Block_** buffer, bint direct) except NULL
cdef Rack_* Rack_Extend(Rack_* that, Rack_* more) except NULL

cdef Rack_* Rack_ExtendLeft_(Rack_* that, size_t size, Block_** buffer, bint direct) except NULL
cdef Rack_* Rack_ExtendLeft(Rack_* that, Rack_* more) except NULL

cdef vint Rack_Read_(const Rack_* that, size_t offset,
                     size_t size, Block_** buffer, bint direct) except -1
cdef Rack_* Rack_Write_(Rack_* that, size_t offset,
                        size_t size, Block_** buffer, bint direct) except NULL

cdef vint Rack_ReadSlice_(const Rack_* that, size_t start, size_t endex,
                          size_t* size_, Block_** buffer, bint direct) except -1
cdef vint Rack_ReadSlice(const Rack_* that, ssize_t start, ssize_t endex,
                         size_t* size_, Block_** buffer, bint direct) except -1

cdef Rack_* Rack_GetSlice_(const Rack_* that, size_t start, size_t endex) except NULL
cdef Rack_* Rack_GetSlice(const Rack_* that, ssize_t start, ssize_t endex) except NULL

cdef Rack_* Rack_WriteSlice_(Rack_* that, size_t start, size_t endex,
                             size_t size, Block_** buffer, bint direct) except NULL
cdef Rack_* Rack_WriteSlice(Rack_* that, ssize_t start, ssize_t endex,
                            size_t size, Block_** buffer, bint direct) except NULL

cdef Rack_* Rack_SetSlice_(Rack_* that, size_t start, size_t endex,
                           Rack_* src, size_t start2, size_t endex2) except NULL
cdef Rack_* Rack_SetSlice(Rack_* that, ssize_t start, ssize_t endex,
                          Rack_* src, ssize_t start2, ssize_t endex2) except NULL

cdef Rack_* Rack_DelSlice_(Rack_* that, size_t start, size_t endex) except NULL
cdef Rack_* Rack_DelSlice(Rack_* that, ssize_t start, ssize_t endex) except NULL

cdef ssize_t Rack_IndexAt(const Rack_* that, addr_t address) except -2
cdef ssize_t Rack_IndexStart(const Rack_* that, addr_t address) except -2
cdef ssize_t Rack_IndexEndex(const Rack_* that, addr_t address) except -2


# =====================================================================================================================

cdef class Rover:
    cdef:
        bint _forward
        bint _infinite
        addr_t _start
        addr_t _endex
        addr_t _address

        Memory _memory
        const Rack_* _blocks
        size_t _block_count
        size_t _block_index
        Block_* _block
        addr_t _block_start
        addr_t _block_endex
        const byte_t* _block_ptr

        size_t _pattern_size
        const byte_t* _pattern_data
        size_t _pattern_offset
        const byte_t[:] _pattern_view
        byte_t _pattern_value

    cdef int next_(self) except -2

    cdef vint dispose_(self) except -1


# ---------------------------------------------------------------------------------------------------------------------

cdef class Memory:  # TODO: make ALL as cdef and move to cython Memory_!!!
    cdef:
        Rack_* _  # C implementation
        addr_t _trim_start
        addr_t _trim_endex
        bint _trim_start_
        bint _trim_endex_

    # TODO: prototype indentation as per _py.py

    cdef bint __eq__same_(self, Memory other) except -1
    cdef bint __eq__raw_(self, size_t data_size, const byte_t* data_ptr) except -1
    cdef bint __eq__view_(self, const byte_t[:] view) except -1
    cdef bint __eq__iter_(self, iterable) except -1

    cdef saddr_t find_unbounded_(self, size_t size, const byte_t* buffer) except -2
    cdef saddr_t find_bounded_(self, size_t size, const byte_t* buffer, addr_t start, addr_t endex) except -2

    cdef saddr_t rfind_unbounded_(self, size_t size, const byte_t* buffer) except -2
    cdef saddr_t rfind_bounded_(self, size_t size, const byte_t* buffer, addr_t start, addr_t endex) except -2

    cdef addr_t count_unbounded_(self, size_t size, const byte_t* buffer) except -1
    cdef addr_t count_bounded_(self, size_t size, const byte_t* buffer, addr_t start, addr_t endex) except -1

    cdef vint append_(self, byte_t value) except -1

    cdef vint extend_same_(self, Memory items, addr_t offset) except -1
    cdef vint extend_raw_(self, size_t items_size, const byte_t* items_ptr, addr_t offset) except -1

    cdef int pop_last_(self) except -2
    cdef int pop_at_(self, addr_t address) except -2

    cdef addr_t start_(self)
    cdef addr_t endex_(self)
    cdef (addr_t, addr_t) span_(self)

    cdef addr_t content_start_(self)
    cdef addr_t content_endex_(self)
    cdef (addr_t, addr_t) content_span_(self)
    cdef addr_t content_size_(self)
    cdef size_t content_parts_(self)

    cdef vint validate_(self) except -1

    cdef (addr_t, addr_t) bound_(self, object start, object endex)

    cdef int peek_(self, addr_t address) except -2

    cdef int poke_none_(self, addr_t address) except -2
    cdef vint poke_none__(self, addr_t address) except -1
    cdef int poke_(self, addr_t address, byte_t value) except -2

    cdef Memory extract_(self, addr_t start, addr_t endex,
                         size_t pattern_size, const byte_t* pattern_ptr,
                         saddr_t step, bint bound)

    cdef vint shift_left_(self, addr_t offset, list backups) except -1
    cdef vint shift_right_(self, addr_t offset, list backups) except -1

    cdef vint reserve_(self, addr_t address, addr_t size, list backups) except -1

    cdef BlockView _memview(self)

    cdef Memory copy_(self)

    cdef bint _insert_(self, addr_t address, size_t size, const byte_t* data, bint shift_after) except -1

    cdef bint _erase_(self, addr_t start, addr_t endex, bint shift_after, bint merge_deletion) except -1

    cdef vint insert_same_(self, addr_t address, Memory data, list backups) except -1
    cdef vint insert_raw_(self, addr_t address, size_t data_size, const byte_t* data_ptr, list backups) except -1

    cdef vint delete_(self, addr_t start, addr_t endex, list backups) except -1

    cdef vint clear_(self, addr_t start, addr_t endex, list backups) except -1

    cdef vint _pretrim_start_(self, addr_t endex_max, addr_t size, list backups) except -1
    cdef vint _pretrim_endex_(self, addr_t start_min, addr_t size, list backups) except -1

    cdef vint _crop_(self, addr_t start, addr_t endex, list backups) except -1

    cdef vint write_same_(self, addr_t address, Memory data, bint clear, list backups) except -1
    cdef vint write_raw_(self, addr_t address, size_t data_size, const byte_t* data_ptr, list backups) except -1

    cdef vint fill_(self, addr_t start, addr_t endex, Block_** pattern, list backups, addr_t start_) except -1

    cdef vint flood_(self, addr_t start, addr_t endex, Block_** pattern, list backups) except -1
