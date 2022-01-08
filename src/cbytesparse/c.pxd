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
# from cpython.mem cimport PyMem_Calloc  # FIXME: Not yet provided by the current Cython
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


cdef void* PyMem_Calloc(size_t nelem, size_t elsize)  # FIXME: Not yet provided by the current Cython


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

    cdef vint check_(self) except -1
    cdef vint dispose_(self) except -1


# =====================================================================================================================

cdef extern from *:
    r"""
    typedef struct Rack_ {
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

cdef extern from *:
    r"""
    typedef struct Memory_ {
        Rack_* blocks;
        addr_t trim_start;
        addr_t trim_endex;
        int trim_start_;  // bint
        int trim_endex_;  // bint
    } Memory_;

    #define Memory_HEADING (sizeof(Memory_))
    """

    ctypedef struct Memory_:
        # Stored memory blocks
        Rack_* blocks

        # Trimming start address, if _trim_start_
        addr_t trim_start

        # Triming exclusive end address, if _trim_endex_
        addr_t trim_endex

        # Enables trimming start address
        bint trim_start_

        # Enables timming exclusive end address
        bint trim_endex_

    size_t Memory_HEADING

cdef class Memory


cdef Memory Memory_AsObject(Memory_* that)

cdef Memory_* Memory_Alloc() except NULL
cdef Memory_* Memory_Free(Memory_* that) except? NULL

cdef Memory_* Memory_Create(
    Memory_* memory,
    const byte_t[:] data,
    object offset,
    object blocks,
    object start,
    object endex,
    bint copy,
    bint validate,
) except NULL

cdef bint Memory_EqSame_(const Memory_* that, const Memory_* other) except -1
cdef bint Memory_EqRaw_(const Memory_* that, size_t data_size, const byte_t* data_ptr) except -1
cdef bint Memory_EqView_(const Memory_* that, const byte_t[:] view) except -1
cdef bint Memory_EqIter_(const Memory_* that, object iterable) except -1
cdef bint Memory_Eq(const Memory_* that, object other) except -1

cdef Memory_* Memory_Add(const Memory_* that, object value) except NULL
cdef Memory_* Memory_IAdd(Memory_* that, object value) except NULL

cdef Memory_* Memory_Mul(const Memory_* that, addr_t times) except NULL
cdef Memory_* Memory_IMul(Memory_* that, addr_t times) except NULL

cdef addr_t Memory_Length(const Memory_* that) nogil
cdef bint Memory_IsEmpty(const Memory_* that) nogil

cdef object Memory_ObjFind(const Memory_* that, object item, object start, object endex)
cdef object Memory_RevObjFind(const Memory_* that, object item, object start, object endex)

cdef addr_t Memory_FindUnbounded_(const Memory_* that, size_t size, const byte_t* buffer) except? -1
cdef addr_t Memory_FindBounded_(const Memory_* that, size_t size, const byte_t* buffer,
                                addr_t start, addr_t endex) except? -1
cdef object Memory_Find(const Memory_* that, object item, object start, object endex)

cdef addr_t Memory_RevFindUnbounded_(const Memory_* that, size_t size, const byte_t* buffer) except? -1
cdef addr_t Memory_RevFindBounded_(const Memory_* that, size_t size, const byte_t* buffer,
                                   addr_t start, addr_t endex) except? -1
cdef object Memory_RevFind(const Memory_* that, object item, object start, object endex)

cdef object Memory_Index(const Memory_* that, object item, object start, object endex)
cdef object Memory_RevIndex(const Memory_* that, object item, object start, object endex)

cdef bint Memory_Contains(const Memory_* that, object item) except -1

cdef addr_t Memory_CountUnbounded_(const Memory_* that, size_t size, const byte_t* buffer) except? -1
cdef addr_t Memory_CountBounded_(const Memory_* that, size_t size, const byte_t* buffer,
                                 addr_t start, addr_t endex) except? -1
cdef addr_t Memory_Count(const Memory_* that, object item, object start, object endex) except? -1

cdef object Memory_GetItem(const Memory_* that, object key)
cdef object Memory_SetItem(Memory_* that, object key, object value)
cdef vint Memory_DelItem(Memory_* that, object key) except -1

cdef vint Memory_Append_(Memory_* that, byte_t value) except -1
cdef vint Memory_Append(Memory_* that, object item) except -1

cdef vint Memory_ExtendSame_(Memory_* that, const Memory_* items, addr_t offset) except -1
cdef vint Memory_ExtendRaw_(Memory_* that, size_t items_size, const byte_t* items_ptr, addr_t offset) except -1
cdef vint Memory_Extend(Memory_* that, object items, object offset) except -1

cdef int Memory_PopLast_(Memory_* that) except -2
cdef int Memory_PopAt_(Memory_* that, addr_t address) except -2
cdef object Memory_Pop(Memory_* that, object address)

cdef BlockView Memory_View(const Memory_* that, addr_t start, addr_t endex)

cdef Memory_* Memory_Copy(const Memory_* that) except NULL

cdef bint Memory_Contiguous(const Memory_* that) nogil

cdef object Memory_GetTrimStart(const Memory_* that)
cdef vint Memory_SetTrimStart(Memory_* that, object trim_start) except -1

cdef object Memory_GetTrimEndex(const Memory_* that)
cdef vint Memory_SetTrimEndex(Memory_* that, object trim_endex) except -1

cdef object Memory_GetTrimSpan(const Memory_* that)
cdef vint Memory_SetTrimSpan(Memory_* that, object trim_span) except -1

cdef addr_t Memory_Start(const Memory_* that) nogil
cdef addr_t Memory_Endex(const Memory_* that) nogil
cdef object Memory_Endin(const Memory_* that)

cdef addr_t Memory_ContentStart(const Memory_* that) nogil
cdef addr_t Memory_ContentEndex(const Memory_* that) nogil
cdef object Memory_ContentEndin(const Memory_* that)
cdef addr_t Memory_ContentSize(const Memory_* that) nogil
cdef size_t Memory_ContentParts(const Memory_* that) nogil

cdef vint Memory_Validate(const Memory_* that) except -1

cdef (addr_t, addr_t) Memory_Bound_(const Memory_* that, addr_t start, addr_t endex,
                                    bint start_, bint endex_) nogil
cdef (addr_t, addr_t) Memory_Bound(const Memory_* that, object start, object endex) except *

cdef int Memory_Peek_(const Memory_* that, addr_t address) except -2

cdef object Memory_Peek(const Memory_* that, object address)

cdef vint Memory_PokeNone_(Memory_* that, addr_t address) except -1
cdef vint Memory_Poke_(Memory_* that, addr_t address, byte_t item) except -1
cdef vint Memory_Poke(Memory_* that, object address, object item) except -1

cdef Memory_* Memory_Extract__(const Memory_* that, addr_t start, addr_t endex,
                               size_t pattern_size, const byte_t* pattern_ptr,
                               saddr_t step, bint bound) except NULL

cdef object Memory_Extract_(const Memory_* that, addr_t start, addr_t endex,
                            size_t pattern_size, const byte_t* pattern_ptr,
                            saddr_t step, bint bound)

cdef object Memory_Extract(const Memory_* that, object start, object endex,
                           object pattern, object step, bint bound)

cdef vint Memory_ShiftLeft_(Memory_* that, addr_t offset) except -1
cdef vint Memory_ShiftRight_(Memory_* that, addr_t offset) except -1
cdef vint Memory_Shift(Memory_* that, object offset) except -1

cdef vint Memory_Reserve_(Memory_* that, addr_t address, addr_t size) except -1
cdef vint Memory_Reserve(Memory_* that, object address, object size) except -1

cdef vint Memory_Place__(Memory_* that, addr_t address, size_t size, const byte_t* buffer,
                         bint shift_after) except -1
cdef vint Memory_Erase__(Memory_* that, addr_t start, addr_t endex, bint shift_after) except -1

cdef vint Memory_InsertSame_(Memory_* that, addr_t address, Memory_* data) except -1
cdef vint Memory_InsertRaw_(Memory_* that, addr_t address, size_t data_size, const byte_t* data_ptr) except -1
cdef vint Memory_Insert(Memory_* that, object address, object data) except -1

cdef vint Memory_Delete_(Memory_* that, addr_t start, addr_t endex) except -1
cdef vint Memory_Delete(Memory_* that, object start, object endex) except -1

cdef vint Memory_Clear_(Memory_* that, addr_t start, addr_t endex) except -1
cdef vint Memory_Clear(Memory_* that, object start, object endex) except -1

cdef vint Memory_PretrimStart_(Memory_* that, addr_t endex_max, addr_t size) except -1
cdef vint Memory_PretrimStart(Memory_* that, object endex_max, object size) except -1

cdef vint Memory_PretrimEndex_(Memory_* that, addr_t start_min, addr_t size) except -1
cdef vint Memory_PretrimEndex(Memory_* that, object start_min, object size) except -1

cdef vint Memory_Crop_(Memory_* that, addr_t start, addr_t endex) except -1
cdef vint Memory_Crop(Memory_* that, object start, object endex) except -1

cdef vint Memory_WriteSame_(Memory_* that, addr_t address, const Memory_* data, bint clear) except -1
cdef vint Memory_WriteRaw_(Memory_* that, addr_t address, size_t data_size, const byte_t* data_ptr) except -1
cdef vint Memory_Write(Memory_* that, object address, object data, bint clear) except -1

cdef vint Memory_Fill_(Memory_* that, addr_t start, addr_t endex, Block_** pattern, addr_t start_) except -1
cdef vint Memory_Fill(Memory_* that, object start, object endex, object pattern) except -1

cdef vint Memory_Flood_(Memory_* that, addr_t start, addr_t endex, Block_** pattern) except -1
cdef vint Memory_Flood(Memory_* that, object start, object endex, object pattern) except -1

cdef list Memory_AsViews(const Memory_* that)

cdef list Memory_ToBlocks(const Memory_* that)


# =====================================================================================================================

cdef extern from *:
    r"""
    typedef struct Rover_ {
        // Sorted by data size
        addr_t start;
        addr_t endex;
        addr_t address;
        addr_t block_start;
        addr_t block_endex;

        const Memory_* memory;
        const byte_t* pattern_data;
        Block_* block;
        const byte_t* block_ptr;

        size_t pattern_size;
        size_t pattern_offset;
        size_t block_count;
        size_t block_index;

        int forward;  // bint
        int infinite;  // bint
    } Rover_;

    #define Rover_HEADING (sizeof(Rover_))
    """

    ctypedef struct Rover_:
        # Sorted by data size
        addr_t start
        addr_t endex
        addr_t address
        addr_t block_start
        addr_t block_endex

        const Memory_* memory
        const byte_t* pattern_data
        Block_* block
        const byte_t* block_ptr

        size_t pattern_size
        size_t pattern_offset
        size_t block_count
        size_t block_index

        bint forward
        bint infinite

    size_t Rover_HEADING


cdef Rover_* Rover_Alloc() except NULL
cdef Rover_* Rover_Free(Rover_* that) except? NULL

cdef Rover_* Rover_Create(
    const Memory_* memory,
    addr_t start,
    addr_t endex,
    size_t pattern_size,
    const byte_t* pattern_data,
    bint forward,
    bint infinite,
) except NULL

cdef addr_t Rover_Length(const Rover_* that) nogil

cdef bint Rover_HasNext(const Rover_* that) nogil
cdef int Rover_Next_(Rover_* that) except -2
cdef object Rover_Next(Rover_* that)

cdef vint Rover_Dispose(Rover_* that) except -1

cdef bint Rover_Forward(const Rover_* that) nogil
cdef bint Rover_Infinite(const Rover_* that) nogil
cdef addr_t Rover_Address(const Rover_* that) nogil
cdef addr_t Rover_Start(const Rover_* that) nogil
cdef addr_t Rover_Endex(const Rover_* that) nogil


# =====================================================================================================================

cdef class Memory:
    cdef:
        Memory_* _  # C implementation
        addr_t _trim_start
        addr_t _trim_endex
        bint _trim_start_
        bint _trim_endex_
