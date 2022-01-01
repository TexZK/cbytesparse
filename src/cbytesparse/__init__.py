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

r"""Utilities for sparse blocks of bytes.

Blocks are a useful way to describe sparse linear data.

The audience of this module are most importantly those who have to manage
sparse blocks of bytes, where a very broad addressing space (*e.g.* 4 GiB)
is used only in some sparse parts (*e.g.* physical memory addressing in a
microcontroller).

This module also provides the :obj:`Memory` class, which is a handy wrapper
around blocks, giving the user the flexibility of most operations of a
:obj:`bytearray` on sparse byte-like chunks.

A `block` is a tuple ``(start, data)`` where `start` is the start address and
`data` is the container of data items (e.g. :obj:`bytearray`).
The length of the block is ``len(data)``.
Actually, the module uses lists instead of tuples, because the latter are
mutables, thus can be changed in-place, without reallocation.

In this module it is common to require *spaces* blocks, *i.e.* blocks
in which a block ``b`` does not start immediately after block ``a``:

+---+---+---+---+---+---+---+---+---+
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
+===+===+===+===+===+===+===+===+===+
|   |[A | B | C]|   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |[x | y | z]|   |
+---+---+---+---+---+---+---+---+---+

>>> a = [1, b'ABC']
>>> b = [5, b'xyz']

Instead, *overlapping* blocks have at least an addressed cell occupied by
more items:

+---+---+---+---+---+---+---+---+---+
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
+===+===+===+===+===+===+===+===+===+
|   |[A | B | C]|   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+
|   |   |   |[x | y | z]|   |   |   |
+---+---+---+---+---+---+---+---+---+
|[# | #]|   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+
|   |   |[!]|   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+

>>> a = [1, b'ABC']
>>> b = [3, b'xyz']
>>> c = [0, b'##']
>>> d = [2, b'!']

Contiguous blocks are *non-overlapping*.

+---+---+---+---+---+---+---+---+---+
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
+===+===+===+===+===+===+===+===+===+
|   |[A | B | C]|   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+
|   |   |   |   |[x | y | z]|   |   |
+---+---+---+---+---+---+---+---+---+

>>> a = [1, b'ABC']
>>> b = [4, b'xyz']

This module often deals with *sequences* of blocks, typically :obj:`list`
objects containing blocks:

>>> seq = [[1, b'ABC'], [5, b'xyz']]

Sometimes *sequence generators* are allowed, in that blocks of the sequence
are yielded on-the-fly by a generator, like `seq_gen`:

>>> seq_gen = ([i, (i + 0x21).to_bytes(1, 'little') * 3] for i in range(0, 15, 5))
>>> list(seq_gen)
[[0, b'!!!'], [5, b'&&&'], [10, b'+++']]

It is required that sequences are ordered, which means that a block ``b`` must
follow a block ``a`` which end address is lesser than the `start` of ``b``,
like in:

>>> a = [1, b'ABC']
>>> b = [5, b'xyz']
>>> a[0] + len(a[1]) <= b[0]
True

"""

__version__ = '0.0.2'

from ._c import *  # noqa: F401, F403
