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

r"""Common stuff, shared across modules."""

import abc
import collections.abc
from typing import Any
from typing import ByteString
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover
    TypeAlias = Any  # Python < 3.10

BytesLike: TypeAlias = Union[ByteString, memoryview]
BytesFactory: TypeAlias = Callable[..., BytesLike]


class BaseBytesMethods(ByteString, collections.abc.Sequence):
    r"""Provides useful methods to a byte buffer.

    Python's :obj:`memoryview` and most *byte-like* objects do not provide many
    useful methods found instead within the :obj:`bytes` or :obj:`str` APIs.

    This wrapper class adds a low-level implementation of those methods to
    anything supporting the *buffer protocol*.

    Arguments:
        wrapped (*byte-like*): The target object supporting the
            *buffer protocol*.

    Examples:
        >>> from cbytesparse import BytesMethods
        >>> import numpy
        >>> numbers = list(b'ABC')
        >>> numbers
        [65, 66, 67]
        >>> data = numpy.array(numbers, dtype=numpy.ubyte)
        >>> data
        array([65, 66, 67], dtype=uint8)
        >>> data.lower()  # noqa
        Traceback (most recent call last):
            ...
        AttributeError: 'numpy.ndarray' object has no attribute 'lower'
        >>> wrapped = BytesMethods(data)  # noqa
        >>> bytes(wrapped.lower())
        b'abc'
        >>> wrapped = BytesMethods(memoryview(data))
        >>> bytes(wrapped.lower())
        b'abc'
    """

    @abc.abstractmethod
    def __bool__(
        self,
    ) -> bool:
        r"""Has any items.

        Returns:
            bool: Has any items.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(b'')
            >>> bool(instance)
            False

            >>> instance = BytesMethods(b'Hello, World!')
            >>> bool(instance)
            True
        """
        ...

    @abc.abstractmethod
    def __bytes__(
        self,
    ) -> bytes:
        r"""Creates a bytes clone.

        Returns:
            :obj:`bytes`: Cloned data.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(bytearray(b''))
            >>> bytes(instance)
            b''

            >>> instance = BytesMethods(bytearray(b'Hello, World!'))
            >>> bytes(instance)
            b'Hello, World!'
        """
        ...

    @abc.abstractmethod
    def __contains__(
        self,
        token: Union[BytesLike, int],
    ) -> bool:
        r"""Checks if some items are contained.

        Arguments:
            token (byte-like):
                Token to find.

        Returns:
            bool: Token is contained.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(b'Hello, World!')
            >>> b'World' in instance
            True
            >>> b'$' in instance
            False
            >>> ord('o') in instance
            True
            >>> ord('$') in instance
            False
        """
        ...

    @abc.abstractmethod
    def __delitem__(
        self,
        key: Any,
    ) -> None:
        ...

    @abc.abstractmethod
    def __eq__(
        self,
        other: Any,
    ) -> bool:
        r"""Equality comparison.

        Arguments:
            other (byte-like):
                Data to compare with `self`.

        Returns:
            bool: `self` is equal to `other`.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> data = bytearray(b'Hello, World!')
            >>> instance = BytesMethods(data)
            >>> instance == data
            True
            >>> instance == memoryview(data)
            True
            >>> instance == b'Something else'
            False
        """
        ...

    @abc.abstractmethod
    def __ge__(
        self,
        other: Any,
    ) -> bool:
        ...

    @abc.abstractmethod
    def __getitem__(
        self,
        key: Any,
    ) -> Any:
        r"""Gets data.

        Arguments:
            key (slice or int):
                Selection range or address.
                If it is a :obj:`slice` with bytes-like `step`, the latter is
                interpreted as the filling pattern.

        Returns:
            items: Items from the requested range.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(b'Hello, World!')
            >>> instance[7]  # -> ord('W') = 87
            121
            >>> bytes(instance[:3])
            b'Hel
            >>> bytes(instance[3:10])
            b'lo, Wor'
            >>> bytes(instance[-1:])
            b'!'
            >>> bytes(instance[2:10:3])
            b'l,o'
            >>> bytes(instance[3:10:2])
            b'l,Wr'
        """
        ...

    @abc.abstractmethod
    def __gt__(
        self,
        other: Any,
    ) -> bool:
        ...

    @abc.abstractmethod
    def __init__(
        self,
        wrapped: Optional[BytesLike],
    ):
        ...

    @abc.abstractmethod
    def __iter__(
        self,
    ) -> Iterable[int]:
        r"""Iterates over values.

        Yields:
            int: Value as byte integer.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(b'Hello, World!')
            >>> list(instance)
            [72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33]
        """
        ...

    @abc.abstractmethod
    def __le__(
        self,
        other: Any,
    ) -> bool:
        ...

    @abc.abstractmethod
    def __len__(
        self,
    ) -> int:
        r"""Actual length.

        Computes the actual length of the wrapped data object.

        Returns:
            int: Data length.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(b'')
            >>> len(instance)
            0

            >>> instance = BytesMethods(bytes(7))
            >>> len(instance)
            7

            >>> instance = BytesMethods(memoryview(b'Hello, World!'))
            >>> len(instance)
            13
        """
        ...

    @abc.abstractmethod
    def __lt__(
        self,
        other: Any,
    ) -> bool:
        ...

    @abc.abstractmethod
    def __ne__(
        self,
        other: Any,
    ) -> bool:
        r"""Inquality comparison.

        Arguments:
            other (byte-like):
                Data to compare with `self`.

        Returns:
            bool: `self` is not equal to `other`.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> data = bytearray(b'Hello, World!')
            >>> instance = BytesMethods(data)
            >>> instance != data
            False
            >>> instance != memoryview(data)
            False
            >>> instance != b'Something else'
            True
        """
        ...

    @abc.abstractmethod
    def __reversed__(
        self,
    ) -> Iterable[int]:
        r"""Iterates over values, reversed order.

        Yields:
            int: Value as byte integer.

        Examples:
            >>> from cbytesparse import BytesMethods

            >>> instance = BytesMethods(b'Hello, World!')
            >>> list(instance)
            [72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33]
            >>> list(reversed(instance))
            [33, 100, 108, 114, 111, 87, 32, 44, 111, 108, 108, 101, 72]
        """
        ...

    @abc.abstractmethod
    def __setitem__(
        self,
        key: Any,
        value: Any,
    ) -> None:
        ...

    @abc.abstractmethod
    def __sizeof__(
        self,
    ) -> int:
        ...

    @property
    @abc.abstractmethod
    def c_contiguous(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def capitalize(
        self,
    ) -> 'BaseBytesMethods':
        ...

    @abc.abstractmethod
    def center(
        self,
        width: int,
        fillchar: BytesLike = b' ',
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @abc.abstractmethod
    def contains(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:
        ...

    @property
    @abc.abstractmethod
    def contiguous(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def count(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:
        ...

    @abc.abstractmethod
    def decode(
        self,
        encoding: str = 'utf-8',
        errors: str = 'strict',
    ) -> str:
        ...

    @abc.abstractmethod
    def endswith(
        self,
        token: BytesLike,
    ) -> bool:
        ...

    @property
    @abc.abstractmethod
    def f_contiguous(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def find(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:
        ...

    @property
    @abc.abstractmethod
    def format(
        self,
    ) -> str:
        ...

    @abc.abstractmethod
    def index(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:
        ...

    @abc.abstractmethod
    def isalnum(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isalpha(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isascii(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isdecimal(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isdigit(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isidentifier(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def islower(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isnumeric(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isprintable(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isspace(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def istitle(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def isupper(
        self,
    ) -> bool:
        ...

    @property
    @abc.abstractmethod
    def itemsize(
        self,
    ) -> int:
        ...

    @abc.abstractmethod
    def ljust(
        self,
        width: int,
        fillchar: BytesLike = b' ',
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @abc.abstractmethod
    def lstrip(
        self,
        chars: Optional[BytesLike] = None,
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @abc.abstractmethod
    def lower(
        self,
    ) -> 'BaseBytesMethods':
        ...

    @property
    @abc.abstractmethod
    def nbytes(
        self,
    ) -> int:
        ...

    @property
    @abc.abstractmethod
    def ndim(
        self,
    ) -> int:
        ...

    @property
    @abc.abstractmethod
    def obj(
        self,
    ) -> Optional[BytesLike]:
        ...

    @abc.abstractmethod
    def partition(
        self,
        sep: BytesLike,
        factory: BytesFactory = bytes,
    ) -> Tuple[BytesLike, BytesLike, BytesLike]:
        ...

    @property
    @abc.abstractmethod
    def readonly(
        self,
    ) -> bool:
        ...

    @abc.abstractmethod
    def release(
        self,
    ) -> None:
        ...

    @abc.abstractmethod
    def removeprefix(
        self,
        prefix: BytesLike,
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @abc.abstractmethod
    def removesuffix(
        self,
        suffix: BytesLike,
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @abc.abstractmethod
    def replace(
        self,
        old: BytesLike,
        new: BytesLike,
        count: Optional[int] = None,
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> 'BaseBytesMethods':
        ...

    @abc.abstractmethod
    def rfind(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:
        ...

    @abc.abstractmethod
    def rindex(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:
        ...

    @abc.abstractmethod
    def rjust(
        self,
        width: int,
        fillchar: BytesLike = b' ',
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @abc.abstractmethod
    def rpartition(
        self,
        sep: BytesLike,
        factory: BytesFactory = bytes,
    ) -> Tuple[BytesLike, BytesLike, BytesLike]:
        ...

    @abc.abstractmethod
    def rstrip(
        self,
        chars: Optional[BytesLike] = None,
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @property
    @abc.abstractmethod
    def shape(
        self,
    ) -> Tuple[int]:
        ...

    @abc.abstractmethod
    def startswith(
        self,
        token: BytesLike,
    ) -> bool:
        ...

    @property
    @abc.abstractmethod
    def strides(
        self,
    ) -> Tuple[int]:
        ...

    @abc.abstractmethod
    def strip(
        self,
        chars: Optional[BytesLike] = None,
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...

    @property
    @abc.abstractmethod
    def suboffsets(
        self,
    ) -> Tuple:
        ...

    @abc.abstractmethod
    def swapcase(
        self,
    ) -> 'BaseBytesMethods':
        ...

    @abc.abstractmethod
    def title(
        self,
    ) -> 'BaseBytesMethods':
        ...

    @abc.abstractmethod
    def tobytes(
        self,
    ) -> bytes:
        ...

    @abc.abstractmethod
    def tolist(
        self,
    ) -> List[int]:
        ...

    @abc.abstractmethod
    def translate(
        self,
        table: BytesLike,
    ) -> 'BaseBytesMethods':
        ...

    @abc.abstractmethod
    def upper(
        self,
    ) -> 'BaseBytesMethods':
        ...

    @abc.abstractmethod
    def zfill(
        self,
        width: int,
        factory: BytesFactory = bytes,
    ) -> BytesLike:
        ...


class BaseInplaceView(BaseBytesMethods):
    r"""Provides inplace methods to a byte buffer.

    Standard Python *byte-like* objects (e.g. :obj:`memoryview` and
    :obj:`bytearray`) only provide a very small set of methods for
    *inplace* editing of their underlying bytes buffer.

    This wrapper class adds a low-level implementation of those methods to
    anything supporting the *buffer protocol* with a mutable buffer.

    Note:
        Editing support is only limited to the exising buffer items, i.e. the
        wrapper and its underlying buffer cannot be resized via the methods
        provided by this class, just like with standard memory views.

    Arguments:
        wrapped (*byte-like*): The target object supporting the
            *buffer protocol*.

    Examples:
        >>> from cbytesparse import InplaceView
        >>> buffer = bytearray(b'Hello, World!')
        >>> wrapped = InplaceView(buffer)
        >>> wrapped.lower()
        >>> buffer
        bytearray(b'hello, world!')
        >>> wrapped.replace(b'l', b'p')
        >>> buffer
        bytearray(b'heppo, worpd!')
        >>> wrapped.find(b'w')
        7
        >>> wrapped.rfind(b'o')
        8
        >>> wrapped.count(b'o')
        2
    """

    @abc.abstractmethod
    def __init__(
        self,
        wrapped: Optional[BytesLike],
    ):
        ...

    @abc.abstractmethod
    def toreadonly(
        self,
    ) -> 'BaseInplaceView':
        ...
