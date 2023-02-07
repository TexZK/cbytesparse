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

r"""Python wrappers.

Useful for dynamically declared stuff, e.g. docstrings and return types.
"""

from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast as _cast

from bytesparse.base import Address
from bytesparse.base import AddressValueMapping
from bytesparse.base import AnyBytes
from bytesparse.base import Block
from bytesparse.base import BlockIndex
from bytesparse.base import BlockIterable
from bytesparse.base import BlockList
from bytesparse.base import ClosedInterval
from bytesparse.base import EllipsisType
from bytesparse.base import ImmutableMemory
from bytesparse.base import MutableBytesparse
from bytesparse.base import MutableMemory
from bytesparse.base import OpenInterval
from bytesparse.base import Value

from .base import BaseBytesMethods
from .base import BaseInplaceView
from .base import BytesFactory
from .base import BytesLike

# noinspection PyUnresolvedReferences,PyPackageRequirements
from .c import BytesMethods as _CythonBytesMethods  # isort:skip
# noinspection PyUnresolvedReferences,PyPackageRequirements
from .c import InplaceView as _CythonInplaceView  # isort:skip
# noinspection PyUnresolvedReferences,PyPackageRequirements
from .c import Memory as _CythonMemory  # isort:skip
# noinspection PyUnresolvedReferences,PyPackageRequirements,PyPep8Naming
from .c import bytesparse as _CythonBytesparse  # isort:skip

try:
    from typing import Self
except ImportError:  # pragma: no cover  # Python < 3.11
    Self = None  # dummy
    _MemorySelf = TypeVar('_MemorySelf', bound='Memory')
    _BytesparseSelf = TypeVar('_BytesparseSelf', bound='bytesparse')
else:  # pragma: no cover
    _MemorySelf = Self
    _BytesparseSelf = Self


class BytesMethods(BaseBytesMethods):
    __doc__ = BaseBytesMethods.__doc__

    def __bool__(
        self,
    ) -> bool:

        return self._impl.__bool__()

    def __bytes__(
        self,
    ) -> bytes:

        return self._impl.__bytes__()

    def __contains__(
        self,
        token: Union[BytesLike, int],
    ) -> bool:

        return self._impl.__contains__(token)

    def __delitem__(
        self,
        key: Any,
    ) -> None:

        self._impl.__delitem__(key)

    def __eq__(
        self,
        other: Any,
    ) -> bool:

        return self._impl == other

    def __ge__(
        self,
        other: Any,
    ) -> bool:

        return self._impl >= other

    def __getitem__(
        self,
        key: Any,
    ) -> Any:

        return self._impl.__getitem__(key)

    def __gt__(
        self,
        other: Any,
    ) -> bool:

        return self._impl > other

    def __init__(
        self,
        wrapped: Optional[BytesLike],
    ):

        self._impl: _CythonBytesMethods = _CythonBytesMethods(wrapped)

    def __iter__(
        self,
    ) -> Iterable[int]:

        yield from self._impl.__iter__()

    def __le__(
        self,
        other: Any,
    ) -> bool:

        return self._impl <= other

    def __len__(
        self,
    ) -> int:

        return self._impl.__len__()

    def __lt__(
        self,
        other: Any,
    ) -> bool:

        return self._impl < other

    def __ne__(
        self,
        other: Any,
    ) -> bool:

        return self._impl != other

    def __reversed__(
        self,
    ) -> Iterable[int]:

        yield from self._impl.__reversed__()

    def __setitem__(
        self,
        key: Any,
        value: Any,
    ) -> None:

        self._impl.__setitem__(key, value)

    def __sizeof__(
        self,
    ) -> int:

        return self._impl.__sizeof__()

    @property
    def c_contiguous(
        self,
    ) -> bool:

        return self._impl.c_contiguous

    def capitalize(
        self,
    ) -> BaseBytesMethods:

        return self._impl.capitalize()

    def center(
        self,
        width: int,
        fillchar: BytesLike = b' ',
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.center(width, fillchar, factory=factory)

    def contains(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:

        return self._impl.contains(token, start=start, endex=endex)

    @property
    def contiguous(
        self,
    ) -> bool:

        return self._impl.contiguous

    def count(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:

        return self._impl.count(token, start=start, endex=endex)

    def decode(
        self,
        encoding: str = 'utf-8',
        errors: str = 'strict',
    ) -> str:

        return self._impl.decode(encoding=encoding, errors=errors)

    def endswith(
        self,
        token: BytesLike,
    ) -> bool:

        return self._impl.endswith(token)

    @property
    def f_contiguous(
        self,
    ) -> bool:

        return self._impl.f_contiguous

    def find(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:

        return self._impl.find(token, start=start, endex=endex)

    @property
    def format(
        self,
    ) -> str:

        return self._impl.format

    def index(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:

        return self._impl.index(token, start=start, endex=endex)

    def isalnum(
        self,
    ) -> bool:

        return self._impl.isalnum()

    def isalpha(
        self,
    ) -> bool:

        return self._impl.isalpha()

    def isascii(
        self,
    ) -> bool:

        return self._impl.isascii()

    def isdecimal(
        self,
    ) -> bool:

        return self._impl.isdecimal()

    def isdigit(
        self,
    ) -> bool:

        return self._impl.isdigit()

    def isidentifier(
        self,
    ) -> bool:

        return self._impl.isidentifier()

    def islower(
        self,
    ) -> bool:

        return self._impl.islower()

    def isnumeric(
        self,
    ) -> bool:

        return self._impl.isnumeric()

    def isprintable(
        self,
    ) -> bool:

        return self._impl.isprintable()

    def isspace(
        self,
    ) -> bool:

        return self._impl.isspace()

    def istitle(
        self,
    ) -> bool:

        return self._impl.istitle()

    def isupper(
        self,
    ) -> bool:

        return self._impl.isupper()

    @property
    def itemsize(
        self,
    ) -> int:

        return self._impl.itemsize

    def ljust(
        self,
        width: int,
        fillchar: BytesLike = b' ',
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.ljust(width, fillchar, factory=factory)

    def lstrip(
        self,
        chars: Optional[BytesLike] = None,
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.lstrip(chars, factory=factory)

    def lower(
        self,
    ) -> BaseBytesMethods:

        return self._impl.lower()

    @staticmethod
    def maketrans(
        chars_from: BytesLike,
        chars_to: BytesLike,
    ) -> bytes:

        return _CythonBytesMethods.maketrans(chars_from, chars_to)

    @property
    def nbytes(
        self,
    ) -> int:

        return self._impl.nbytes

    @property
    def ndim(
        self,
    ) -> int:

        return self._impl.ndim

    @property
    def obj(
        self,
    ) -> Optional[BytesLike]:

        return self._impl.obj

    def partition(
        self,
        sep: BytesLike,
        factory: BytesFactory = bytes,
    ) -> Tuple[BytesLike, BytesLike, BytesLike]:

        return self._impl.partition(sep, factory=factory)

    @property
    def readonly(
        self,
    ) -> bool:

        return self._impl.readonly

    def release(
        self,
    ) -> None:

        self._impl.release()

    def removeprefix(
        self,
        prefix: BytesLike,
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.removeprefix(prefix, factory=factory)

    def removesuffix(
        self,
        suffix: BytesLike,
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.removesuffix(suffix, factory=factory)

    def replace(
        self,
        old: BytesLike,
        new: BytesLike,
        count: Optional[int] = None,
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> BaseBytesMethods:

        return self._impl.replace(old, new, count=count, start=start, endex=endex)

    def rfind(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:

        return self._impl.rfind(token, start=start, endex=endex)

    def rindex(
        self,
        token: Union[BytesLike, int],
        start: Optional[int] = None,
        endex: Optional[int] = None,
    ) -> int:

        return self._impl.rindex(token, start=start, endex=endex)

    def rjust(
        self,
        width: int,
        fillchar: BytesLike = b' ',
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.rjust(width, fillchar, factory=factory)

    def rpartition(
        self,
        sep: BytesLike,
        factory: BytesFactory = bytes,
    ) -> Tuple[BytesLike, BytesLike, BytesLike]:

        return self._impl.rpartition(sep, factory=factory)

    def rstrip(
        self,
        chars: Optional[BytesLike] = None,
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.rstrip(chars, factory=factory)

    @property
    def shape(
        self,
    ) -> Tuple[int]:

        return self._impl.shape

    def startswith(
        self,
        token: BytesLike,
    ) -> bool:

        return self._impl.startswith(token)

    @property
    def strides(
        self,
    ) -> Tuple[int]:

        return self._impl.strides

    def strip(
        self,
        chars: Optional[BytesLike] = None,
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.strip(chars, factory=factory)

    @property
    def suboffsets(
        self,
    ) -> Tuple:

        return self._impl.suboffsets

    def swapcase(
        self,
    ) -> BaseBytesMethods:

        return self._impl.swapcase()

    def title(
        self,
    ) -> BaseBytesMethods:

        return self._impl.title()

    def tobytes(
        self,
    ) -> bytes:

        return self._impl.tobytes()

    def tolist(
        self,
    ) -> List[int]:

        return self._impl.tolist()

    def translate(
        self,
        table: BytesLike,
    ) -> BaseBytesMethods:

        return self._impl.translate(table)

    def upper(
        self,
    ) -> BaseBytesMethods:

        return self._impl.upper()

    def zfill(
        self,
        width: int,
        factory: BytesFactory = bytes,
    ) -> BytesLike:

        return self._impl.zfill(width, factory=factory)


class InplaceView(BytesMethods, BaseInplaceView):
    __doc__ = BaseInplaceView.__doc__

    def __init__(
        self,
        wrapped: Optional[BytesLike],
    ):

        super().__init__(None)  # dummy
        self._impl: _CythonInplaceView = _CythonInplaceView(wrapped)

    def toreadonly(
        self,
    ) -> BaseInplaceView:

        readonly = self.__class__(None)
        readonly._impl = self._impl.toreadonly()
        return readonly


class Memory(MutableMemory):
    __doc__ = MutableMemory.__doc__

    _Memory = _cast(Type[MutableMemory], _CythonMemory)

    def __add__(
        self,
        value: Union[AnyBytes, ImmutableMemory],
    ) -> _MemorySelf:

        impl = self._impl.__add__(value)
        return self._wrap_impl(impl)

    def __bool__(
        self,
    ) -> bool:

        return self._impl.__bool__()

    def __bytes__(
        self,
    ) -> bytes:

        return self._impl.__bytes__()

    def __contains__(
        self,
        item: Union[AnyBytes, Value],
    ) -> bool:

        return self._impl.__contains__(item)

    def __copy__(
        self,
    ) -> _MemorySelf:

        impl = self._impl.__copy__()
        return self._wrap_impl(impl)

    def __deepcopy__(
        self,
    ) -> _MemorySelf:

        impl = self._impl.__deepcopy__()
        return self._wrap_impl(impl)

    def __delitem__(
        self,
        key: Union[Address, slice],
    ) -> None:

        self._impl.__delitem__(key)

    def __eq__(
        self,
        other: Any,
    ) -> bool:

        return self._impl.__eq__(other)

    def __getitem__(
        self,
        key: Union[Address, slice],
    ) -> Any:

        item = self._impl.__getitem__(key)
        if isinstance(item, self._Memory):
            item = self._wrap_impl(item)
        return item

    def __iadd__(
        self,
        value: Union[AnyBytes, ImmutableMemory],
    ) -> _MemorySelf:

        self._impl.__iadd__(value)
        return self

    def __imul__(
        self,
        times: int,
    ) -> _MemorySelf:

        self._impl.__imul__(times)
        return self

    def __init__(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ):

        impl = self._Memory(start=start, endex=endex)
        self._impl = _cast(MutableMemory, impl)

    def __iter__(
        self,
    ) -> Iterator[Optional[Value]]:

        yield from self._impl.__iter__()

    def __len__(
        self,
    ) -> Address:

        return self._impl.__len__()

    def __mul__(
        self,
        times: int,
    ) -> _MemorySelf:

        impl = self._impl.__mul__(times)
        return self._wrap_impl(impl)

    def __repr__(
        self,
    ) -> str:

        return f'<{self.__class__.__name__}[0x{self.start:X}:0x{self.endex:X}]@0x{id(self):X}>'

    def __reversed__(
        self,
    ) -> Iterator[Optional[Value]]:

        yield from self._impl.__reversed__()

    def __setitem__(
        self,
        key: Union[Address, slice],
        value: Optional[Union[AnyBytes, Value]],
    ) -> None:

        self._impl.__setitem__(key, value)

    def __sizeof__(
        self,
    ) -> int:

        return super().__sizeof__() + self._impl.__sizeof__()  # approximate

    def __str__(
        self,
    ) -> str:

        str_ = self._impl.__str__()
        repr_ = self._impl.__repr__()

        if str_ == repr_:
            return self.__repr__()  # self repr style
        else:
            return str_  # custom style

    def _block_index_at(
        self,
        address: Address,
    ) -> Optional[BlockIndex]:

        return self._impl._block_index_at(address)

    def _block_index_endex(
        self,
        address: Address,
    ) -> BlockIndex:

        return self._impl._block_index_endex(address)

    def _block_index_start(
        self,
        address: Address,
    ) -> BlockIndex:

        return self._impl._block_index_start(address)

    def _prebound_endex(
        self,
        start_min: Optional[Address],
        size: Address,
    ) -> None:

        self._impl._prebound_endex(start_min, size)

    def _prebound_endex_backup(
        self,
        start_min: Optional[Address],
        size: Address,
    ) -> ImmutableMemory:

        return self._impl._prebound_endex_backup(start_min, size)

    def _prebound_start(
        self,
        endex_max: Optional[Address],
        size: Address,
    ) -> None:

        self._impl._prebound_start(endex_max, size)

    def _prebound_start_backup(
        self,
        endex_max: Optional[Address],
        size: Address,
    ) -> ImmutableMemory:

        return self._impl._prebound_start_backup(endex_max, size)

    @classmethod
    def _wrap_impl(
        cls,
        impl: MutableMemory,
    ) -> _MemorySelf:

        memory = cls()  # FIXME: use cls.__new__ ???
        memory._impl = impl
        return memory

    def append(
        self,
        item: Union[AnyBytes, Value],
    ) -> None:

        self._impl.append(item)

    # noinspection PyMethodMayBeStatic
    def append_backup(
        self,
    ) -> None:

        return self._impl.append_backup()

    def append_restore(
        self,
    ) -> None:

        self._impl.append_restore()

    def block_span(
        self,
        address: Address,
    ) -> Tuple[Optional[Address], Optional[Address], Optional[Value]]:

        return self._impl.block_span(address)

    def blocks(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Tuple[Address, Union[memoryview, bytearray]]]:

        yield from self._impl.blocks(start=start, endex=endex)

    def bound(
        self,
        start: Optional[Address],
        endex: Optional[Address],
    ) -> ClosedInterval:

        return self._impl.bound(start, endex)

    @ImmutableMemory.bound_endex.getter
    def bound_endex(
        self,
    ) -> Optional[Address]:

        return self._impl.bound_endex

    @bound_endex.setter
    def bound_endex(
        self,
        bound_endex: Optional[Address],
    ) -> None:

        self._impl.bound_endex = bound_endex

    @ImmutableMemory.bound_span.getter
    def bound_span(
        self,
    ) -> OpenInterval:

        return self._impl.bound_span

    @bound_span.setter
    def bound_span(
        self,
        bound_span: Optional[OpenInterval],
    ) -> None:

        self._impl.bound_span = bound_span

    @ImmutableMemory.bound_start.getter
    def bound_start(
        self,
    ) -> Optional[Address]:

        return self._impl.bound_start

    @bound_start.setter
    def bound_start(
        self,
        bound_start: Optional[Address],
    ) -> None:

        self._impl.bound_start = bound_start

    def clear(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        self._impl.clear(start=start, endex=endex)

    def clear_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        return self._impl.clear_backup(start=start, endex=endex)

    def clear_restore(
        self,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.clear_restore(backup)

    @classmethod
    def collapse_blocks(
        cls,
        blocks: BlockIterable,
    ) -> BlockList:

        return cls._Memory.collapse_blocks(blocks)

    def content_blocks(
        self,
        block_index_start: Optional[BlockIndex] = None,
        block_index_endex: Optional[BlockIndex] = None,
        block_index_step: Optional[BlockIndex] = None,
    ) -> Iterator[Union[Tuple[Address, Union[memoryview, bytearray]], Block]]:

        yield from self._impl.content_blocks(
            block_index_start=block_index_start,
            block_index_endex=block_index_endex,
            block_index_step=block_index_step,
        )

    @ImmutableMemory.content_endex.getter
    def content_endex(
        self,
    ) -> Address:

        return self._impl.content_endex

    @ImmutableMemory.content_endin.getter
    def content_endin(
        self,
    ) -> Address:

        return self._impl.content_endin

    def content_items(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Tuple[Address, Value]]:

        yield from self._impl.content_items(start=start, endex=endex)

    def content_keys(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Address]:

        yield from self._impl.content_keys(start=start, endex=endex)

    @ImmutableMemory.content_parts.getter
    def content_parts(
        self,
    ) -> int:

        return self._impl.content_parts

    @ImmutableMemory.content_size.getter
    def content_size(
        self,
    ) -> Address:

        return self._impl.content_size

    @ImmutableMemory.content_span.getter
    def content_span(
        self,
    ) -> ClosedInterval:

        return self._impl.content_span

    @ImmutableMemory.content_start.getter
    def content_start(
        self,
    ) -> Address:

        return self._impl.content_start

    def content_values(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[Value]:

        yield from self._impl.content_values(start=start, endex=endex)

    @ImmutableMemory.contiguous.getter
    def contiguous(
        self,
    ) -> bool:

        return self._impl.contiguous

    def count(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> int:

        return self._impl.count(item, start=start, endex=endex)

    def copy(
        self,
    ) -> _MemorySelf:

        return self.__deepcopy__()

    def crop(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        self._impl.crop(start=start, endex=endex)

    def crop_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Tuple[Optional[ImmutableMemory], Optional[ImmutableMemory]]:

        return self._impl.crop_backup(start=start, endex=endex)

    def crop_restore(
        self,
        backup_start: Optional[ImmutableMemory],
        backup_endex: Optional[ImmutableMemory],
    ) -> None:

        self._impl.crop_restore(backup_start=backup_start, backup_endex=backup_endex)

    def cut(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        bound: bool = True,
    ) -> _MemorySelf:

        memory = self._impl.cut(start=start, endex=endex, bound=bound)
        memory = _cast(MutableMemory, memory)
        return self._wrap_impl(memory)

    def delete(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        self._impl.delete(start=start, endex=endex)

    def delete_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        return self._impl.delete_backup(start=start, endex=endex)

    def delete_restore(
        self,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.delete_restore(backup)

    @ImmutableMemory.endex.getter
    def endex(
        self,
    ) -> Address:

        return self._impl.endex

    @ImmutableMemory.endin.getter
    def endin(
        self,
    ) -> Address:

        return self._impl.endin

    def equal_span(
        self,
        address: Address,
    ) -> Tuple[Optional[Address], Optional[Address], Optional[Value]]:

        return self._impl.equal_span(address)

    def extend(
        self,
        items: Union[AnyBytes, ImmutableMemory],
        offset: Address = 0,
    ) -> None:

        self._impl.extend(items, offset=offset)

    def extend_backup(
        self,
        offset: Address = 0,
    ) -> Address:

        return self._impl.extend_backup(offset=offset)

    def extend_restore(
        self,
        content_endex: Address,
    ) -> None:

        self._impl.extend_restore(content_endex)

    def extract(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
        step: Optional[Address] = None,
        bound: bool = True,
    ) -> _MemorySelf:

        memory = self._impl.extract(start=start, endex=endex, pattern=pattern, step=step, bound=bound)
        return self._wrap_impl(memory)

    def fill(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
    ) -> None:

        self._impl.fill(start=start, endex=endex, pattern=pattern)

    def fill_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        return self._impl.fill_backup(start=start, endex=endex)

    def fill_restore(
        self,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.fill_restore(backup)

    def find(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        return self._impl.find(item, start=start, endex=endex)

    def flood(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        pattern: Union[AnyBytes, Value] = 0,
    ) -> None:

        self._impl.flood(start=start, endex=endex, pattern=pattern)

    def flood_backup(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> List[OpenInterval]:

        return self._impl.flood_backup(start=start, endex=endex)

    def flood_restore(
        self,
        gaps: List[OpenInterval],
    ) -> None:

        self._impl.flood_restore(gaps)

    @classmethod
    def from_blocks(
        cls,
        blocks: BlockList,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> _MemorySelf:

        memory = cls._Memory.from_blocks(
            blocks,
            offset=offset,
            start=start,
            endex=endex,
            copy=copy,
            validate=validate,
        )
        return cls._wrap_impl(memory)

    @classmethod
    def from_bytes(
        cls,
        data: AnyBytes,
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> _MemorySelf:

        return cls._wrap_impl(cls._Memory.from_bytes(
            data,
            offset=offset,
            start=start,
            endex=endex,
            copy=copy,
            validate=validate,
        ))

    @classmethod
    def from_items(
        cls,
        items: Union[AddressValueMapping,
                     Iterable[Tuple[Address, Optional[Value]]],
                     Mapping[Address, Optional[Union[Value, AnyBytes]]],
                     ImmutableMemory],
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        validate: bool = True,
    ) -> _MemorySelf:

        return cls._wrap_impl(cls._Memory.from_items(
            items,
            offset=offset,
            start=start,
            endex=endex,
            validate=validate,
        ))

    @classmethod
    def from_memory(
        cls,
        memory: Union[ImmutableMemory, MutableMemory, 'Memory'],
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        copy: bool = True,
        validate: bool = True,
    ) -> _MemorySelf:

        return cls._wrap_impl(cls._Memory.from_memory(
            memory,
            offset=offset,
            start=start,
            endex=endex,
            copy=copy,
            validate=validate,
        ))

    @classmethod
    def from_values(
        cls,
        values: Iterable[Optional[Value]],
        offset: Address = 0,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
        validate: bool = True,
    ) -> _MemorySelf:

        return cls._wrap_impl(cls._Memory.from_values(
            values,
            offset=offset,
            start=start,
            endex=endex,
            validate=validate,
        ))

    @classmethod
    def fromhex(
        cls,
        string: str,
    ) -> _MemorySelf:

        return cls._wrap_impl(cls._Memory.fromhex(string))

    def gaps(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[OpenInterval]:

        yield from self._impl.gaps(start=start, endex=endex)

    def get(
        self,
        address: Address,
        default: Optional[Value] = None,
    ) -> Optional[Value]:

        return self._impl.get(address, default=default)

    def hex(
        self,
        *args: Any,
    ) -> str:

        return self._impl.hex(*args)

    def index(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        return self._impl.index(item, start=start, endex=endex)

    def insert(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
    ) -> None:

        self._impl.insert(address, data)

    def insert_backup(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
    ) -> Tuple[Address, ImmutableMemory]:

        return self._impl.insert_backup(address, data)

    def insert_restore(
        self,
        address: Address,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.insert_restore(address, backup)

    def intervals(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Iterator[ClosedInterval]:

        yield from self._impl.intervals(start=start, endex=endex)

    def items(
        self,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Tuple[Address, Optional[Value]]]:

        yield from self._impl.items(start=start, endex=endex, pattern=pattern)

    def keys(
        self,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
    ) -> Iterator[Address]:

        yield from self._impl.keys(start=start, endex=endex)

    def peek(
        self,
        address: Address,
    ) -> Optional[Value]:

        return self._impl.peek(address)

    def poke(
        self,
        address: Address,
        item: Optional[Union[AnyBytes, Value]],
    ) -> None:

        self._impl.poke(address, item)

    def poke_backup(
        self,
        address: Address,
    ) -> Tuple[Address, Optional[Value]]:

        return self._impl.poke_backup(address)

    def poke_restore(
        self,
        address: Address,
        item: Optional[Value],
    ) -> None:

        self._impl.poke_restore(address, item)

    def pop(
        self,
        address: Optional[Address] = None,
        default: Optional[Value] = None,
    ) -> Optional[Value]:

        return self._impl.pop(address=address, default=default)

    def pop_backup(
        self,
        address: Optional[Address] = None,
    ) -> Tuple[Address, Optional[Value]]:

        return self._impl.pop_backup(address=address)

    def pop_restore(
        self,
        address: Address,
        item: Optional[Value],
    ) -> None:

        self._impl.pop_restore(address, item)

    def popitem(
        self,
    ) -> Tuple[Address, Value]:

        return self._impl.popitem()

    def popitem_backup(
        self,
    ) -> Tuple[Address, Value]:

        return self._impl.popitem_backup()

    def popitem_restore(
        self,
        address: Address,
        item: Value,
    ) -> None:

        self._impl.popitem_restore(address, item)

    def read(
        self,
        address: Address,
        size: Address,
    ) -> memoryview:

        return self._impl.read(address, size)

    def remove(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> None:

        self._impl.remove(item, start=start, endex=endex)

    def remove_backup(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> ImmutableMemory:

        return self._impl.remove_backup(item, start=start, endex=endex)

    def remove_restore(
        self,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.remove_restore(backup)

    def reserve(
        self,
        address: Address,
        size: Address,
    ) -> None:

        self._impl.reserve(address, size)

    def reserve_backup(
        self,
        address: Address,
        size: Address,
    ) -> Tuple[Address, ImmutableMemory]:

        return self._impl.reserve_backup(address, size)

    def reserve_restore(
        self,
        address: Address,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.reserve_restore(address, backup)

    def reverse(
        self,
    ) -> None:

        self._impl.reverse()

    def rfind(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        return self._impl.rfind(item, start=start, endex=endex)

    def rindex(
        self,
        item: Union[AnyBytes, Value],
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> Address:

        return self._impl.rindex(item, start=start, endex=endex)

    def rvalues(
        self,
        start: Optional[Union[Address, EllipsisType]] = None,
        endex: Optional[Address] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Optional[Value]]:

        yield from self._impl.rvalues(start=start, endex=endex, pattern=pattern)

    def setdefault(
        self,
        address: Address,
        default: Optional[Union[AnyBytes, Value]] = None,
    ) -> Optional[Value]:

        return self._impl.setdefault(address, default=default)

    def setdefault_backup(
        self,
        address: Address,
    ) -> Tuple[Address, Optional[Value]]:

        return self._impl.setdefault_backup(address)

    def setdefault_restore(
        self,
        address: Address,
        item: Optional[Value],
    ) -> None:

        self._impl.setdefault_restore(address, item)

    def shift(
        self,
        offset: Address,
    ) -> None:

        self._impl.shift(offset)

    def shift_backup(
        self,
        offset: Address,
    ) -> Tuple[Address, ImmutableMemory]:

        return self._impl.shift_backup(offset)

    def shift_restore(
        self,
        offset: Address,
        backup: ImmutableMemory,
    ) -> None:

        self._impl.shift_restore(offset, backup)

    @ImmutableMemory.span.getter
    def span(
        self,
    ) -> ClosedInterval:

        return self._impl.span

    @ImmutableMemory.start.getter
    def start(
        self,
    ) -> Address:

        return self._impl.start

    def to_blocks(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> BlockList:

        return self._impl.to_blocks(start=start, endex=endex)

    def to_bytes(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> bytes:

        return self._impl.to_bytes(start=start, endex=endex)

    def update(
        self,
        data: Union[AddressValueMapping,
                    Iterable[Tuple[Address, Optional[Value]]],
                    Mapping[Address, Optional[Union[Value, AnyBytes]]],
                    ImmutableMemory],
        clear: bool = False,
        **kwargs: Any,  # string keys cannot become addresses
    ) -> None:

        self._impl.update(data, clear=clear, **kwargs)

    def update_backup(
        self,
        data: Union[AddressValueMapping,
                    Iterable[Tuple[Address, Optional[Value]]],
                    Mapping[Address, Optional[Union[Value, AnyBytes]]],
                    ImmutableMemory],
        clear: bool = False,
        **kwargs: Any,  # string keys cannot become addresses
    ) -> Union[AddressValueMapping, List[ImmutableMemory]]:

        return self._impl.update_backup(data, clear=clear, **kwargs)

    def update_restore(
        self,
        backups: Union[AddressValueMapping, List[ImmutableMemory]],
    ) -> None:

        self._impl.update_restore(backups)

    def validate(
        self,
    ) -> None:

        self._impl.validate()

    def values(
        self,
        start: Optional[Address] = None,
        endex: Optional[Union[Address, EllipsisType]] = None,
        pattern: Optional[Union[AnyBytes, Value]] = None,
    ) -> Iterator[Optional[Value]]:

        yield from self._impl.values(start=start, endex=endex, pattern=pattern)

    def view(
        self,
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ) -> memoryview:

        return self._impl.view(start=start, endex=endex)

    def write(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory, 'Memory'],
        clear: bool = False,
    ) -> None:

        self._impl.write(address, data, clear=clear)

    def write_backup(
        self,
        address: Address,
        data: Union[AnyBytes, Value, ImmutableMemory],
        clear: bool = False,
    ) -> List[ImmutableMemory]:

        return self._impl.write_backup(address, data, clear=clear)

    def write_restore(
        self,
        backups: Sequence[ImmutableMemory],
    ) -> None:

        self._impl.write_restore(backups)


# noinspection PyPep8Naming
class bytesparse(Memory, MutableBytesparse):
    __doc__ = MutableBytesparse.__doc__

    _Memory = _cast(Type[MutableBytesparse], _CythonBytesparse)

    def __init__(
        self,
        *args: Any,  # see docstring
        start: Optional[Address] = None,
        endex: Optional[Address] = None,
    ):

        super().__init__()  # dummy
        impl = self._Memory(*args, start=start, endex=endex)
        self._impl = _cast(MutableBytesparse, impl)

    def _rectify_address(
        self,
        address: Address,
    ) -> Address:

        return self._impl._rectify_address(address)

    def _rectify_span(
        self,
        start: Optional[Address],
        endex: Optional[Address],
    ) -> OpenInterval:

        return self._impl._rectify_span(start, endex)
