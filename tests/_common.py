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
import sys
from itertools import islice
from typing import Any
from typing import List
from typing import Optional

import pytest
from bytesparse.base import STR_MAX_CONTENT_SIZE
from bytesparse.base import Address
from bytesparse.base import BlockList
from bytesparse.base import OpenInterval
from bytesparse.base import Value

MAX_START: Address = 22
MAX_SIZE: Address = 26
MAX_TIMES: int = 5
BITMASK_SIZE: int = 16


def create_template_blocks() -> BlockList:
    return [
        [2, bytearray(b'234')],
        [8, bytearray(b'89A')],
        [12, bytearray(b'C')],
        [16, bytearray(b'EF')],
        [21, bytearray(b'I')],
    ]


def create_hello_world_blocks() -> BlockList:
    return [
        [2, bytearray(b'Hello')],
        [10, bytearray(b'World!')],
    ]


def blocks_to_values(
    blocks: BlockList,
    extra: Address = 0,
) -> List[Optional[Value]]:

    values = []

    for block_start, block_data in blocks:
        values.extend(None for _ in range(block_start - len(values)))
        values.extend(block_data)

    values.extend(None for _ in range(extra))
    return values


def values_to_blocks(
    values: List[Optional[int]],
    offset: Address = 0,
) -> BlockList:

    blocks = []
    cell_count = len(values)
    block_start = None
    block_data = None
    i = 0

    while i < cell_count:
        while i < cell_count and values[i] is None:
            i += 1

        if i != block_start:
            block_start = i
            block_data = bytearray()

        while i < cell_count and values[i] is not None:
            block_data.append(values[i])
            i += 1

        if block_data:
            blocks.append([block_start + offset, block_data])

    return blocks


def values_to_equal_span(
    values: List[Optional[int]],
    address: Address,
) -> OpenInterval:

    size = len(values)
    value = values[address]
    start = endex = address

    while 0 <= start and values[start] == value:
        start -= 1
    start += 1

    while endex < size and values[endex] == value:
        endex += 1

    if value is None:
        if start <= 0:
            start = None
        if endex >= size:
            endex = None
    return start, endex


def values_to_intervals(
    values: List[Optional[int]],
    start: Optional[Address] = None,
    endex: Optional[Address] = None,
) -> List[OpenInterval]:

    intervals = []
    size = len(values)

    if start is None:
        for offset in range(size):
            if values[offset] is not None:
                start = offset
                break
        else:
            start = size
    offset = start

    if endex is not None:
        size = endex

    while offset < size:
        while offset < size and values[offset] is None:
            offset += 1
        if offset < size:
            start = offset
            while offset < size and values[offset] is not None:
                offset += 1
            if start < offset:
                intervals.append((start, offset))

    return intervals


def values_to_gaps(
    values: List[Optional[int]],
    start: Optional[Address] = None,
    endex: Optional[Address] = None,
    bound: bool = False,
) -> List[OpenInterval]:

    gaps = []

    if any(x is not None for x in values):
        size = len(values)

        if start is None:
            for offset in range(size):
                if values[offset] is not None:
                    if not bound:
                        gaps.append((None, offset))
                    start = offset
                    break
            else:
                start = size
        offset = start

        if endex is not None:
            size = endex

        while offset < size:
            while offset < size and values[offset] is None:
                offset += 1
            if offset < size:
                if start < offset:
                    gaps.append((start, offset))

                while offset < size and values[offset] is not None:
                    offset += 1
                start = offset

        if endex is None and not bound:
            gaps.append((start, None))
        elif start is not None and endex is not None and start < endex:
            gaps.append((start, endex))

    elif not bound:
        gaps.append((None, None))

    return gaps


def create_bitmask_values(
    index: int,
    size: int = BITMASK_SIZE,
) -> List[Optional[Value]]:

    values: List[Optional[Value]] = [None] * size
    for shift in range(size):
        if index & (1 << shift):
            values[shift] = shift
    return values


def test_create_bitmask_values():
    assert create_bitmask_values(0, 4) == [None, None, None, None]
    assert create_bitmask_values(1, 4) == [0, None, None, None]
    assert create_bitmask_values(2, 4) == [None, 1, None, None]
    assert create_bitmask_values(4, 4) == [None, None, 2, None]
    assert create_bitmask_values(8, 4) == [None, None, None, 3]
    assert create_bitmask_values(15, 4) == [0, 1, 2, 3]


class BaseMemorySuite:

    Memory: Any = None  # replace by subclassing 'Memory'
    ADDR_NEG: bool = True

    def test___init___doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert memory.to_blocks() == []

        memory = Memory.from_bytes(b'Hello, World!', offset=5)
        assert memory.to_blocks() == [[5, b'Hello, World!']]

    def test___init___bounds(self):
        Memory = self.Memory

        Memory(start=None, endex=None)
        Memory(start=0, endex=None)
        Memory(start=None, endex=0)

        Memory(start=0, endex=0)
        Memory(start=0, endex=1)
        if self.ADDR_NEG:
            Memory(start=-1, endex=0)

        Memory(start=1, endex=0)
        Memory(start=2, endex=0)
        if self.ADDR_NEG:
            Memory(start=0, endex=-1)
            Memory(start=0, endex=-2)

    def test___init___bounds_invalid(self):
        Memory = self.Memory
        match = r'invalid bounds'

        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[1, b'1'], [0, b'0']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[2, b'2'], [0, b'0']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[3, b'345'], [0, b'012']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[7, b'789'], [0, b'012']])
        if self.ADDR_NEG:
            with pytest.raises(ValueError, match=match):
                Memory.from_blocks([[0, b'0'], [-1, b'1']])
            with pytest.raises(ValueError, match=match):
                Memory.from_blocks([[0, b'0'], [-2, b'2']])

    def test___init___offset_template(self):
        Memory = self.Memory
        for offset in range(-MAX_SIZE if self.ADDR_NEG else 0, MAX_SIZE):
            blocks_ref = create_template_blocks()
            for block in blocks_ref:
                block[0] += offset

            for copy in (False, True):
                memory = Memory.from_blocks(create_template_blocks(), offset=offset, copy=copy)
                blocks_out = memory.to_blocks()
                assert blocks_out == blocks_ref

    def test___init___null(self):
        Memory = self.Memory
        Memory.from_bytes(b'')

        match = r'invalid block data size'
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[0, b'0'], [5, b''], [9, b'9']])

    def test___init___interleaving(self):
        Memory = self.Memory
        match = r'invalid block interleaving'
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[0, b'0'], [1, b'1'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[1, b'1'], [2, b'2'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[0, b'012'], [3, b'345'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[1, b'1'], [0, b'0'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[2, b'2'], [0, b'0'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[3, b'345'], [0, b'012'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory.from_blocks([[7, b'789'], [0, b'012'], [15, b'F']])
        if self.ADDR_NEG:
            with pytest.raises(ValueError, match=match):
                Memory.from_blocks([[0, b'0'], [-1, b'1'], [15, b'F']])
            with pytest.raises(ValueError, match=match):
                Memory.from_blocks([[0, b'0'], [-2, b'2'], [15, b'F']])

    def test___init___offset(self):
        Memory = self.Memory
        data = b'5'
        blocks = [[0, b'0'], [5, data], [9, b'9']]
        offset = 123

        memory = Memory.from_bytes(data, offset=offset)
        sm = memory.to_blocks()[0][0]
        assert sm == offset

        memory = Memory.from_blocks(blocks, offset=offset)
        for (sm, _), (sb, _) in zip(memory.to_blocks(), blocks):
            assert sm == sb + offset

        memory = Memory.from_blocks(blocks)
        memory2 = Memory.from_memory(memory, offset=offset)
        for (sm1, _), (sm2, _) in zip(memory.to_blocks(), memory2.to_blocks()):
            assert sm2 == sm1 + offset

    def test_from_blocks_doctest(self):
        Memory = self.Memory
        blocks = [[1, b'ABC'], [5, b'xyz']]

        memory = Memory.from_blocks(blocks)
        blocks_out = memory.to_blocks()
        blocks_ref = [[1, b'ABC'], [5, b'xyz']]
        assert blocks_out == blocks_ref

        memory = Memory.from_blocks(blocks, offset=3)
        blocks_out = memory.to_blocks()
        blocks_ref = [[4, b'ABC'], [8, b'xyz']]
        assert blocks_out == blocks_ref

    def test_to_blocks_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])

        ans_out = memory.to_blocks()
        ans_ref = [[1, b'AB'], [5, b'x'], [7, b'123']]
        assert ans_out == ans_ref

        ans_out = memory.to_blocks(2, 9)
        ans_ref = [[2, b'B'], [5, b'x'], [7, b'12']]
        assert ans_out == ans_ref

        ans_out = memory.to_blocks(3, 5)
        ans_ref = []
        assert ans_out == ans_ref

    def test_from_bytes_doctest(self):
        Memory = self.Memory

        memory = Memory.from_bytes(b'')
        assert memory.to_blocks() == []

        memory = Memory.from_bytes(b'ABCxyz', 2)
        assert memory.to_blocks() == [[2, b'ABCxyz']]

    def test_to_bytes_doctest(self):
        Memory = self.Memory

        memory = Memory.from_bytes(b'')
        assert memory.to_bytes() == b''

        memory = Memory.from_bytes(b'ABCxyz', 2)
        assert memory.to_bytes() == b'ABCxyz'
        assert memory.to_bytes(start=4) == b'Cxyz'
        assert memory.to_bytes(endex=6) == b'ABCx'
        assert memory.to_bytes(4, 6) == b'Cx'

    def test_from_memory_doctest(self):
        Memory = self.Memory

        memory1 = Memory.from_bytes(b'ABC', 5)
        memory2 = Memory.from_memory(memory1)
        assert memory2.to_blocks() == [[5, b'ABC']]
        assert (memory1 == memory2) is True
        assert (memory1 is memory2) is False

        memory1 = Memory.from_bytes(b'ABC', 10)
        memory2 = Memory.from_memory(memory1, -3)
        assert memory2.to_blocks() == [[7, b'ABC']]
        assert (memory1 == memory2) is False

    def test_fromhex_doctest(self):
        Memory = self.Memory

        memory = Memory.fromhex('')
        assert bytes(memory) == b''

        memory = Memory.fromhex('48656C6C6F2C20576F726C6421')
        assert bytes(memory) == b'Hello, World!'

    def test_fromhex_validate(self):
        Memory = self.Memory

        memory = Memory.fromhex('')
        memory.validate()
        assert bytes(memory) == b''

        memory = Memory.fromhex('48656C6C6F2C20576F726C6421')
        memory.validate()
        assert bytes(memory) == b'Hello, World!'

    def test_hex_doctest(self):
        Memory = self.Memory

        assert Memory().hex() == ''

        memory = Memory.from_bytes(b'Hello, World!')
        assert memory.hex() == '48656c6c6f2c20576f726c6421'

        if sys.version_info >= (3, 8):  # pragma: no cover
            assert memory.hex('.') == '48.65.6c.6c.6f.2c.20.57.6f.72.6c.64.21'
            assert memory.hex('.', 4) == '48.656c6c6f.2c20576f.726c6421'

    def test_hex_multi(self):
        Memory = self.Memory
        memory = Memory.from_blocks(create_template_blocks())

        with pytest.raises(ValueError, match='non-contiguous data within range'):
            memory.hex()

    def test___repr__(self):
        Memory = self.Memory
        start, endex = 0, 0
        memory = Memory()
        repr_out = repr(memory)
        repr_ref = f'<{memory.__class__.__name__}[0x{start}:0x{endex}]@0x{id(memory):X}>'
        assert repr_out == repr_ref

        start, endex = 1, 9
        memory = Memory(start=start, endex=endex)
        repr_out = repr(memory)
        repr_ref = f'<{memory.__class__.__name__}[0x{start}:0x{endex}]@0x{id(memory):X}>'
        assert repr_out == repr_ref

        start, endex = 3, 6
        memory = Memory.from_bytes(b'abc', offset=3)
        repr_out = repr(memory)
        repr_ref = f'<{memory.__class__.__name__}[0x{start}:0x{endex}]@0x{id(memory):X}>'
        assert repr_out == repr_ref

    def test___str___doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABC'], [7, b'xyz']])
        assert str(memory) == "<[[1, b'ABC'], [7, b'xyz']]>"

    def test___str__(self):
        Memory = self.Memory
        memory = Memory()
        str_out = str(memory)
        str_ref = '<[]>'
        assert str_out == str_ref

        memory = Memory(start=3, endex=9)
        str_out = str(memory)
        str_ref = '<3, [], 9>'
        assert str_out == str_ref

        data = b'abc'
        memory = Memory.from_bytes(data, offset=3)
        str_out = str(memory)
        str_ref = "<[[3, b'abc']]>"
        assert str_out == str_ref

        data = b'abc' * STR_MAX_CONTENT_SIZE
        memory = Memory.from_bytes(data, offset=3)
        str_out = str(memory)
        str_ref = repr(memory)
        assert str_out == str_ref

    def test___bool___doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert bool(memory) is False

        memory = Memory.from_bytes(b'Hello, World!', 5)
        assert bool(memory) is True

    def test___bool__(self):
        Memory = self.Memory
        assert Memory.from_memory(Memory.from_bytes(b'\0'))
        assert Memory.from_bytes(b'\0')
        assert Memory.from_blocks([[0, b'\0']])

        assert not Memory()
        assert not Memory.from_memory(Memory())
        assert not Memory.from_bytes(b'')
        assert not Memory.from_blocks([])

    def test___eq___doctest(self):
        Memory = self.Memory

        data = b'Hello, World!'
        memory = Memory.from_bytes(data)
        assert (memory == data) is True
        memory.shift(1)
        assert (memory == data) is True

        data = b'Hello, World!'
        memory = Memory.from_bytes(data)
        assert (memory == list(data)) is True
        memory.shift(1)
        assert (memory == list(data)) is True

    def test___eq___empty(self):
        Memory = self.Memory
        memory = Memory()

        assert memory == bytes()
        assert memory == bytearray()
        assert memory == ()
        assert memory == []
        assert memory == iter(())
        assert memory == Memory()

        assert memory != bytes(1)
        assert memory != bytearray(1)
        assert memory != (0,)
        assert memory != [0]
        assert memory != iter((0,))
        assert memory != Memory.from_bytes(bytes(1))

    def test___eq___memory(self):
        Memory = self.Memory
        memory1 = Memory.from_blocks(create_template_blocks())
        memory2 = Memory.from_blocks(create_template_blocks())
        assert memory1 == memory2

        memory1.append(0)
        assert memory1 != memory2

        memory1.pop()
        memory1.shift(1)
        assert memory1 != memory2

    def test___eq___multi_bytes(self):
        Memory = self.Memory
        memory1 = Memory.from_blocks(create_template_blocks())
        assert memory1 != b'abc'

    def test___eq___bytelike(self):
        Memory = self.Memory
        data = bytes(range(256))
        memory = Memory.from_bytes(data, offset=256)
        assert memory == data
        assert memory != data + b'\0'

        memory.shift(1)
        assert memory == data
        assert memory != data + b'\0'

    def test___eq___generator(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(create_template_blocks())
        values = blocks_to_values(blocks)[memory.start:memory.endex]

        assert memory == iter(values)
        assert memory != reversed(values)
        assert memory != iter(values[:-1])
        assert memory != iter(values + [0])

    def test___eq___bitmask(self):
        Memory = self.Memory
        bitmask_size = BITMASK_SIZE
        index_base = (1 << (bitmask_size - 1)) | 1
        for bitmask_index in range(1, bitmask_size - 1):
            index = index_base | bitmask_index
            values = create_bitmask_values(index, bitmask_size)
            blocks = values_to_blocks(values)
            memory = Memory.from_blocks(blocks)
            assert memory == values

    def test___iter___doctest(self):
        pass  # no doctest

    def test___iter___empty_bruteforce(self):
        # self.test_values_empty_bruteforce()
        Memory = self.Memory
        assert all(x == y for x, y in zip(Memory(), []))

    def test___iter___template(self):
        # self.test_values_template()
        Memory = self.Memory
        blocks = create_template_blocks()
        start = blocks[0][0]
        endex = blocks[-1][0] + len(blocks[-1][1])
        values = blocks_to_values(blocks)[start:endex]
        memory = Memory.from_blocks(blocks)
        assert all(x == y for x, y in zip(memory, values))

    def test___reversed___doctest(self):
        pass  # no doctest

    def test___reversed___empty_bruteforce(self):
        # self.test_rvalues_empty_bruteforce()
        Memory = self.Memory
        assert all(x == y for x, y in zip(reversed(Memory()), []))

    def test___reversed___template(self):
        # self.test_rvalues_template()
        Memory = self.Memory
        blocks = create_template_blocks()
        start = blocks[0][0]
        endex = blocks[-1][0] + len(blocks[-1][1])
        values = blocks_to_values(blocks)[start:endex]
        memory = Memory.from_blocks(blocks)
        assert all(x == y for x, y in zip(reversed(memory), reversed(values)))

    def test_reverse_empty(self):
        Memory = self.Memory

        memory = Memory()
        memory.reverse()
        assert not memory

        memory = Memory(start=2, endex=10)
        memory.reverse()
        assert not memory

    def test_reverse_doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        memory.reverse()
        assert memory.to_blocks() == [[1, b'zyx'], [5, b'$'], [7, b'DCBA']]

        memory = Memory.from_bytes(b'ABC', 4, start=2, endex=10)
        memory.reverse()
        assert memory.to_blocks() == [[5, b'CBA']]

    def test___add___doctest(self):
        pass  # no doctest

    def test___add___template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        blocks2 = create_hello_world_blocks()

        block = blocks1[-1]
        offset = block[0] + len(block[1])
        for block in blocks2:
            block[0] += offset

        memory1 = Memory.from_blocks(blocks1)
        memory2 = Memory.from_blocks(blocks2)
        memory3 = memory1 + memory2
        memory3.validate()
        blocks_out = memory3.to_blocks()

        values = blocks_to_values(blocks1)
        values += blocks_to_values(blocks2)
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref

    def test___iadd___doctest(self):
        pass  # no doctest

    def test___iadd___template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        blocks2 = create_hello_world_blocks()

        block = blocks1[-1]
        offset = block[0] + len(block[1])
        for block in blocks2:
            block[0] += offset

        memory1 = Memory.from_blocks(blocks1)
        memory2 = Memory.from_blocks(blocks2)
        memory1 += memory2
        memory1.validate()
        blocks_out = memory1.to_blocks()

        values = blocks_to_values(blocks1)
        values += blocks_to_values(blocks2)
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref

    def test___mul___doctest(self):
        pass  # no doctest

    def test___mul___template(self):
        Memory = self.Memory
        for times in range(-1, MAX_TIMES):
            blocks = create_template_blocks()

            memory1 = Memory.from_blocks(blocks)
            memory2 = memory1 * times
            memory2.validate()
            blocks_out = memory2.to_blocks()

            values = blocks_to_values(blocks)
            offset = blocks[0][0]
            values = ([None] * offset) + (values[offset:] * times)
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

    def test___imul___doctest(self):
        pass  # no doctest

    def test___imul___template(self):
        Memory = self.Memory
        for times in range(-1, MAX_TIMES):
            blocks = create_template_blocks()

            memory = Memory.from_blocks(blocks)
            memory *= times
            memory.validate()
            blocks_out = memory.to_blocks()

            values = blocks_to_values(blocks)
            offset = blocks[0][0]
            values = ([None] * offset) + (values[offset:] * times)
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

    def test___len___doctest(self):
        pass  # no doctest

    def test___len___empty(self):
        Memory = self.Memory
        assert len(Memory()) == 0
        assert len(Memory.from_memory(Memory())) == 0
        assert len(Memory.from_bytes(b'')) == 0
        assert len(Memory.from_blocks([])) == 0

    def test___len___template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(1, MAX_SIZE):
                endex = start + size
                data = bytes(range(size))
                memory = Memory.from_bytes(data, offset=start)

                assert memory.start == start
                assert memory.endex == endex
                assert len(memory) == size

    def test___len___bounds(self):
        Memory = self.Memory

        assert len(Memory(start=1, endex=9)) == 9 - 1
        assert len(Memory.from_memory(Memory(), start=1, endex=9)) == 9 - 1
        assert len(Memory.from_bytes(b'', start=1, endex=9)) == 9 - 1
        assert len(Memory.from_blocks([], start=1, endex=9)) == 9 - 1

        assert len(Memory(start=1)) == 0
        assert len(Memory.from_memory(Memory(), start=1)) == 0
        assert len(Memory.from_bytes(b'', start=1)) == 0
        assert len(Memory.from_blocks([], start=1)) == 0

        assert len(Memory(endex=9)) == 9
        assert len(Memory.from_memory(Memory(), endex=9)) == 9
        assert len(Memory.from_bytes(b'', endex=9)) == 9
        assert len(Memory.from_blocks([], endex=9)) == 9

    def test___len__(self):
        Memory = self.Memory

        memory = Memory.from_blocks(create_hello_world_blocks())
        assert len(memory) == memory.endex - memory.start
        assert len(memory) == (16 - 2)

        memory = Memory.from_blocks(create_template_blocks())
        assert len(memory) == memory.endex - memory.start
        assert len(memory) == (22 - 2)

    def test_ofind_doctest(self):
        pass  # no doctest

    def test_ofind(self):
        Memory = self.Memory

        memory = Memory.from_blocks(create_hello_world_blocks())

        assert memory.ofind(b'X') is None
        assert memory.ofind(b'W') == 10
        assert memory.ofind(b'o') == 6
        assert memory.ofind(b'l') == 4

    def test_rofind_doctest(self):
        pass  # no doctest

    def test_rofind(self):
        Memory = self.Memory

        memory = Memory.from_blocks(create_hello_world_blocks())

        assert memory.rofind(b'X') is None
        assert memory.rofind(b'W') == 10
        assert memory.rofind(b'o') == 11
        assert memory.rofind(b'l') == 13

    def test_find_doctest(self):
        pass  # no doctest

    def test_find(self):
        Memory = self.Memory

        memory = Memory.from_blocks(create_hello_world_blocks())

        assert memory.find(b'X') == -1
        assert memory.find(b'W') == 10
        assert memory.find(b'o') == 6
        assert memory.find(b'l') == 4

    def test_rfind_doctest(self):
        pass  # no doctest

    def test_rfind(self):
        Memory = self.Memory

        memory = Memory.from_blocks(create_hello_world_blocks())

        assert memory.rfind(b'X') == -1
        assert memory.rfind(b'W') == 10
        assert memory.rfind(b'o') == 11
        assert memory.rfind(b'l') == 13

    def test_index_doctest(self):
        pass  # no doctest

    def test_index(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory.from_blocks(blocks)
        values = blocks_to_values(blocks, MAX_SIZE)
        chars = (set(values) - {None}) | {b'X'[0]}
        match = r'subsection not found'

        for start in range(MAX_START):
            for endex in range(start, MAX_START):
                for c in chars:
                    expected = None
                    for i in range(start, endex):
                        if values[i] == c:
                            expected = i
                            break

                    if expected is None:
                        with pytest.raises(ValueError, match=match):
                            index = memory.index(c, start, endex)
                            assert index

                        with pytest.raises(ValueError, match=match):
                            index = memory.index(bytes([c]), start, endex)
                            assert index
                    else:
                        index = memory.index(c, start, endex)
                        assert index == expected

                        index = memory.index(bytes([c]), start, endex)
                        assert index == expected

        for c in chars:
            expected = None
            for i in range(len(values)):
                if values[i] == c:
                    expected = i
                    break

            if expected is None:
                with pytest.raises(ValueError, match=match):
                    index = memory.index(c)
                    assert index

                with pytest.raises(ValueError, match=match):
                    index = memory.index(bytes([c]))
                    assert index
            else:
                index = memory.index(c)
                assert index == expected

                index = memory.index(bytes([c]))
                assert index == expected

    def test_rindex_doctest(self):
        pass  # no doctest

    def test_rindex(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory.from_blocks(blocks)
        values = blocks_to_values(blocks, MAX_SIZE)
        chars = (set(values) - {None}) | {b'X'[0]}
        match = r'subsection not found'

        for start in range(MAX_START):
            for endex in range(start, MAX_START):
                values2 = values[start:endex]
                for c in chars:
                    expected = None
                    for i in reversed(range(start, endex)):
                        if values2[i - start] == c:
                            expected = i
                            break

                    if expected is None:
                        with pytest.raises(ValueError, match=match):
                            index = memory.rindex(c, start, endex)
                            assert index

                        with pytest.raises(ValueError, match=match):
                            index = memory.rindex(bytes([c]), start, endex)
                            assert index
                    else:
                        index = memory.rindex(c, start, endex)
                        assert index == expected

                        index = memory.rindex(bytes([c]), start, endex)
                        assert index == expected

        for c in chars:
            expected = None
            for i in reversed(range(len(values))):
                if values[i] == c:
                    expected = i
                    break

            if expected is None:
                with pytest.raises(ValueError, match=match):
                    index = memory.rindex(c)
                    assert index

                with pytest.raises(ValueError, match=match):
                    index = memory.rindex(bytes([c]))
                    assert index
            else:
                index = memory.rindex(c)
                assert index == expected

                index = memory.rindex(bytes([c]))
                assert index == expected

    def test_remove_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        memory.remove(b'BC')
        assert memory.to_blocks() == [[1, b'AD'], [4, b'$'], [6, b'xyz']]
        memory.remove(ord('$'))
        assert memory.to_blocks() == [[1, b'AD'], [5, b'xyz']]

        with pytest.raises(ValueError, match='subsection not found'):
            memory.remove(b'?')

    def test_remove(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])

        memory_backup1 = memory.__deepcopy__()
        backup1 = memory.remove_backup(b'BC')
        memory.remove(b'BC')
        memory.validate()
        assert memory.to_blocks() == [[1, b'AD'], [4, b'$'], [6, b'xyz']]

        memory_backup2 = memory.__deepcopy__()
        backup2 = memory.remove_backup(ord('$'))
        memory.remove(ord('$'))
        memory.validate()
        assert memory.to_blocks() == [[1, b'AD'], [5, b'xyz']]

        memory.remove_restore(backup2)
        assert memory == memory_backup2

        memory.remove_restore(backup1)
        assert memory == memory_backup1

    def test_remove_backup_doctest(self):
        pass  # no doctest

    def test_remove_restore_doctest(self):
        pass  # no doctest

    def test___contains___doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABC'], [5, b'123'], [9, b'xyz']])
        assert (b'23' in memory) is True
        assert (ord('y') in memory) is True
        assert (b'$' in memory) is False

    def test___contains___empty_bruteforce(self):
        Memory = self.Memory
        memory = Memory()

        checks = [i not in memory for i in range(256)]
        assert all(checks)

        checks = [bytes([i]) not in memory for i in range(256)]
        assert all(checks)

    def test___contains__(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory.from_blocks(blocks)
        values = blocks_to_values(blocks)
        chars = (set(values) - {None}) | {b'X'[0]}

        for c in chars:
            expected = c in values

            check = c in memory
            assert check == expected

            check = bytes([c]) in memory
            assert check == expected

    def test_count_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABC'], [5, b'Bat'], [9, b'tab']])
        assert memory.count(b'a') == 2

    def test_count_empty_bruteforce(self):
        Memory = self.Memory
        memory = Memory()

        checks = [not memory.count(i) for i in range(256)]
        assert all(checks)

        checks = [not memory.count(bytes([i])) for i in range(256)]
        assert all(checks)

        for start in range(8):
            for endex in range(start, 8):
                checks = [not memory.count(i, start, endex) for i in range(8)]
                assert all(checks)

                checks = [not memory.count(bytes([i]), start, endex) for i in range(8)]
                assert all(checks)

    def test_count(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory.from_blocks(blocks)
        values = blocks_to_values(blocks)
        chars = (set(values) - {None}) | {b'X'[0]}

        for start in range(MAX_START):
            for endex in range(start, MAX_START):
                for c in chars:
                    expected = values[start:endex].count(c)

                    count = memory.count(c, start, endex)
                    assert count == expected

                    count = memory.count(bytes([c]), start, endex)
                    assert count == expected

    def test___getitem___doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        assert memory[9] == 121
        assert memory[:3].to_blocks() == [[1, b'AB']]
        assert memory[3:10].to_blocks() == [[3, b'CD'], [6, b'$'], [8, b'xy']]
        assert bytes(memory[3:10:b'.']) == b'CD.$.xy'
        assert memory[memory.endex] is None
        assert bytes(memory[3:10:3]) == b'C$y'
        assert memory[3:10:2].to_blocks() == [[3, b'C'], [6, b'y']]

        with pytest.raises(ValueError, match='non-contiguous data within range'):
            bytes(memory[3:10:2])

    def test___getitem___single_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for start in range(MAX_START):
            value = memory[start]
            assert value == values[start]

    def test___getitem___contiguous(self):
        Memory = self.Memory
        blocks = [[3, b'abc']]
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for start in range(3, 6):
            for endex in range(start, 6):
                data_out = list(memory[start:endex].values())
                data_ref = values[start:endex]
                assert data_out == data_ref

    def test___getitem___contiguous_step(self):
        Memory = self.Memory
        data = bytes(range(ord('a'), ord('z') + 1))
        blocks = [[3, data]]
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        stop = 3 + len(data)
        for start in range(3, stop):
            for endex in range(start, stop):
                for step in range(1, 4):
                    data_out = list(memory[start:endex:step].values())
                    data_ref = values[start:endex:step]
                    assert data_out == data_ref

                for step in range(-1, 1):
                    blocks_out = memory[start:endex:step].to_blocks()
                    blocks_ref = []
                    assert blocks_out == blocks_ref

    def test___getitem___non_contiguous(self):
        Memory = self.Memory
        data = b'abc'
        memory = Memory.from_bytes(data, offset=5)
        dot = b'.'

        assert memory[:] == data
        assert memory[::dot] == data
        assert memory[:5] == b''
        assert memory[:5:dot] == b''
        assert memory[8:] == b''
        assert memory[8::dot] == b''
        assert memory[::dot] == data

        extracted = memory[1:9]
        assert extracted.to_blocks() == [[5, data]]
        assert extracted.span == (1, 9)

        extracted = memory[1:7]
        assert extracted.to_blocks() == [[5, data[:7 - 5]]]
        assert extracted.span == (1, 7)

        extracted = memory[7:9]
        assert extracted.to_blocks() == [[7, data[7 - 5:]]]
        assert extracted.span == (7, 9)

        memory = Memory.from_bytes(data, offset=5)
        extracted = memory[:]
        assert extracted.to_blocks() == [[5, data]]
        assert extracted.span == (5, 8)

        memory = Memory.from_bytes(data, offset=5, start=2, endex=22)
        extracted = memory[:]
        assert extracted.to_blocks() == [[5, data]]
        assert extracted.span == (2, 22)

        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        extracted = memory[:]
        assert extracted.to_blocks() == blocks
        assert extracted.span == memory.span

    def test___getitem___pattern(self):
        Memory = self.Memory
        dot = ord('.')
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                values_out = list(memory[start:endex:b'.'])

                for index in range(start):
                    values[index] = None
                for index in range(start, endex):
                    if values[index] is None:
                        values[index] = dot
                values_ref = values[start:endex]
                assert values_out == values_ref

    def test___setitem___doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
        memory[7:10] = None
        assert memory.to_blocks() == [[5, b'AB'], [10, b'yz']]
        memory[7] = b'C'
        memory[9] = b'x'
        assert (memory.to_blocks() == [[5, b'ABC'], [9, b'xyz']]) is True
        memory[6:12:3] = None
        assert memory.to_blocks() == [[5, b'A'], [7, b'C'], [10, b'yz']]
        memory[6:13:3] = b'123'
        assert memory.to_blocks() == [[5, b'A1C'], [9, b'2yz3']]

        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
        memory[0:4] = b'$'
        assert memory.to_blocks() == [[0, b'$'], [2, b'ABC'], [6, b'xyz']]
        memory[4:7] = b'45678'
        assert memory.to_blocks() == [[0, b'$'], [2, b'AB45678yz']]
        memory[6:8] = b'<>'
        assert memory.to_blocks() == [[0, b'$'], [2, b'AB45<>8yz']]

    def test___setitem___single_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            memory[start] = start
            memory.validate()
            blocks_out = memory.to_blocks()

            values[start] = start
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

    def test___setitem___replace_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                data = bytearray(range(size))
                endex = start + size

                memory[start:endex] = data
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:endex] = list(data)
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

    def test___setitem___shrink_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size + 1, MAX_SIZE):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory = Memory.from_blocks(blocks)
                    memory[start:endex] = data
                    memory.validate()
                    blocks_out = memory.to_blocks()

                    values_ref = values[:]
                    values_ref[start:endex] = list(data)
                    blocks_ref = values_to_blocks(values_ref)

                    assert blocks_out == blocks_ref

    def test___setitem___shrink_unbounded_start_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size + 1, MAX_SIZE):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory = Memory.from_blocks(blocks)
                    memory[:endex] = data
                    memory.validate()
                    blocks_out = memory.to_blocks()

                    values_ref = values[:]
                    values_ref[blocks[0][0]:endex] = list(data)
                    blocks_ref = values_to_blocks(values_ref)

                    assert blocks_out == blocks_ref

    def test___setitem___shrink_unbounded_end_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size + 1, MAX_SIZE):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    data = bytearray(range(source_size))

                    memory = Memory.from_blocks(blocks)
                    memory[start:] = data
                    memory.validate()
                    blocks_out = memory.to_blocks()

                    values_ref = values[:]
                    values_ref[start:] = list(data)
                    blocks_ref = values_to_blocks(values_ref)

                    assert blocks_out == blocks_ref

    def test___setitem___enlarge_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory.from_blocks(blocks)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory[start:endex] = data
                    memory.validate()
                    blocks_out = memory.to_blocks()

                    values[start:endex] = list(data)
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref

    def test___setitem___none_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                memory[start:endex] = None
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:endex] = [None] * (endex - start)
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref

    def test___setitem___none_step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(1, MAX_TIMES):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory.from_blocks(blocks)
                    endex = start + size

                    memory[start:endex:step] = None
                    memory.validate()
                    blocks_out = memory.to_blocks()

                    values[start:endex:step] = [None] * ((size + step - 1) // step)
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref

    def test___setitem___value_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                memory[start:endex] = start
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:endex] = [start] * (endex - start)
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref

    def test___setitem___misstep_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(-1, 1):
                    blocks = create_template_blocks()
                    memory = Memory.from_blocks(blocks)
                    endex = start + size

                    memory[start:endex:step] = b''
                    memory.validate()
                    blocks_out = memory.to_blocks()
                    blocks_ref = blocks

                    assert blocks_out == blocks_ref

    def test___setitem___step_template(self):
        Memory = self.Memory
        match = r'attempt to assign'

        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(MAX_SIZE):
                    for step in range(2, MAX_TIMES):
                        blocks = create_template_blocks()
                        values = blocks_to_values(blocks, MAX_SIZE)
                        memory = Memory.from_blocks(blocks)
                        endex = start + target_size
                        data = bytes(source_size)

                        values_ref = values[:]
                        try:
                            values_ref[start:endex:step] = data
                        except ValueError as e:
                            assert str(e).startswith(match)

                            with pytest.raises(ValueError, match=match):
                                memory[start:endex:step] = data
                        else:
                            blocks_ref = values_to_blocks(values_ref)

                            memory[start:endex:step] = data
                            memory.validate()
                            blocks_out = memory.to_blocks()

                            assert blocks_out == blocks_ref

    def test___delitem___doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        del memory[4:9]
        assert memory.to_blocks() == [[1, b'ABCyz']]

        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        del memory[9]
        assert memory.to_blocks() == [[1, b'ABCD'], [6, b'$'], [8, b'xz']]
        del memory[3]
        assert memory.to_blocks() == [[1, b'ABD'], [5, b'$'], [7, b'xz']]
        del memory[2:10:3]
        assert memory.to_blocks() == [[1, b'AD'], [5, b'x']]

    def test___delitem___empty(self):
        Memory = self.Memory
        memory = Memory()
        del memory[:]
        memory.validate()

    def test___delitem___single_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            del memory[start]
            memory.validate()
            blocks_out = memory.to_blocks()

            del values[start]
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref

    def test___delitem___step_negative_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for step in range(-1, 1):
                blocks = create_template_blocks()
                memory = Memory.from_blocks(blocks)

                del memory[start::step]
                memory.validate()
                blocks_out = memory.to_blocks()
                assert blocks_out == blocks

    def test___delitem___template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                del memory[start:endex]
                memory.validate()
                blocks_out = memory.to_blocks()

                del values[start:endex]
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref

    def test___delitem___step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(1, MAX_TIMES):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory.from_blocks(blocks)
                    endex = start + size

                    del memory[start:endex:step]
                    memory.validate()
                    blocks_out = memory.to_blocks()

                    del values[start:endex:step]
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref

    def test___delitem___unbounded_start_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                del memory[:endex]
                memory.validate()
                blocks_out = memory.to_blocks()

                del values[blocks[0][0]:endex]
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref

    def test___delitem___unbounded_endex_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)

                del memory[start:]
                memory.validate()
                blocks_out = memory.to_blocks()

                del values[start:]
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref

    def test_append_doctest(self):
        Memory = self.Memory

        memory = Memory()
        memory.append(b'$')
        assert memory.to_blocks() == [[0, b'$']]

        memory = Memory()
        memory.append(3)
        assert memory.to_blocks() == [[0, b'\x03']]

    def test_append_empty_int(self):
        Memory = self.Memory
        memory = Memory()
        memory.append(ord('X'))
        memory.validate()
        values = [ord('X')]

        blocks_out = memory.to_blocks()
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref

    def test_append_empty_byte(self):
        Memory = self.Memory
        memory = Memory()
        memory.append(b'X')
        memory.validate()
        values = [ord('X')]
        blocks_out = memory.to_blocks()
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref

    def test_append_empty_multi(self):
        Memory = self.Memory
        memory = Memory()
        with pytest.raises(ValueError, match='expecting single item'):
            memory.append(b'XY')

    def test_append(self):
        Memory = self.Memory
        memory = Memory.from_blocks(create_template_blocks())
        blocks_ref = create_template_blocks()
        blocks_ref[-1][1].append(ord('X'))

        memory_backup = memory.__deepcopy__()
        backup = memory.append_backup()
        assert backup is None

        memory.append(ord('X'))
        memory.validate()
        blocks_out = memory.to_blocks()
        assert blocks_out == blocks_ref

        memory.append_restore()
        assert memory == memory_backup

    def test_append_backup_doctest(self):
        pass  # no doctest

    def test_append_restore_doctest(self):
        pass  # no doctest

    def test_extend_doctest(self):
        pass  # no doctest

    def test_extend_empty(self):
        Memory = self.Memory
        memory = Memory()
        blocks_ref = []
        memory.extend(b'')
        memory.validate()
        blocks_out = memory.to_blocks()
        assert blocks_out == blocks_ref

    def test_extend_invalid(self):
        Memory = self.Memory
        match = r'negative extension offset'
        memory = Memory()
        with pytest.raises(ValueError, match=match):
            memory.extend([], offset=-1)

    def test_extend_bytes(self):
        Memory = self.Memory
        for size in range(MAX_SIZE):
            blocks = create_template_blocks()
            memory = Memory.from_blocks(blocks)
            values = blocks_to_values(blocks)
            data = bytes(range(size))

            memory.extend(data)
            memory.validate()
            blocks_out = memory.to_blocks()

            values.extend(data)
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref

    def test_extend_template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        blocks2 = create_hello_world_blocks()

        block = blocks1[-1]
        offset = block[0] + len(block[1])
        for block in blocks2:
            block[0] += offset

        memory1 = Memory.from_blocks(blocks1)
        memory2 = Memory.from_blocks(blocks2)
        memory1.extend(memory2)
        memory1.validate()
        blocks_out = memory1.to_blocks()

        values = blocks_to_values(blocks1)
        values.extend(blocks_to_values(blocks2))
        blocks_ref = values_to_blocks(values)

        assert blocks_out == blocks_ref

    def test_extend_bytes_bounded(self):
        Memory = self.Memory
        for size in range(MAX_SIZE):
            for offset in range(MAX_SIZE):
                blocks = create_template_blocks()
                memory = Memory.from_blocks(blocks, endex=MAX_SIZE)
                values = blocks_to_values(blocks)
                data = bytes(range(size))

                memory_backup = memory.__deepcopy__()
                backup_content_endex = memory.extend_backup(offset)

                memory.extend(data, offset)
                memory.validate()
                blocks_out = memory.to_blocks()
                values.extend(None for _ in range(offset))
                values.extend(data)
                del values[MAX_SIZE:]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.extend_restore(backup_content_endex)
                assert memory == memory_backup

    def test_extend_backup_doctest(self):
        pass  # no doctest

    def test_extend_backup_negative(self):
        Memory = self.Memory
        memory = Memory()

        with pytest.raises(ValueError, match='negative extension offset'):
            memory.extend_backup(-1)

    def test_extend_restore_doctest(self):
        pass  # no doctest

    def test_pop_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        assert memory.pop() == 122
        assert memory.pop(3) == 67
        assert memory.pop(6, 63) == 63

    def test_pop_empty(self):
        Memory = self.Memory
        memory = Memory()
        value_out = memory.pop()
        assert value_out is None

    def test_pop_simple(self):
        Memory = self.Memory
        blocks = [[0, b'\x00\x01\x02']]
        values = blocks_to_values(blocks)
        memory = Memory.from_blocks(blocks)

        memory_backup = memory.__deepcopy__()
        backup_address, backup_value = memory.pop_backup()
        assert backup_address == 2
        assert backup_value == 2

        value_out = memory.pop()
        memory.validate()
        values.pop()
        assert value_out == 2

        blocks_out = memory.to_blocks()
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref

        memory.pop_restore(backup_address, backup_value)
        memory.validate()
        assert memory == memory_backup

    def test_pop_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks)
        memory = Memory.from_blocks(blocks)

        memory_backup = memory.__deepcopy__()
        backup_address, backup_value = memory.pop_backup()
        assert backup_address == memory_backup.content_endin
        assert backup_value == values[backup_address]

        value_out = memory.pop()
        memory.validate()
        value_ref = values.pop()
        assert value_out == value_ref

        blocks_out = memory.to_blocks()
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref

        memory.pop_restore(backup_address, backup_value)
        memory.validate()
        assert memory == memory_backup

    def test_pop_template_bruteforce(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            memory_backup = memory.__deepcopy__()
            backup_address, backup_value = memory.pop_backup(start)
            assert backup_address == start
            assert backup_value == values[start]

            value_out = memory.pop(start)
            memory.validate()
            value_ref = values.pop(start)
            assert value_out == value_ref

            blocks_out = memory.to_blocks()
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

            memory.pop_restore(backup_address, backup_value)
            memory.validate()
            assert memory == memory_backup

    def test_pop_backup_doctest(self):
        pass  # no doctest

    def test_pop_restore_doctest(self):
        pass  # no doctest

    def test_popitem_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'A'], [9, b'yz']])
        assert memory.popitem() == (10, 122)
        assert memory.popitem() == (9, 121)
        assert memory.popitem() == (1, 65)
        with pytest.raises(KeyError, match='empty'):
            memory.popitem()

    def test_popitem(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        while values:
            value = values.pop()
            if value is not None:
                offset = len(values)
                memory_backup = memory.__deepcopy__()
                backup_address, backup_value = memory.popitem_backup()
                assert backup_address == offset
                assert backup_value == value

                address_out, value_out = memory.popitem()
                memory.validate()
                value_ref = value
                assert address_out == offset
                assert value_out == value_ref

                blocks_out = memory.to_blocks()
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.popitem_restore(backup_address, backup_value)
                memory.validate()
                assert memory == memory_backup

                memory.popitem()
                memory.validate()

    def test_popitem_empty(self):
        Memory = self.Memory
        memory = Memory()
        with pytest.raises(KeyError, match='empty'):
            memory.popitem()
        with pytest.raises(KeyError, match='empty'):
            memory.popitem_backup()

    def test_popitem_backup_doctest(self):
        pass  # no doctest

    def test_popitem_restore_doctest(self):
        pass  # no doctest

    def test___bytes___doctest(self):
        pass  # no doctest

    def test___bytes__(self):
        Memory = self.Memory
        memory = Memory()
        data = memory.__bytes__()
        assert data == b''

        memory = Memory.from_bytes(b'xyz', offset=5)
        data = memory.__bytes__()
        assert data == b'xyz'

        blocks = [[5, b'xyz']]
        memory = Memory.from_blocks(blocks, copy=False)
        data = memory.__bytes__()
        assert data == blocks[0][1]

    def test___bytes___invalid(self):
        Memory = self.Memory
        match = r'non-contiguous data within range'

        memory = Memory(start=1, endex=9)
        with pytest.raises(ValueError, match=match):
            bytes(memory)

        memory = Memory.from_bytes(b'xyz', offset=5, start=1)
        with pytest.raises(ValueError, match=match):
            bytes(memory)

        memory = Memory.from_bytes(b'xyz', offset=5, endex=9)
        with pytest.raises(ValueError, match=match):
            bytes(memory)

        memory = Memory.from_blocks(create_template_blocks())
        with pytest.raises(ValueError, match=match):
            bytes(memory)

    def test___copy___doctest(self):
        pass  # no doctest

    def test___copy___empty(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = memory1.__copy__()
        memory2.validate()
        assert memory1.span == memory2.span
        assert memory1.trim_span == memory2.trim_span
        assert memory1.content_span == memory2.content_span
        assert memory1.content_parts == memory2.content_parts
        assert all(b1 == b2 for b1, b2 in zip(memory1.blocks(), memory2.blocks()))

    def test___copy___template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory1 = Memory.from_blocks(blocks, copy=False)
        memory2 = memory1.__copy__()
        memory2.validate()
        assert memory1.span == memory2.span
        assert memory1.trim_span == memory2.trim_span
        assert memory1.content_span == memory2.content_span
        assert memory1.content_parts == memory2.content_parts
        assert all(b1 == b2 for b1, b2 in zip(memory1.blocks(), memory2.blocks()))

    def test_copy_empty(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = memory1.copy()
        memory2.validate()
        assert memory1.span == memory2.span
        assert memory1.trim_span == memory2.trim_span
        assert memory1.content_span == memory2.content_span
        assert memory1.content_parts == memory2.content_parts
        assert all(b1 == b2 for b1, b2 in zip(memory1.blocks(), memory2.blocks()))

    def test_copy_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory1 = Memory.from_blocks(blocks, copy=False)
        memory2 = memory1.copy()
        memory2.validate()
        assert memory1.span == memory2.span
        assert memory1.trim_span == memory2.trim_span
        assert memory1.content_span == memory2.content_span
        assert memory1.content_parts == memory2.content_parts
        assert all(b1 == b2 for b1, b2 in zip(memory1.blocks(), memory2.blocks()))

    def test___deepcopy___doctest(self):
        pass  # no doctest

    def test___deepcopy___empty(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = memory1.__copy__()
        memory2.validate()
        assert memory1.span == memory2.span
        assert memory1.trim_span == memory2.trim_span
        assert memory1.content_span == memory2.content_span
        assert memory1.content_parts == memory2.content_parts
        assert all(b1 == b2 for b1, b2 in zip(memory1.blocks(), memory2.blocks()))

    def test___deepcopy___template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory1 = Memory.from_blocks(blocks, copy=False)
        memory2 = memory1.__deepcopy__()
        memory2.validate()
        assert memory1.span == memory2.span
        assert memory1.trim_span == memory2.trim_span
        assert memory1.content_span == memory2.content_span
        assert memory1.content_parts == memory2.content_parts
        assert all(b1 == b2 for b1, b2 in zip(memory1.blocks(), memory2.blocks()))

    def test_contiguous_doctest(self):
        pass  # no doctest

    def test_contiguous(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.contiguous

        memory = Memory(start=1, endex=9)
        assert not memory.contiguous

        memory = Memory.from_bytes(b'xyz', offset=3)
        assert memory.contiguous

        memory = Memory.from_bytes(b'xyz', offset=3, start=3, endex=6)
        assert memory.contiguous

        memory = Memory.from_bytes(b'xyz', offset=3, start=1)
        assert not memory.contiguous

        memory = Memory.from_bytes(b'xyz', offset=3, endex=9)
        assert not memory.contiguous

        memory = Memory.from_bytes(b'xyz', offset=3, start=1, endex=9)
        assert not memory.contiguous

        memory = Memory.from_bytes(b'xyz')
        assert memory.contiguous

        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        assert not memory.contiguous

        memory.trim_endex = MAX_SIZE
        assert not memory.contiguous

        memory.trim_endex = None
        memory.trim_start = 0
        assert not memory.contiguous

    def test_trim_start_doctest(self):
        pass  # no doctest

    def test_trim_endex_doctest(self):
        pass  # no doctest

    def test_trim_start_bytes(self):
        Memory = self.Memory
        data = bytes(range(8))
        memory = Memory.from_bytes(data)

        for offset in range(8):
            memory.trim_start = offset
            memory.validate()
            assert memory.content_start == offset
            assert memory.content_endex == 8

        for offset in reversed(range(8)):
            memory.trim_start = offset
            memory.validate()
            assert memory.content_start == 7
            assert memory.content_endex == 8

        memory.trim_start = 8
        assert memory.content_start == 8
        assert memory.content_endex == 8

        for offset in range(8):
            memory.trim_start = offset
            memory.validate()
            assert memory.content_start == offset
            assert memory.content_endex == offset

        for offset in reversed(range(8)):
            memory.trim_start = offset
            memory.validate()
            assert memory.content_start == offset
            assert memory.content_endex == offset

        memory.trim_endex = None
        memory.validate()
        assert memory.trim_endex is None
        memory.trim_start = 9
        memory.validate()
        assert memory.trim_start == 9
        assert memory.trim_endex is None

        memory.trim_start = 1
        memory.validate()
        memory.trim_endex = 5
        memory.validate()
        assert memory.trim_start == 1
        assert memory.trim_endex == 5
        memory.trim_start = 9
        memory.validate()
        assert memory.trim_start == 9
        assert memory.trim_endex == 9

    def test_trim_endex_bytes(self):
        Memory = self.Memory
        data = bytes(range(8))
        memory = Memory.from_bytes(data)

        for offset in range(8, 0, -1):
            memory.trim_endex = offset
            memory.validate()
            assert memory.content_start == 0
            assert memory.content_endex == offset

        for offset in range(1, 8):
            memory.trim_endex = offset
            memory.validate()
            assert memory.content_start == 0
            assert memory.content_endex == 1

        memory.trim_endex = 0
        memory.validate()
        assert memory.content_start == 0
        assert memory.content_endex == 0

        for offset in range(8, 0, -1):
            memory.trim_endex = offset
            memory.validate()
            assert memory.content_start == 0
            assert memory.content_endex == 0

        for offset in range(1, 8):
            memory.trim_endex = offset
            memory.validate()
            assert memory.content_start == 0
            assert memory.content_endex == 0

        memory.trim_start = None
        memory.validate()
        assert memory.trim_start is None
        memory.trim_endex = 9
        memory.validate()
        assert memory.trim_start is None
        assert memory.trim_endex == 9

        memory.trim_start = 5
        memory.validate()
        memory.trim_endex = 9
        memory.validate()
        assert memory.trim_start == 5
        assert memory.trim_endex == 9
        memory.trim_endex = 1
        memory.validate()
        assert memory.trim_start == 1
        assert memory.trim_endex == 1

    def test_trim_start_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for offset in range(1, MAX_SIZE):
            memory.trim_start = offset
            memory.validate()
            blocks_out = memory.to_blocks()
            values[offset - 1] = None
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

        blocks_ref = values_to_blocks(values)
        for offset in reversed(range(MAX_SIZE)):
            memory.trim_start = offset
            memory.validate()
            blocks_out = memory.to_blocks()
            assert blocks_out == blocks_ref

    def test_trim_endex_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for offset in reversed(range(MAX_SIZE)):
            memory.trim_endex = offset
            memory.validate()
            blocks_out = memory.to_blocks()
            values[offset] = None
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

        blocks_ref = values_to_blocks(values)
        for offset in range(MAX_SIZE):
            memory.trim_endex = offset
            memory.validate()
            blocks_out = memory.to_blocks()
            assert blocks_out == blocks_ref

    def test_trim_span_doctest(self):
        pass  # no doctest

    def test_trim_span(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.trim_span == (None, None)
        memory.trim_span = (1, 9)
        memory.validate()
        assert memory.trim_span == (1, 9)
        memory.trim_span = (5, 5)
        memory.validate()
        assert memory.trim_span == (5, 5)

    def test_start_doctest(self):
        Memory = self.Memory

        assert Memory().start == 0

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.start == 1

        memory = Memory.from_blocks([[5, b'xyz']], start=1)
        assert memory.start == 1

    def test_start_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.start == 0
        assert memory.content_start == 0
        assert memory.trim_start is None

        start = 123
        memory = Memory(start=start)
        assert memory.start == start
        assert memory.content_start == start

    def test_start(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        assert memory.start == blocks[0][0]
        assert memory.content_start == blocks[0][0]
        assert memory.trim_start is None

    def test_endex_doctest(self):
        Memory = self.Memory

        assert Memory().endex == 0

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.endex == 8

        memory = Memory.from_blocks([[1, b'ABC']], endex=8)
        assert memory.endex == 8

    def test_endex_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.endex == 0
        assert memory.content_endex == 0
        assert memory.trim_endex is None

    def test_endex(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        block_start, block_data = blocks[-1]
        block_endex = block_start + len(block_data)
        assert memory.endex == block_endex
        assert memory.content_endex == block_endex
        assert memory.trim_endex is None

    def test_span_doctest(self):
        Memory = self.Memory

        assert Memory().span == (0, 0)
        assert Memory(start=1, endex=8).span == (1, 8)

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.span == (1, 8)

    def test_span(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.span == (0, 0)

        memory.trim_span = (1, 9)
        assert memory.span == (1, 9)

        memory.trim_span = (9, 1)
        assert memory.span == (9, 9)

        memory.trim_span = (None, None)
        assert memory.span == (0, 0)

        memory.write(5, b'xyz')
        assert memory.span == (5, 8)

        memory.trim_span = (1, 9)
        assert memory.span == (1, 9)

        memory.trim_span = (None, None)
        assert memory.span == (5, 8)

    def test_endin_doctest(self):
        Memory = self.Memory

        assert Memory().endin == -1

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.endin == 7

        memory = Memory.from_blocks([[1, b'ABC']], endex=8)
        assert memory.endin == 7

    def test_endin_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.endin == -1
        assert memory.content_endin == -1

        endex = 123
        memory = Memory(endex=endex)
        assert memory.endin == endex - 1
        assert memory.content_endin == -1

        start = 33
        memory = Memory(start=start, endex=endex)
        assert memory.endin == endex - 1
        assert memory.content_endin == start - 1

    def test_endin(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        block_start, block_data = blocks[-1]
        block_endin = block_start + len(block_data) - 1
        assert memory.endin == block_endin
        assert memory.content_endin == block_endin

        endex = 123
        memory = Memory.from_blocks(blocks, endex=endex)
        assert memory.endin == endex - 1
        assert memory.content_endin == block_endin

        start = 1
        memory = Memory.from_blocks(blocks, start=start, endex=endex)
        assert memory.endin == endex - 1
        assert memory.content_endin == block_endin

    def test_content_start_doctest(self):
        Memory = self.Memory

        assert Memory().content_start == 0
        assert Memory(start=1).content_start == 1
        assert Memory(start=1, endex=8).content_start == 1

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.content_start == 1

        memory = Memory.from_blocks([[5, b'xyz']], start=1)
        assert memory.content_start == 5

    def test_content_start(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_start == memory.start
        assert memory.content_start == 0

        memory.write(5, b'xyz')
        assert memory.content_start == memory.start
        assert memory.content_start == 5

    def test_content_start_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        assert memory.content_start == memory.start
        assert memory.content_start == blocks[0][0]

        memory.trim_start = 0
        assert memory.content_start > memory.start
        assert memory.content_start == blocks[0][0]

    def test_content_endex_doctest(self):
        Memory = self.Memory

        assert Memory().content_endex == 0
        assert Memory(endex=8).content_endex == 0
        assert Memory(start=1, endex=8).content_endex == 1

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.content_endex == 8

        memory = Memory.from_blocks([[1, b'ABC']], endex=8)
        assert memory.content_endex == 4

    def test_content_endex(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_endex == memory.endex
        assert memory.content_endex == 0

        memory.write(5, b'xyz')
        assert memory.content_endex == memory.endex
        assert memory.content_endex == 8

    def test_content_endex_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        endex = blocks[-1][0] + len(blocks[-1][1])
        assert memory.content_endex == memory.endex
        assert memory.content_endex == endex

        memory.trim_endex = MAX_SIZE
        assert memory.content_endex < memory.endex
        assert memory.content_endex == endex

    def test_content_span_doctest(self):
        Memory = self.Memory

        assert Memory().content_span == (0, 0)
        assert Memory(start=1).content_span == (1, 1)
        assert Memory(endex=8).content_span == (0, 0)
        assert Memory(start=1, endex=8).content_span == (1, 1)

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.content_span == (1, 8)

    def test_content_span(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_span == (0, 0)

        memory.write(5, b'xyz')
        assert memory.content_span == (5, 8)

        memory.trim_span = (1, 9)
        assert memory.content_span == (5, 8)

        memory.trim_span = (None, None)
        assert memory.content_span == (5, 8)

    def test_content_span_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        start = blocks[0][0]
        endex = blocks[-1][0] + len(blocks[-1][1])
        assert memory.content_span == (start, endex)

        memory.trim_span = (0, MAX_SIZE)
        assert memory.content_span == (start, endex)

        memory.trim_span = (None, None)
        assert memory.content_span == (start, endex)

    def test_content_endin_doctest(self):
        Memory = self.Memory

        assert Memory().content_endin == -1
        assert Memory(endex=8).content_endin == -1
        assert Memory(start=1, endex=8).content_endin == 0

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.content_endin == 7

        memory = Memory.from_blocks([[1, b'ABC']], endex=8)
        assert memory.content_endin == 3

    def test_content_endin(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_endin == memory.endin
        assert memory.content_endin == 0 - 1

        memory.write(5, b'xyz')
        assert memory.content_endin == memory.endin
        assert memory.content_endin == 8 - 1

    def test_content_endin_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        endin = blocks[-1][0] + len(blocks[-1][1]) - 1
        assert memory.content_endin == memory.endin
        assert memory.content_endin == endin

        memory.trim_endex = MAX_SIZE
        assert memory.content_endin < memory.endin
        assert memory.content_endin == endin

    def test_content_items_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
        assert dict(memory.content_items()) == {1: 65, 2: 66, 5: 120, 7: 49, 8: 50, 9: 51}
        assert dict(memory.content_items(2, 9)) == {2: 66, 5: 120, 7: 49, 8: 50}
        assert dict(memory.content_items(3, 5)) == {}

    def test_content_items_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert dict(memory.content_items()) == {}

    def test_content_keys_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
        assert list(memory.content_keys()) == [1, 2, 5, 7, 8, 9]
        assert list(memory.content_keys(2, 9)) == [2, 5, 7, 8]
        assert list(memory.content_keys(3, 5)) == []

    def test_content_keys_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert list(memory.content_keys()) == []

    def test_content_size_doctest(self):
        Memory = self.Memory

        assert Memory().content_size == 0
        assert Memory(start=1, endex=8).content_size == 0

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.content_size == 6

        memory = Memory.from_blocks([[1, b'ABC']], endex=8)
        assert memory.content_size == 3

    def test_content_size(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_size == len(memory)
        assert memory.content_size == 0

        memory.write(5, b'xyz')
        assert memory.content_size == len(memory)
        assert memory.content_size == 3

        memory.trim_span = (1, 9)
        assert memory.content_size == 3

    def test_content_parts_doctest(self):
        Memory = self.Memory

        assert Memory().content_parts == 0

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.content_parts == 2

        memory = Memory.from_blocks([[1, b'ABC']], endex=8)
        assert memory.content_parts == 1

    def test_content_parts(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_parts == 0

        memory.write(5, b'xyz')
        assert memory.content_parts == 1

    def test_content_parts_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)
        assert memory.content_parts == len(blocks)

    def test_content_values_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
        assert list(memory.content_values()) == [65, 66, 120, 49, 50, 51]
        assert list(memory.content_values(2, 9)) == [66, 120, 49, 50]
        assert list(memory.content_values(3, 5)) == []

    def test_content_values_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert list(memory.content_values()) == []

    def test_validate_doctest(self):
        pass  # no doctest

    def test_validate_empty(self):
        Memory = self.Memory
        memory = Memory()
        memory.validate()

    def test_validate_invalid_bounds(self):
        Memory = self.Memory
        blocks = [[10, b'ABC'], [5, b'xyz']]
        memory = Memory.from_blocks(blocks, validate=False)

        with pytest.raises(ValueError, match='invalid bounds'):
            memory.validate()

    def test_validate_invalid_block_interleaving(self):
        Memory = self.Memory

        blocks = [[2, b'ABC'], [5, b'xyz']]
        memory = Memory.from_blocks(blocks, validate=False)

        with pytest.raises(ValueError, match='invalid block interleaving'):
            memory.validate()

        blocks = [[2, b'ABC'], [3, b'xyz']]
        memory = Memory.from_blocks(blocks, validate=False)

        with pytest.raises(ValueError, match='invalid block interleaving'):
            memory.validate()

    def test_validate_invalid_block_bounds(self):
        Memory = self.Memory

        blocks = [[1, b'ABC']]
        memory = Memory.from_blocks(blocks, start=3, endex=6, validate=False)

        with pytest.raises(ValueError, match='invalid block bounds'):
            memory.validate()

        blocks = [[5, b'xyz']]
        memory = Memory.from_blocks(blocks, start=3, endex=6, validate=False)

        with pytest.raises(ValueError, match='invalid block bounds'):
            memory.validate()

        blocks = [[0, b'123'], [10, b'ABC'], [5, b'xyz']]
        memory = Memory.from_blocks(blocks, validate=False)

        with pytest.raises(ValueError, match='invalid block bounds'):
            memory.validate()

    def test_bound_doctest(self):
        Memory = self.Memory

        assert Memory().bound(None, None) == (0, 0)
        assert Memory().bound(None, 100) == (0, 100)

        memory = Memory.from_blocks([[1, b'ABC'], [5, b'xyz']])
        assert memory.bound(0, 30) == (0, 30)
        assert memory.bound(2, 6) == (2, 6)
        assert memory.bound(None, 6) == (1, 6)
        assert memory.bound(2, None) == (2, 8)

        memory = Memory.from_blocks([[3, b'ABC']], start=1, endex=8)
        assert memory.bound(None, None) == (1, 8)
        assert memory.bound(0, 30) == (1, 8)
        assert memory.bound(2, 6) == (2, 6)
        assert memory.bound(2, None) == (2, 8)
        assert memory.bound(None, 6) == (1, 6)

    def test_bound_none(self):
        Memory = self.Memory
        memory = Memory()

        bound = memory.bound(11, 44)
        assert bound == (11, 44)

        bound = memory.bound(22, 33)
        assert bound == (22, 33)

        bound = memory.bound(0, 99)
        assert bound == (0, 99)

        bound = memory.bound(0, 0)
        assert bound == (0, 0)

        bound = memory.bound(99, 99)
        assert bound == (99, 99)

        bound = memory.bound(99, 0)
        assert bound == (99, 99)

        bound = memory.bound(None, 44)
        assert bound == (0, 44)

        bound = memory.bound(None, 33)
        assert bound == (0, 33)

        bound = memory.bound(None, 99)
        assert bound == (0, 99)

        bound = memory.bound(11, None)
        assert bound == (11, 11)

        bound = memory.bound(22, None)
        assert bound == (22, 22)

        bound = memory.bound(0, None)
        assert bound == (0, 0)

    def test_bound_span(self):
        Memory = self.Memory
        memory = Memory(start=11, endex=44)

        bound = memory.bound(11, 44)
        assert bound == (11, 44)

        bound = memory.bound(22, 33)
        assert bound == (22, 33)

        bound = memory.bound(0, 99)
        assert bound == (11, 44)

        bound = memory.bound(0, 0)
        assert bound == (11, 11)

        bound = memory.bound(99, 99)
        assert bound == (44, 44)

        bound = memory.bound(99, 0)
        assert bound == (44, 44)

        bound = memory.bound(None, 44)
        assert bound == (11, 44)

        bound = memory.bound(None, 33)
        assert bound == (11, 33)

        bound = memory.bound(None, 99)
        assert bound == (11, 44)

        bound = memory.bound(11, None)
        assert bound == (11, 44)

        bound = memory.bound(22, None)
        assert bound == (22, 44)

        bound = memory.bound(0, None)
        assert bound == (11, 44)

    def test_bound_start(self):
        Memory = self.Memory
        memory = Memory(start=11)

        bound = memory.bound(11, 44)
        assert bound == (11, 44)

        bound = memory.bound(22, 33)
        assert bound == (22, 33)

        bound = memory.bound(0, 99)
        assert bound == (11, 99)

        bound = memory.bound(0, 0)
        assert bound == (11, 11)

        bound = memory.bound(99, 99)
        assert bound == (99, 99)

        bound = memory.bound(99, 0)
        assert bound == (99, 99)

        bound = memory.bound(None, 44)
        assert bound == (11, 44)

        bound = memory.bound(None, 33)
        assert bound == (11, 33)

        bound = memory.bound(None, 99)
        assert bound == (11, 99)

        bound = memory.bound(11, None)
        assert bound == (11, 11)

        bound = memory.bound(22, None)
        assert bound == (22, 22)

        bound = memory.bound(0, None)
        assert bound == (11, 11)

    def test_bound_endex(self):
        Memory = self.Memory
        memory = Memory(endex=44)

        bound = memory.bound(11, 44)
        assert bound == (11, 44)

        bound = memory.bound(22, 33)
        assert bound == (22, 33)

        bound = memory.bound(0, 99)
        assert bound == (0, 44)

        bound = memory.bound(0, 0)
        assert bound == (0, 0)

        bound = memory.bound(99, 99)
        assert bound == (44, 44)

        bound = memory.bound(99, 0)
        assert bound == (44, 44)

        bound = memory.bound(None, 44)
        assert bound == (0, 44)

        bound = memory.bound(None, 33)
        assert bound == (0, 33)

        bound = memory.bound(None, 99)
        assert bound == (0, 44)

        bound = memory.bound(11, None)
        assert bound == (11, 44)

        bound = memory.bound(22, None)
        assert bound == (22, 44)

        bound = memory.bound(0, None)
        assert bound == (0, 44)

    def test__block_index_at_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        blocks_index_out = [memory._block_index_at(i) for i in range(12)]
        blocks_index_ref = [None, 0, 0, 0, 0, None, 1, None, 2, 2, 2, None]
        assert blocks_index_out == blocks_index_ref

    def test__block_index_at_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)

        blocks_index_ref: List[Any] = [None] * MAX_SIZE
        for block_index, (block_start, block_data) in enumerate(blocks):
            for offset in range(len(block_data)):
                blocks_index_ref[block_start + offset] = block_index

        blocks_index_out = [memory._block_index_at(address) for address in range(MAX_SIZE)]
        assert blocks_index_out == blocks_index_ref

    def test__block_index_at_empty(self):
        Memory = self.Memory
        memory = Memory()
        blocks_index_out = [memory._block_index_at(address) for address in range(MAX_SIZE)]
        blocks_index_ref = [None] * MAX_SIZE
        assert blocks_index_out == blocks_index_ref

    def test__block_index_start_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        blocks_index_out = [memory._block_index_start(i) for i in range(12)]
        blocks_index_ref = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
        assert blocks_index_out == blocks_index_ref

    def test__block_index_start(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)

        blocks_index_ref = [len(blocks)] * MAX_SIZE
        for block_index in reversed(range(len(blocks))):
            block_start, block_data = blocks[block_index]
            block_endex = block_start + len(block_data)
            for offset in range(block_endex):
                blocks_index_ref[offset] = block_index

        blocks_index_out = [memory._block_index_start(address) for address in range(MAX_SIZE)]
        assert blocks_index_out == blocks_index_ref

    def test__block_index_endex_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        blocks_index_out = [memory._block_index_endex(i) for i in range(12)]
        blocks_index_ref = [0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
        assert blocks_index_out == blocks_index_ref

    def test__block_index_endex(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory.from_blocks(blocks)

        blocks_index_ref = [0] * MAX_SIZE
        for block_index in range(len(blocks)):
            block_start = blocks[block_index][0]
            block_index += 1
            for offset in range(block_start, MAX_SIZE):
                blocks_index_ref[offset] = block_index

        blocks_index_out = [memory._block_index_endex(address) for address in range(MAX_SIZE)]
        assert blocks_index_out == blocks_index_ref

    def test__pretrim_start_unbounded(self):
        Memory = self.Memory
        data = b'56789'
        memory = Memory.from_bytes(data, offset=5)

        memory_backup = memory.__deepcopy__()
        backup = memory._pretrim_start_backup(None, 4)
        assert not backup.to_blocks()

        memory._pretrim_start(None, 4)
        memory.validate()

        memory.write(0, backup)
        memory.validate()
        assert memory == memory_backup

    def test__pretrim_start_bounded(self):
        Memory = self.Memory
        data = b'56789'
        memory = Memory.from_bytes(data, offset=5, start=3)

        memory_backup = memory.__deepcopy__()
        backup = memory._pretrim_start_backup(7, 4)
        assert backup.to_blocks() == [[5, b'56']]

        memory._pretrim_start(7, 4)
        memory.validate()

        memory.write(0, backup)
        memory.validate()
        assert memory == memory_backup

    def test__pretrim_endex_unbounded(self):
        Memory = self.Memory
        data = b'34567'
        memory = Memory.from_bytes(data, offset=3)

        memory_backup = memory.__deepcopy__()
        backup = memory._pretrim_endex_backup(None, 4)
        assert not backup.to_blocks()

        memory._pretrim_endex(None, 4)
        memory.validate()

        memory.write(0, backup)
        memory.validate()
        assert memory == memory_backup

    def test__pretrim_endex_bounded(self):
        Memory = self.Memory
        data = b'34567'
        memory = Memory.from_bytes(data, offset=3, endex=9)

        memory_backup = memory.__deepcopy__()
        backup = memory._pretrim_endex_backup(6, 4)
        assert backup.to_blocks() == [[6, b'67']]

        memory._pretrim_endex(6, 4)
        memory.validate()

        memory.write(0, backup)
        memory.validate()
        assert memory == memory_backup

    def test_get_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        assert memory.get(3) == 67
        assert memory.get(6) == 36
        assert memory.get(10) == 122
        assert memory.get(0) is None
        assert memory.get(7) is None
        assert memory.get(11) is None
        assert memory.get(0, 123) == 123
        assert memory.get(7, 123) == 123
        assert memory.get(11, 123) == 123

    def test_get_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for address in range(MAX_START):
            value = memory.get(address)
            assert value == values[address]

            if values[address] is None:
                value = memory.get(address, 123)
                assert value == 123

    def test_peek_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        assert memory.peek(3) == 67
        assert memory.peek(6) == 36
        assert memory.peek(10) == 122
        assert memory.peek(0) is None
        assert memory.peek(7) is None
        assert memory.peek(11) is None

    def test_peek_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for address in range(MAX_START):
            value = memory.peek(address)
            assert value == values[address]

    def test_poke_doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        memory.poke(3, b'@')
        assert memory.peek(3) == 64

        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        memory.poke(5, b'@')
        assert memory.peek(5) == 64

    def test_poke_value_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            memory_backup = memory.__deepcopy__()
            backup_address, backup_value = memory.poke_backup(start)
            assert backup_address == start
            assert backup_value == values[start]

            memory.poke(start, start)
            memory.validate()
            blocks_out = memory.to_blocks()

            values[start] = start
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref

            memory.poke_restore(backup_address, backup_value)
            memory.validate()
            assert memory == memory_backup

    def test_poke_single_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            memory.poke(start, bytes([start]))
            memory.validate()
            blocks_out = memory.to_blocks()

            values[start] = start
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref

    def test_poke_multi_template(self):
        Memory = self.Memory
        match = r'expecting single item'

        for start in range(MAX_START):
            blocks = create_template_blocks()
            memory = Memory.from_blocks(blocks)

            with pytest.raises(ValueError, match=match):
                memory.poke(start, b'123')

    def test_poke_none_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            memory.poke(start, None)
            memory.validate()
            blocks_out = memory.to_blocks()

            values[start] = None
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref

    def test_poke_backup_doctest(self):
        pass  # no doctest

    def test_poke_restore_doctest(self):
        pass  # no doctest

    def test_setdefault_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])

        assert memory.setdefault(3, b'@') == 67
        memory.validate()
        assert memory.peek(3) == 67

        assert memory.setdefault(5, 64) == 64
        memory.validate()
        assert memory.peek(5) == 64

        assert memory.setdefault(9) is not None
        memory.validate()
        assert memory.peek(9) is not None

        assert memory.setdefault(7) is None
        memory.validate()
        assert memory.peek(7) is None

    def test_setdefault_multi(self):
        Memory = self.Memory
        memory = Memory()

        value = memory.setdefault(0, b'X')
        assert value == b'X'[0]

        with pytest.raises(ValueError, match='expecting single item'):
            memory.setdefault(1, b'YZ')

    def test_setdefault_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)
            memory_backup = memory.__deepcopy__()

            backup_address, backup_value = memory.setdefault_backup(start)
            assert backup_address == start
            assert backup_value == values[start]

            answer = memory.setdefault(start, start)
            memory.validate()
            if values[start] is None:
                assert answer == start
                values[start] = start
            else:
                assert answer == values[start]
            blocks_out = memory.to_blocks()
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

            memory.setdefault_restore(backup_address, backup_value)
            memory.validate()
            assert memory == memory_backup

    def test_setdefault_backup_doctest(self):
        pass  # no doctest

    def test_setdefault_restore_doctest(self):
        pass  # no doctest

    def test_update_doctest(self):
        Memory = self.Memory
        memory = Memory()

        memory.update(Memory.from_bytes(b'ABC', 5))
        memory.validate()
        assert memory.to_blocks() == [[5, b'ABC']]

        memory.update({1: b'x', 2: ord('y')})
        memory.validate()
        assert memory.to_blocks() == [[1, b'xy'], [5, b'ABC']]

        memory.update([(6, b'?'), (3, ord('@'))])
        memory.validate()
        assert memory.to_blocks() == [[1, b'xy@'], [5, b'A?C']]

    def test_update_memory(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = Memory.from_bytes(b'ABC', 5)
        memory_backup = memory1.__deepcopy__()

        backup, = memory1.update_backup(memory2)
        start = memory2.start
        endex = memory2.endex
        memory_ref = memory_backup.extract(start=start, endex=endex)
        assert backup == memory_ref

        memory1.update(memory2)
        memory1.validate()
        assert memory1.to_blocks() == [[5, b'ABC']]

        memory1.update_restore([backup])
        memory1.validate()
        assert memory1 == memory_backup

    def test_update_dict(self):
        Memory = self.Memory
        memory = Memory()
        data = {6: ord('B'), 5: ord('A'), 7: ord('C')}
        memory_backup = memory.__deepcopy__()

        backups = memory.update_backup(data)
        items_ref = dict(memory_backup.items(start=5, endex=8))
        assert backups == items_ref

        memory.update(data)
        memory.validate()
        assert memory.to_blocks() == [[5, b'ABC']]

        memory.update_restore(backups)
        memory.validate()
        assert memory == memory_backup

    def test_update_pairs(self):
        Memory = self.Memory
        memory = Memory()
        data = [(6, ord('B')), (5, ord('A')), (7, ord('C'))]
        memory_backup = memory.__deepcopy__()

        backups = memory.update_backup(data)
        items_ref = dict(memory_backup.items(start=5, endex=8))
        assert backups == items_ref

        memory.update(data)
        memory.validate()
        assert memory.to_blocks() == [[5, b'ABC']]

        memory.update_restore(backups)
        memory.validate()
        assert memory == memory_backup

    def test_update_kwargs(self):
        Memory = self.Memory
        memory = Memory()
        with pytest.raises(KeyError, match='cannot convert'):
            memory.update((), dummy=0)
        with pytest.raises(KeyError, match='cannot convert'):
            memory.update_backup((), dummy=0)

    def test_update_template_memory(self):
        Memory = self.Memory
        memory1 = Memory.from_blocks(create_template_blocks())
        memory2 = Memory.from_blocks(create_hello_world_blocks())
        memory_backup = memory1.__deepcopy__()

        backup, = memory1.update_backup(memory2, clear=True)
        start = memory2.start
        endex = memory2.endex
        memory_ref = memory_backup.extract(start=start, endex=endex)
        assert backup == memory_ref

        memory1.update(memory2)
        memory1.validate()
        memory_ref = memory_backup.__deepcopy__()
        memory_ref.write(0, memory2)
        assert memory1 == memory_ref

        memory1.update_restore([backup])
        memory1.validate()
        assert memory1 == memory_backup

    def test_update_template_mapping(self):
        Memory = self.Memory
        memory1 = Memory.from_blocks(create_template_blocks())
        memory2 = Memory.from_blocks(create_hello_world_blocks())
        memory2 = {k: v for k, v in memory2.items() if v is not None}
        memory_backup = memory1.__deepcopy__()

        backup = memory1.update_backup(memory2)
        backup_dict_out = dict(backup.items())
        backup_dict_ref = {k: memory_backup[k] for k in memory2.keys()}
        assert backup_dict_out == backup_dict_ref

        memory1.update(memory2)
        memory1.validate()
        memory_ref = dict(memory_backup.items())
        memory_ref.update(memory2)
        values_out = list(memory1.values(0, MAX_SIZE))
        values_ref = [memory_ref.get(k) for k in range(MAX_SIZE)]
        assert values_out == values_ref

        memory1.update_restore(backup)
        memory1.validate()
        assert memory1 == memory_backup

    def test_update_backup_doctest(self):
        pass  # no doctest

    def test_update_restore_doctest(self):
        pass  # no doctest

    def test_extract_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        assert memory.extract().to_blocks() == [[1, b'ABCD'], [6, b'$'], [8, b'xyz']]
        assert memory.extract(start=2, endex=9).to_blocks() == [[2, b'BCD'], [6, b'$'], [8, b'x']]
        assert memory.extract(start=2).to_blocks() == [[2, b'BCD'], [6, b'$'], [8, b'xyz']]
        assert memory.extract(endex=9).to_blocks() == [[1, b'ABCD'], [6, b'$'], [8, b'x']]
        assert memory.extract(start=5, endex=8).span == (5, 8)
        assert memory.extract(pattern=b'.').to_blocks() == [[1, b'ABCD.$.xyz']]
        assert memory.extract(pattern=b'.', step=3).to_blocks() == [[1, b'AD.z']]

    def test_extract_negative(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        extracted = memory.extract(start=2, endex=0)
        assert extracted.span == (2, 2)
        assert extracted.to_blocks() == []

    def test_extract_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)

                for bound in (False, True):
                    extracted = memory.extract(start=start, endex=(start + size), bound=bound)
                    extracted.validate()
                    assert not bound or extracted.span == (start, start + size)
                    blocks_out = extracted.to_blocks()
                    blocks_ref = values_to_blocks(values[start:(start + size)], start)
                    assert blocks_out == blocks_ref

    def test_extract_step_negative_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                memory = Memory.from_blocks(blocks)

                for step in range(-1, 1):
                    extracted = memory.extract(start=start, endex=(start + size), step=step)
                    extracted.validate()
                    blocks_out = extracted.to_blocks()
                    blocks_ref = []
                    assert blocks_out == blocks_ref

    def test_extract_step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                for step in range(1, MAX_TIMES):
                    for bound in (False, True):
                        extracted = memory.extract(start=start, endex=endex, step=step, bound=bound)
                        extracted.validate()
                        blocks_out = extracted.to_blocks()
                        blocks_ref = values_to_blocks(values[start:endex:step], start)
                        assert blocks_out == blocks_ref

    def test_extract_pattern_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                pattern = b'xyz'

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                for index in range(size):
                    if values[start + index] is None:
                        values[start + index] = tiled[index]

                extracted = memory.extract(start=start, endex=(start + size), pattern=pattern)
                blocks_out = extracted.to_blocks()

                blocks_ref = values_to_blocks(values[start:(start + size)], start)
                assert blocks_out == blocks_ref

    def test_cut_all(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])

        memory_backup = memory.__deepcopy__()
        backup = memory.cut()
        assert backup.span == memory_backup.span
        assert backup == memory_backup.extract()

        memory.write(0, backup)
        memory.validate()
        assert memory == memory_backup

    def test_cut_negative(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        cut = memory.cut(2, 0)
        assert cut.span == (2, 2)
        assert cut.to_blocks() == []

    def test_cut_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                endex = start + size
                for bound in (False, True):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory.from_blocks(blocks)
                    memory_backup = memory.__deepcopy__()

                    cut = memory.cut(start, endex, bound=bound)
                    memory.validate()
                    cut.validate()
                    assert not bound or cut.span == (start, endex)
                    extracted = memory_backup.extract(start=start, endex=endex)
                    assert cut == extracted
                    blocks_out = cut.to_blocks()
                    blocks_ref = values_to_blocks(values[start:endex], start)
                    assert blocks_out == blocks_ref

                    memory.write(0, cut)
                    memory.validate()
                    assert memory == memory_backup

    def test_view_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABCD'], [6, b'$'], [8, b'xyz']])
        assert bytes(memory.view(2, 5)) == b'BCD'
        assert bytes(memory.view(9, 10)) == b'y'
        match = 'non-contiguous data within range'

        with pytest.raises(ValueError, match=match):
            memory.view()
        with pytest.raises(ValueError, match=match):
            memory.view(0, 6)

    def test_view_template(self):
        Memory = self.Memory
        match = 'non-contiguous data within range'
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                values_ref = values[start:(start + size)]

                if all(x is not None for x in values_ref):
                    values_out = list(memory.view(start, start + size))
                    assert values_out == values_ref
                else:
                    with pytest.raises(ValueError, match=match):
                        memory.view(start, start + size)

    def test_shift_doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
        memory.shift(-2)
        assert memory.to_blocks() == [[3, b'ABC'], [7, b'xyz']]

        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']], start=3)
        memory.shift(-7)
        assert memory.to_blocks() == [[3, b'yz']], memory.to_blocks()

    def test_shift_template(self):
        Memory = self.Memory
        for offset in range(-MAX_SIZE if self.ADDR_NEG else 0, MAX_SIZE):
            memory = Memory.from_blocks(create_template_blocks())
            blocks_ref = create_template_blocks()
            for block in blocks_ref:
                block[0] += offset

            memory.shift(offset)
            memory.validate()
            blocks_out = memory.to_blocks()
            assert blocks_out == blocks_ref

    def test_shift_bounded_template(self):
        Memory = self.Memory
        for offset in range(-MAX_SIZE if self.ADDR_NEG else 0, MAX_SIZE):
            blocks = create_template_blocks()
            memory = Memory.from_blocks(blocks, 0, 1, MAX_SIZE - 1)
            values = blocks_to_values(blocks, MAX_SIZE)

            memory_backup = memory.__deepcopy__()
            backup_offset, backup = memory.shift_backup(offset)
            assert backup_offset == offset

            memory.shift(offset)
            memory.validate()
            blocks_out = memory.to_blocks()

            if offset < 0:
                values_ref = values[-offset:]
                values_ref += [None] * (MAX_SIZE + offset)
                values_ref[0] = None
            else:
                values_ref = values[:(MAX_SIZE - offset)]
                values_ref[0:0] = [None] * offset
                values_ref[-1] = None
            blocks_ref = values_to_blocks(values_ref)

            assert blocks_out == blocks_ref

            memory.shift_restore(backup_offset, backup)
            memory.validate()
            assert memory == memory_backup

    def test_shift_backup_doctest(self):
        pass  # no doctest

    def test_shift_restore_doctest(self):
        pass  # no doctest

    def test_reserve_doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[3, b'ABC'], [7, b'xyz']])
        memory.reserve(4, 2)
        assert memory.to_blocks() == [[3, b'A'], [6, b'BC'], [9, b'xyz']]

        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']], endex=12)
        memory.reserve(5, 5)
        assert memory.to_blocks() == [[10, b'AB']]

    def test_reserve_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)

                memory.reserve(start, size)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:start] = [None] * size
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

    def test_reserve_bounded_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks, 0, 1, MAX_SIZE - 1)

                memory_backup = memory.__deepcopy__()
                backup_address, backup = memory.reserve_backup(start, size)
                assert backup_address == start

                memory.reserve(start, size)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:start] = [None] * size
                values[0] = None
                del values[MAX_SIZE - 1:]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.reserve_restore(backup_address, backup)
                memory.validate()
                assert memory == memory_backup

    def test_reserve_backup_doctest(self):
        pass  # no doctest

    def test_reserve_restore_doctest(self):
        pass  # no doctest

    def test_insert_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        memory.insert(10, b'$')
        assert memory.to_blocks() == [[1, b'ABC'], [6, b'xyz'], [10, b'$']]
        memory.insert(8, b'1')
        assert memory.to_blocks() == [[1, b'ABC'], [6, b'xy1z'], [11, b'$']]

    def test_insert_single(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            memory.insert(start, start)
            memory.validate()
            blocks_out = memory.to_blocks()

            values[start:start] = [start]
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

    def test_insert_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                data = bytes(range(ord('a'), ord('a') + size))

                memory.insert(start, data)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:start] = data
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

    def test_insert_bounded_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks, 0, 1, MAX_SIZE - 1)
                data = bytes(range(ord('a'), ord('a') + size))

                memory_backup = memory.__deepcopy__()
                backup_address, backup = memory.insert_backup(start, data)
                assert backup_address == start

                memory.insert(start, data)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:start] = data
                values[0] = None
                del values[MAX_SIZE - 1:]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.insert_restore(backup_address, backup)
                memory.validate()
                assert memory == memory_backup

    def test_insert_backup_doctest(self):
        pass  # no doctest

    def test_insert_restore_doctest(self):
        pass  # no doctest

    def test_delete_all(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])

        memory_backup = memory.__deepcopy__()
        backup = memory.delete_backup()
        assert backup.span == memory_backup.span
        assert backup == memory_backup.extract()

        memory.delete()
        assert memory.to_blocks() == []

        memory.delete_restore(backup)
        memory.validate()
        assert memory == memory_backup

    def test_delete_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
        memory.delete(6, 10)
        assert memory.to_blocks() == [[5, b'Ayz']]

    def test_delete_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                memory_backup = memory.__deepcopy__()
                backup = memory.delete_backup(start, endex)
                assert backup == memory_backup.extract(start=start, endex=endex)

                memory.delete(start, endex)
                memory.validate()
                blocks_out = memory.to_blocks()

                del values[start:endex]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.delete_restore(backup)
                memory.validate()
                assert memory == memory_backup

    def test_delete_backup_doctest(self):
        pass  # no doctest

    def test_delete_restore_doctest(self):
        pass  # no doctest

    def test_clear_all(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])

        memory_backup = memory.__deepcopy__()
        backup = memory.clear_backup()
        assert backup.span == memory_backup.span
        assert backup == memory_backup.extract()

        memory.clear()
        assert memory.to_blocks() == []

        memory.clear_restore(backup)
        memory.validate()
        assert memory == memory_backup

    def test_clear_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
        memory.clear(6, 10)
        assert memory.to_blocks() == [[5, b'A'], [10, b'yz']]

    def test_clear_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                memory_backup = memory.__deepcopy__()
                backup = memory.clear_backup(start, endex)
                assert backup.span == (start, endex)
                assert backup == memory_backup.extract(start=start, endex=endex)

                memory.clear(start, endex)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:endex] = [None] * (endex - start)
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.clear_restore(backup)
                memory.validate()
                assert memory == memory_backup

    def test_clear_backup_doctest(self):
        pass  # no doctest

    def test_clear_restore_doctest(self):
        pass  # no doctest

    def test_crop_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])
        memory.crop(6, 10)
        memory.validate()
        assert memory.to_blocks() == [[6, b'BC'], [9, b'x']]

    def test_crop_empty(self):
        Memory = self.Memory
        memory = Memory()

        memory_backup = memory.__deepcopy__()
        backup_start, backup_endex = memory.crop_backup()
        assert backup_start is None
        assert backup_endex is None

        memory.crop()
        memory.validate()
        assert memory == memory_backup

        memory.crop_restore(backup_start, backup_endex)
        assert memory == memory_backup

    def test_crop_all(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[5, b'ABC'], [9, b'xyz']])

        memory_backup = memory.__deepcopy__()
        backup_start, backup_endex = memory.crop_backup()
        assert backup_start is None
        assert backup_endex is None

        memory.crop()
        memory.validate()
        assert memory == memory_backup

        memory.crop_restore(backup_start, backup_endex)
        assert memory == memory_backup

    def test_crop_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                memory_backup = memory.__deepcopy__()
                backup_start, backup_endex = memory.crop_backup(start, endex)
                if backup_start is not None:
                    assert backup_start.endex == start
                    assert backup_start == memory_backup.extract(start=None, endex=start)
                if backup_endex is not None:
                    assert backup_endex.start == endex
                    assert backup_endex == memory_backup.extract(start=endex, endex=None)

                memory.crop(start, endex)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[:start] = [None] * start
                values[endex:] = [None] * (len(values) - endex)
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.crop_restore(backup_start, backup_endex)
                assert memory == memory_backup

    def test_crop_backup_doctest(self):
        pass  # no doctest

    def test_crop_restore_doctest(self):
        pass  # no doctest

    def test_write_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        memory.write(5, b'123')
        assert memory.to_blocks() == [[1, b'ABC'], [5, b'123z']]

    def test_write_single(self):
        Memory = self.Memory
        for offset in range(MAX_SIZE):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)
            memory.trim_start = memory.content_start - 1
            memory.trim_endex = memory.content_endex + 1

            memory.write(offset, offset)
            memory.validate()
            blocks_out = memory.to_blocks()

            if memory.trim_start <= offset < memory.trim_endex:
                values[offset] = offset
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

    def test_write_simple(self):
        Memory = self.Memory
        chunk = b'<=>'
        for offset in range(MAX_SIZE - 3):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)
            memory.trim_start = memory.content_start - 1
            memory.trim_endex = memory.content_endex + 1

            memory_backup = memory.__deepcopy__()
            backup, = memory.write_backup(offset, chunk)
            assert backup.span == (offset, offset + len(chunk))
            assert backup == memory_backup.extract(start=offset, endex=(offset + len(chunk)))

            memory.write(offset, chunk)
            memory.validate()
            blocks_out = memory.to_blocks()

            values[offset:(offset + 3)] = chunk
            values[:memory.trim_start] = [None] * memory.trim_start
            values[memory.trim_endex:] = [None] * (len(values) - memory.trim_endex)
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref

            memory.write_restore([backup])
            memory.validate()
            assert memory == memory_backup

    def test_write_bounded_empty(self):
        Memory = self.Memory
        memory = Memory(start=3, endex=6)

        memory.write(0, b'xyz')
        memory.validate()
        assert not memory

        memory.write(6, b'xyz')
        memory.validate()
        assert not memory

    def test_write_memory_bounded_outside(self):
        Memory = self.Memory
        blocks1 = [[5, b'abc'], [10, b'xyz']]
        memory1 = Memory.from_blocks(blocks1, start=4, endex=14)

        blocks2 = [[1, b'123'], [14, b'ABC']]
        memory2 = Memory.from_blocks(blocks2)

        memory1.write(0, memory2)
        memory1.validate()
        assert memory1.to_blocks() == blocks1

    def test_write_memory_bounded_across(self):
        Memory = self.Memory
        blocks1 = [[5, b'abc'], [10, b'xyz']]
        memory1 = Memory.from_blocks(blocks1, start=4, endex=14)

        blocks2 = [[2, b'123'], [13, b'ABC']]
        memory2 = Memory.from_blocks(blocks2)

        memory1.write(0, memory2)
        memory1.validate()
        blocks_ref = [[4, b'3abc'], [10, b'xyzA']]
        assert memory1.to_blocks() == blocks_ref

    def test_write_memory_empty(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = Memory()
        memory2.write(0, memory1)
        memory2.validate()
        assert not memory2

    def test_write_memory_clear(self):
        Memory = self.Memory
        memory1 = Memory(start=3, endex=7)
        memory2 = Memory.from_bytes(b'0123456789')
        memory2.write(0, memory1, clear=True)
        memory2.validate()
        blocks_out = memory2.to_blocks()
        blocks_ref = [[0, b'012'], [7, b'789']]
        assert blocks_out == blocks_ref

    def test_write_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                data = bytearray(range(ord('a'), ord('a') + size))
                endex = start + len(data)

                memory.write(start, data)
                memory.validate()
                blocks_out = memory.to_blocks()

                values[start:endex] = data
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

    def test_write_noclear_hello_over_template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        values1 = blocks_to_values(blocks1)
        memory1 = Memory.from_blocks(blocks1)

        blocks2 = create_hello_world_blocks()
        memory2 = Memory.from_blocks(blocks2)

        memory1.write(0, memory2, clear=False)
        memory1.validate()

        values_ref = values1[:]
        for block_start, block_data in blocks2:
            block_endex = block_start + len(block_data)
            values_ref[block_start:block_endex] = block_data

        values_out = blocks_to_values(memory1.to_blocks())
        assert values_out == values_ref

    def test_write_clear_hello_over_template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        values1 = blocks_to_values(blocks1)
        memory1 = Memory.from_blocks(blocks1)

        blocks2 = create_hello_world_blocks()
        values2 = blocks_to_values(blocks2)
        memory2 = Memory.from_blocks(blocks2)

        memory1.write(0, memory2, clear=True)
        memory1.validate()

        values_ref = values1[:]
        start, endex = memory2.start, memory2.endex
        values_ref[start:endex] = values2[start:endex]

        values_out = blocks_to_values(memory1.to_blocks())
        assert values_out == values_ref

    def test_write_backup_doctest(self):
        pass  # no doctest

    def test_write_backup_nothing(self):
        Memory = self.Memory
        memory = Memory()

        backups = memory.write_backup(0, Memory())
        assert backups == []

        backups = memory.write_backup(0, b'')
        assert backups == []

    def test_write_backup_value(self):
        Memory = self.Memory
        memory = Memory()

        backup, = memory.write_backup(0, 123)
        assert backup.span == (0, 1)

    def test_write_restore_doctest(self):
        pass  # no doctest

    def test_fill_doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        memory.fill(pattern=b'123')
        assert memory.to_blocks() == [[1, b'12312312']]

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        memory.fill(3, 7, b'123')
        assert memory.to_blocks() == [[1, b'AB1231yz']]

    def test_fill_template(self):
        Memory = self.Memory
        pattern = b'<xyz>'
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                memory_backup = memory.__deepcopy__()
                backup = memory.fill_backup(start, endex)
                assert backup.span == (start, endex)
                assert backup == memory_backup.extract(start=start, endex=endex)

                memory.fill(start, endex, pattern)
                memory.validate()
                blocks_out = memory.to_blocks()

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                values[start:endex] = tiled
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.fill_restore(backup)
                memory.validate()
                assert memory == memory_backup

    def test_fill_bounded_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks[:-1]) + ([None] * MAX_SIZE)
                memory = Memory.from_blocks(blocks, start=blocks[0][0], endex=blocks[-1][0])
                pattern = b'<xyz>'
                endex = start + size

                memory.fill(start, endex, pattern)
                memory.validate()
                blocks_out = memory.to_blocks()

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                values[start:endex] = tiled
                values[:blocks[0][0]] = [None] * blocks[0][0]
                del values[blocks[-1][0]:]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

    def test_fill_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory.from_blocks(create_template_blocks())
        with pytest.raises(ValueError, match=match):
            memory.fill(pattern=b'')
        with pytest.raises(ValueError, match=match):
            memory.fill(pattern=[])
        with pytest.raises(ValueError, match=match):
            memory.fill(pattern=Memory())
        memory.fill(pattern=0)
        memory.validate()

    def test_fill_backup_doctest(self):
        pass  # no doctest

    def test_fill_restore_doctest(self):
        pass  # no doctest

    def test_flood_doctest(self):
        Memory = self.Memory

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        memory.flood(pattern=b'123')
        assert memory.to_blocks() == [[1, b'ABC12xyz']]

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        memory.flood(3, 7, b'123')
        assert memory.to_blocks() == [[1, b'ABC23xyz']]

    def test_flood_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                pattern = b'<xyz>'
                endex = start + size

                memory_backup = memory.__deepcopy__()
                backup = memory.flood_backup(start, endex)
                assert backup == list(memory_backup.gaps(start, endex))

                memory.flood(start, endex, pattern)
                memory.validate()
                blocks_out = memory.to_blocks()

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                for index in range(size):
                    if values[start + index] is None:
                        values[start + index] = tiled[index]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref

                memory.flood_restore(backup)
                memory.validate()
                assert memory == memory_backup

    def test_flood_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory.from_blocks(create_template_blocks())
        with pytest.raises(ValueError, match=match):
            memory.flood(pattern=b'')
        with pytest.raises(ValueError, match=match):
            memory.flood(pattern=[])
        with pytest.raises(ValueError, match=match):
            memory.flood(pattern=Memory())
        memory.flood(pattern=0)
        memory.validate()

    def test_flood_backup_doctest(self):
        pass  # no doctest

    def test_flood_restore_doctest(self):
        pass  # no doctest

    def test_keys_doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert list(memory.keys()) == []
        assert list(memory.keys(endex=8)) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert list(memory.keys(3, 8)) == [3, 4, 5, 6, 7]
        assert list(islice(memory.keys(3, ...), 7)) == [3, 4, 5, 6, 7, 8, 9]

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        assert list(memory.keys()) == [1, 2, 3, 4, 5, 6, 7, 8]
        assert list(memory.keys(endex=8)) == [1, 2, 3, 4, 5, 6, 7]
        assert list(memory.keys(3, 8)) == [3, 4, 5, 6, 7]
        assert list(islice(memory.keys(3, ...), 7)) == [3, 4, 5, 6, 7, 8, 9]

    def test_keys_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                endex = start + size
                memory = Memory()
                keys_out = list(islice(memory.keys(start, ...), size))
                keys_ref = list(range(start, endex))
                assert keys_out == keys_ref

    def test_values_doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert list(memory.values(endex=8)) == [None, None, None, None, None, None, None, None]
        assert list(memory.values(3, 8)) == [None, None, None, None, None]
        assert list(islice(memory.values(3, ...), 7)) == [None, None, None, None, None, None, None]

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        assert list(memory.values()) == [65, 66, 67, None, None, 120, 121, 122]
        assert list(memory.values(3, 8)) == [67, None, None, 120, 121]
        assert list(islice(memory.values(3, ...), 7)) == [67, None, None, 120, 121, 122, None]

    def test_values_empty_bruteforce(self):
        Memory = self.Memory
        for size in range(MAX_SIZE):
            blocks = []
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            values_out = list(memory.values(0, size))
            values_ref = list(islice(values, size))
            assert values_out == values_ref

    def test_values_halfbounded_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)
        values_out = list(islice(memory.values(None, memory.endex), len(memory)))
        values_ref = list(islice(values, memory.start, memory.endex))
        assert values_out == values_ref

    def test_values_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                iterator = memory.values(start, ...)
                values_out = []
                for _ in range(size):
                    values_out.append(next(iterator))

                values_ref = list(islice(values, start, endex))
                assert values_out == values_ref

    def test_values_pattern_template(self):
        Memory = self.Memory
        pattern = b'0123456789ABCDEF'
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                values_ref = list(islice(values, start, endex))
                for index, value in enumerate(values_ref):
                    if value is None:
                        values_ref[index] = pattern[index & 15]

                values_out = memory.values(start, endex, pattern=pattern)
                values_out = list(islice(values_out, size))
                assert values_out == values_ref

                if start == blocks[0][0]:
                    if endex == blocks[-1][0] + len(blocks[-1][1]):
                        values_out = list(islice(memory.values(pattern=pattern), size))
                        assert values_out == values_ref

                    values_out = list(memory.values(endex=endex, pattern=pattern))
                    assert values_out == values_ref

    def test_values_pattern_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory.from_blocks(create_template_blocks())
        with pytest.raises(ValueError, match=match):
            list(islice(memory.values(pattern=b''), MAX_SIZE))
        with pytest.raises(ValueError, match=match):
            list(islice(memory.values(pattern=[]), MAX_SIZE))
        with pytest.raises(ValueError, match=match):
            list(islice(memory.values(pattern=Memory()), MAX_SIZE))
        list(islice(memory.values(pattern=0), MAX_SIZE))

    def test_rvalues_doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert list(memory.rvalues(endex=8)) == [None, None, None, None, None, None, None, None]
        assert list(memory.rvalues(3, 8)) == [None, None, None, None, None]
        assert list(islice(memory.rvalues(..., 8), 7)) == [None, None, None, None, None, None, None]
        assert list(memory.rvalues(3, 8, b'ABCD')) == [65, 68, 67, 66, 65]

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        assert list(memory.rvalues()) == [122, 121, 120, None, None, 67, 66, 65]
        assert list(memory.rvalues(3, 8)) == [121, 120, None, None, 67]
        assert list(islice(memory.rvalues(..., 8), 7)) == [121, 120, None, None, 67, 66, 65]
        assert list(memory.rvalues(3, 8, b'0123')) == [121, 120, 50, 49, 67]

    def test_rvalues_empty_bruteforce(self):
        Memory = self.Memory
        for size in range(MAX_SIZE):
            blocks = []
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory.from_blocks(blocks)

            rvalues_out = list(memory.rvalues(0, size))
            rvalues_ref = list(islice(values, size))[::-1]
            assert rvalues_out == rvalues_ref

    def test_rvalues_halfunbounded_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)
        rvalues_out = list(islice(memory.rvalues(memory.start, None), len(memory)))
        rvalues_ref = list(islice(values, memory.start, memory.endex))[::-1]
        assert rvalues_out == rvalues_ref

    def test_rvalues_template(self):
        Memory = self.Memory
        for endex in range(MAX_START):
            for size in range(MAX_SIZE if self.ADDR_NEG else endex + 1):
                start = endex - size
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)

                iterator = memory.rvalues(..., endex)
                rvalues_out = []
                for _ in range(size):
                    rvalues_out.append(next(iterator))

                if start < 0:
                    rvalues_ref = list(islice(values, 0, endex))[::-1]
                    rvalues_ref += [None] * -start
                else:
                    rvalues_ref = list(islice(values, start, endex))[::-1]
                assert rvalues_out == rvalues_ref

    def test_rvalues_pattern_template(self):
        Memory = self.Memory
        pattern = b'0123456789ABCDEF'
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                endex = start + size

                rvalues_ref = list(islice(values, start, endex))
                for index, value in enumerate(rvalues_ref):
                    if value is None:
                        rvalues_ref[index] = pattern[index & 15]
                rvalues_ref.reverse()

                rvalues_out = memory.rvalues(start, endex, pattern=pattern)
                rvalues_out = list(islice(rvalues_out, size))
                assert rvalues_out == rvalues_ref

                if start == blocks[0][0]:
                    if endex == blocks[-1][0] + len(blocks[-1][1]):
                        rvalues_out = list(islice(memory.rvalues(pattern=pattern), size))
                        assert rvalues_out == rvalues_ref

                    rvalues_out = list(memory.rvalues(endex=endex, pattern=pattern))
                    assert rvalues_out == rvalues_ref

    def test_rvalues_pattern_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory.from_blocks(create_template_blocks())
        with pytest.raises(ValueError, match=match):
            list(islice(memory.rvalues(pattern=b''), MAX_SIZE))
        with pytest.raises(ValueError, match=match):
            list(islice(memory.rvalues(pattern=[]), MAX_SIZE))
        with pytest.raises(ValueError, match=match):
            list(islice(memory.rvalues(pattern=Memory()), MAX_SIZE))
        list(islice(memory.rvalues(pattern=0), MAX_SIZE))

    def test_items_doctest(self):
        Memory = self.Memory

        from itertools import islice
        memory = Memory()
        values_out = list(memory.items(endex=8))
        values_ref = [(0, None), (1, None), (2, None), (3, None), (4, None), (5, None), (6, None), (7, None)]
        assert values_out == values_ref
        values_out = list(memory.items(3, 8))
        values_ref = [(3, None), (4, None), (5, None), (6, None), (7, None)]
        assert values_out == values_ref
        values_out = list(islice(memory.items(3, ...), 7))
        values_ref = [(3, None), (4, None), (5, None), (6, None), (7, None), (8, None), (9, None)]
        assert values_out == values_ref

        memory = Memory.from_blocks([[1, b'ABC'], [6, b'xyz']])
        values_out = list(memory.items())
        values_ref = [(1, 65), (2, 66), (3, 67), (4, None), (5, None), (6, 120), (7, 121), (8, 122)]
        assert values_out == values_ref
        values_out = list(memory.items(3, 8))
        values_ref = [(3, 67), (4, None), (5, None), (6, 120), (7, 121)]
        assert values_out == values_ref
        values_out = list(islice(memory.items(3, ...), 7))
        values_ref = [(3, 67), (4, None), (5, None), (6, 120), (7, 121), (8, 122), (9, None)]
        assert values_out == values_ref

    def test_items_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)

                items_out = list(islice(memory.items(start, ...), size))

                values_ref = values[start:(start + len(items_out))]
                keys_ref = list(range(start, start + size))
                items_ref = list(zip(keys_ref, values_ref))
                assert items_out == items_ref

    def test_blocks_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])

        ans_out = [[s, bytes(d)] for s, d in memory.blocks()]
        ans_ref = [[1, b'AB'], [5, b'x'], [7, b'123']]
        assert ans_out == ans_ref

        ans_out = [[s, bytes(d)] for s, d in memory.blocks(2, 9)]
        ans_ref = [[2, b'B'], [5, b'x'], [7, b'12']]
        assert ans_out == ans_ref

        ans_out = [[s, bytes(d)] for s, d in memory.blocks(3, 5)]
        ans_ref = []
        assert ans_out == ans_ref

    def test_intervals_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
        assert list(memory.intervals()) == [(1, 3), (5, 6), (7, 10)]
        assert list(memory.intervals(2, 9)) == [(2, 3), (5, 6), (7, 9)]
        assert list(memory.intervals(3, 5)) == []

    def test_intervals(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for endex in range(start, MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)
                intervals_ref = values_to_intervals(values, start, endex)
                intervals_out = list(memory.intervals(start, endex))
                assert intervals_out == intervals_ref

    def test_intervals_unbounded(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)
        intervals_ref = values_to_intervals(values)
        intervals_out = list(memory.intervals())
        assert intervals_out == intervals_ref

    def test_intervals_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)
        intervals_ref = values_to_intervals(values)
        intervals_out = list(memory.intervals())
        assert intervals_out == intervals_ref

    def test_gaps_doctest(self):
        Memory = self.Memory
        memory = Memory.from_blocks([[1, b'AB'], [5, b'x'], [7, b'123']])
        assert list(memory.gaps()) == [(None, 1), (3, 5), (6, 7), (10, None)]
        assert list(memory.gaps(0, 11)) == [(0, 1), (3, 5), (6, 7), (10, 11)]
        assert list(memory.gaps(*memory.span)) == [(3, 5), (6, 7)]
        assert list(memory.gaps(2, 6)) == [(3, 5)]

    def test_gaps(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for endex in range(start, MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory.from_blocks(blocks)

                gaps_ref = values_to_gaps(values, start, endex)
                gaps_out = list(memory.gaps(start, endex))
                assert gaps_out == gaps_ref

    def test_gaps_unbounded(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        gaps_ref = values_to_gaps(values)
        gaps_out = list(memory.gaps())
        assert gaps_out == gaps_ref

    def test_gaps_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        gaps_ref = values_to_gaps(values)
        gaps_out = list(memory.gaps())
        assert gaps_out == gaps_ref

    def test_equal_span_doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert memory.equal_span(0) == (None, None, None)

        memory = Memory.from_blocks([[0, b'ABBBC'], [7, b'CCD']])
        assert memory.equal_span(2) == (1, 4, 66)
        assert memory.equal_span(4) == (4, 5, 67)
        assert memory.equal_span(5) == (5, 7, None)
        assert memory.equal_span(10) == (10, None, None)

    def test_equal_span_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for address in range(MAX_START):
            start, endex, value = memory.equal_span(address)
            span = values_to_equal_span(values, address)

            if values[address] is None:
                assert value is None
                assert (start, endex) == span
            else:
                assert value is not None
                assert (start, endex) == span

    def test_equal_span_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)

        for address in range(MAX_START):
            start, endex, value = memory.equal_span(address)
            span = values_to_equal_span(values, address)

            assert value is None
            assert (start, endex) == span

    def test_block_span_doctest(self):
        Memory = self.Memory

        memory = Memory()
        assert memory.block_span(0) == (None, None, None)

        memory = Memory.from_blocks([[0, b'ABBBC'], [7, b'CCD']])
        assert memory.block_span(2) == (0, 5, 66)
        assert memory.block_span(4) == (0, 5, 67)
        assert memory.block_span(5) == (5, 7, None)
        assert memory.block_span(10) == (10, None, None)

    def test_block_span(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)
        intervals = set(values_to_intervals(values))
        gaps = set(values_to_gaps(values, bound=False))

        for address in range(MAX_START):
            start, endex, value = memory.block_span(address)

            if values[address] is None:
                assert value is None
                assert (start, endex) in gaps
            else:
                assert value is not None
                assert (start, endex) in intervals

    def test_block_span_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory.from_blocks(blocks)
        gaps = set(values_to_gaps(values, bound=False))

        for address in range(MAX_START):
            start, endex, value = memory.block_span(address)

            assert value is None
            assert (start, endex) in gaps


class BaseBytearraySuite:

    bytesparse: Any = None  # replace by subclassing 'bytesparse'

    def test___init___empty(self):
        bytesparse = self.bytesparse
        memory = bytesparse()
        assert memory.span == (0, 0)
        assert memory.content_parts == 0
        assert memory == b''

    def test___init___source_buffer(self):
        bytesparse = self.bytesparse
        for size in range(MAX_SIZE):
            buffer = bytes(range(size))
            memory = bytesparse(buffer)
            assert memory.span == (0, size)
            assert memory.content_parts == (1 if size else 0)
            assert memory == buffer

            view = memoryview(buffer)
            memory = bytesparse(view)
            assert memory.span == (0, size)
            assert memory.content_parts == (1 if size else 0)
            assert memory == view

    def test___init___source_int(self):
        bytesparse = self.bytesparse
        for size in range(MAX_SIZE):
            memory = bytesparse(size)
            assert memory.span == (0, size)
            assert memory.content_parts == (1 if size else 0)
            assert memory == b'\0' * size

    def test___init___source_iterable(self):
        bytesparse = self.bytesparse
        for size in range(MAX_SIZE):
            values = list(range(size))
            memory = bytesparse(values)
            assert memory.span == (0, size)
            assert memory.content_parts == (1 if size else 0)
            assert memory == values

    def test___init___source_str(self):
        bytesparse = self.bytesparse

        text = 'Hello, World!'
        with pytest.raises(TypeError, match='string argument without an encoding'):
            bytesparse(text)

        encoding = 'ascii'
        memory = bytesparse(text, encoding)
        data = text.encode(encoding)
        assert memory.span == (0, len(data))
        assert memory.content_parts == 1
        assert memory == data

        text = 'a\xF0b\xF1cd\xF2efg\xF3hijk\xF4lmn\xF5op\xF6q\xF7r'
        with pytest.raises(UnicodeError, match='ordinal not in range'):
            bytesparse(text, encoding)

        errors = 'strict'
        with pytest.raises(UnicodeError, match='ordinal not in range'):
            bytesparse(text, encoding, errors)

        errors = 'ignore'
        memory = bytesparse(text, encoding, errors)
        data = text.encode(encoding, errors)
        assert memory.span == (0, len(data))
        assert memory.content_parts == 1
        assert memory == data

        errors = 'replace'
        memory = bytesparse(text, encoding, errors)
        data = text.encode(encoding, errors)
        assert memory.span == (0, len(data))
        assert memory.content_parts == 1
        assert memory == data

    def test___init___source_corner(self):
        bytesparse = self.bytesparse

        memory = bytesparse(b'ABCDEF', start=3, endex=2)
        assert not memory
        assert memory.span == (3, 3)
        assert memory.trim_span == (3, 3)

        memory = bytesparse(b'ABCDEF', start=None, endex=2)
        assert memory == b'AB'
        assert memory.span == (0, 2)
        assert memory.trim_span == (None, 2)

    def test__rectify_address(self):
        bytesparse = self.bytesparse
        memory = bytesparse.from_blocks(create_template_blocks())
        endex_ = memory.endex

        for offset in range(-1, -len(memory), -1):
            assert memory._rectify_address(offset) == endex_ + offset
            assert memory[offset] == memory[endex_ + offset]

    def test__rectify_address_overflow(self):
        bytesparse = self.bytesparse
        memory = bytesparse.from_blocks(create_template_blocks())
        endex_ = memory.endex
        assert memory._rectify_address(-endex_) == 0

        with pytest.raises(IndexError, match='index out of range'):
            memory._rectify_address(-endex_ - 1)

    def test__rectify_span(self):
        bytesparse = self.bytesparse
        memory = bytesparse.from_blocks(create_template_blocks())
        start_ = memory.start
        endex_ = memory.endex

        for start in range(-1, -len(memory), -1):
            assert memory._rectify_span(start, None) == ((endex_ + start), None)
            assert memory[start:] == memory[(endex_ + start):]

        for endex in range(-1, -len(memory), -1):
            assert memory._rectify_span(None, endex) == (None, (endex_ + endex))
            assert memory[:endex] == memory[:(endex_ + endex)]

        for start in range(-1, -len(memory), -1):
            for endex in range(-1, -len(memory), -1):
                assert memory._rectify_span(start, endex) == ((endex_ + start), (endex_ + endex))
                assert memory[start:endex] == memory[(endex_ + start):(endex_ + endex)]

    def test__rectify_span_overflow(self):
        bytesparse = self.bytesparse
        memory = bytesparse.from_blocks(create_template_blocks())
        start_ = memory.start
        endex_ = memory.endex
        assert memory._rectify_span(-endex_, -endex_) == (0, 0)
        assert memory._rectify_span(-endex_ - 1, -endex_ - 1) == (0, 0)

    def test_from_blocks_negative(self):
        bytesparse = self.bytesparse
        with pytest.raises(ValueError, match='negative offseted start'):
            bytesparse.from_blocks([[5, b'ABC']], offset=-6)
        with pytest.raises(ValueError, match='negative start'):
            bytesparse.from_blocks([[5, b'ABC']], start=-1)
        with pytest.raises(ValueError, match='negative endex'):
            bytesparse.from_blocks([[5, b'ABC']], endex=-1)

    def test_from_bytes_negative(self):
        bytesparse = self.bytesparse
        with pytest.raises(ValueError, match='negative offset'):
            bytesparse.from_bytes(b'ABC', offset=-1)
        with pytest.raises(ValueError, match='negative start'):
            bytesparse.from_bytes(b'ABC', start=-1)
        with pytest.raises(ValueError, match='negative endex'):
            bytesparse.from_bytes(b'ABC', endex=-1)

    def test_from_memory_negative(self):
        bytesparse = self.bytesparse
        memory = bytesparse(b'ABC')
        with pytest.raises(ValueError, match='negative offset'):
            bytesparse.from_memory(memory, offset=-1)
        with pytest.raises(ValueError, match='negative start'):
            bytesparse.from_memory(memory, start=-1)
        with pytest.raises(ValueError, match='negative endex'):
            bytesparse.from_memory(memory, endex=-1)

    def test_shift_negative(self):
        bytesparse = self.bytesparse

        memory = bytesparse()
        memory.shift(-1)
        memory.shift_backup(-1)

        memory = bytesparse(b'ABC')
        memory.shift(5)
        memory.shift_backup(-3)

        memory = bytesparse(b'ABC')
        with pytest.raises(ValueError, match='negative offseted start'):
            memory.shift(-1)
        with pytest.raises(ValueError, match='negative offseted start'):
            memory.shift_backup(-1)

    def test_trim_endex_negative(self):
        bytesparse = self.bytesparse
        memory = bytesparse(b'ABC')
        with pytest.raises(ValueError, match='negative endex'):
            memory.trim_endex = -1

    def test_trim_span_negative(self):
        bytesparse = self.bytesparse
        memory = bytesparse(b'ABC')
        with pytest.raises(ValueError, match='negative start'):
            memory.trim_span = (-1, None)
        with pytest.raises(ValueError, match='negative endex'):
            memory.trim_span = (None, -1)

    def test_trim_start_negative(self):
        bytesparse = self.bytesparse
        memory = bytesparse(b'ABC')
        with pytest.raises(ValueError, match='negative start'):
            memory.trim_start = -1
