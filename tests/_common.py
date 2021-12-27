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

from itertools import islice
from typing import Any
from typing import List
from typing import Optional

import pytest

from cbytesparse import Address
from cbytesparse import BlockList
from cbytesparse import OpenInterval
from cbytesparse import Value

MAX_START: Address = 20
MAX_SIZE: Address = 24
MAX_TIMES: int = 5
BITMASK_SIZE: int = 16


def create_template_blocks() -> BlockList:
    return [
        [2, bytearray(b'234')],
        [8, bytearray(b'89A')],
        [12, bytearray(b'C')],
        [14, bytearray(b'EF')],
        [18, bytearray(b'I')],
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

    Memory: type = None  # replace by subclassing
    ADDR_NEG: bool = True

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
            Memory(blocks=[[1, b'1'], [0, b'0']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[2, b'2'], [0, b'0']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[3, b'345'], [0, b'012']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[7, b'789'], [0, b'012']])
        if self.ADDR_NEG:
            with pytest.raises(ValueError, match=match):
                Memory(blocks=[[0, b'0'], [-1, b'1']])
            with pytest.raises(ValueError, match=match):
                Memory(blocks=[[0, b'0'], [-2, b'2']])

    def test___init___multi(self):
        Memory = self.Memory
        match = r'only one of \[memory, data, blocks\] is allowed'
        with pytest.raises(ValueError, match=match):
            Memory(memory=Memory(), data=b'\0')
        with pytest.raises(ValueError, match=match):
            Memory(memory=Memory(), blocks=[])
        with pytest.raises(ValueError, match=match):
            Memory(data=b'\0', blocks=[])

    def test___init___offset_template(self):
        Memory = self.Memory
        for offset in range(-MAX_SIZE if self.ADDR_NEG else 0, MAX_SIZE):
            blocks_ref = create_template_blocks()
            for block in blocks_ref:
                block[0] += offset

            for copy in (False, True):
                memory = Memory(blocks=create_template_blocks(), offset=offset, copy=copy)
                blocks_out = memory._blocks
                assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test___init___null(self):
        Memory = self.Memory
        Memory(data=b'')

        match = r'invalid block data size'
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[0, b'0'], [5, b''], [9, b'9']])

    def test___init___interleaving(self):
        Memory = self.Memory
        match = r'invalid block interleaving'
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[0, b'0'], [1, b'1'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[1, b'1'], [2, b'2'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[0, b'012'], [3, b'345'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[1, b'1'], [0, b'0'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[2, b'2'], [0, b'0'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[3, b'345'], [0, b'012'], [15, b'F']])
        with pytest.raises(ValueError, match=match):
            Memory(blocks=[[7, b'789'], [0, b'012'], [15, b'F']])
        if self.ADDR_NEG:
            with pytest.raises(ValueError, match=match):
                Memory(blocks=[[0, b'0'], [-1, b'1'], [15, b'F']])
            with pytest.raises(ValueError, match=match):
                Memory(blocks=[[0, b'0'], [-2, b'2'], [15, b'F']])

    def test___init___offset(self):
        Memory = self.Memory
        data = b'5'
        blocks = [[0, b'0'], [5, data], [9, b'9']]
        offset = 123

        memory = Memory(data=data, offset=offset)
        sm = memory._blocks[0][0]
        assert sm == offset, (sm, offset)

        memory = Memory(blocks=blocks, offset=offset)
        for (sm, _), (sb, _) in zip(memory._blocks, blocks):
            assert sm == sb + offset, (sm, sb, offset)

        memory = Memory(blocks=blocks)
        memory2 = Memory(memory=memory, offset=offset)
        for (sm1, _), (sm2, _) in zip(memory._blocks, memory2._blocks):
            assert sm2 == sm1 + offset, (sm2, sm1, offset)

    def test___repr__(self):
        Memory = self.Memory
        start, endex = 0, 0
        memory = Memory()
        repr_out = repr(memory)
        repr_ref = f'<Memory[0x{start}:0x{endex}]@0x{id(memory):X}>'
        assert repr_out == repr_ref, (repr_out, repr_ref)

        start, endex = 1, 9
        memory = Memory(start=start, endex=endex)
        repr_out = repr(memory)
        repr_ref = f'<Memory[0x{start}:0x{endex}]@0x{id(memory):X}>'
        assert repr_out == repr_ref, (repr_out, repr_ref)

        start, endex = 3, 6
        memory = Memory(data=b'abc', offset=3)
        repr_out = repr(memory)
        repr_ref = f'<Memory[0x{start}:0x{endex}]@0x{id(memory):X}>'
        assert repr_out == repr_ref, (repr_out, repr_ref)

    def test___str__(self):
        Memory = self.Memory
        memory = Memory()
        str_out = str(memory)
        str_ref = str([])
        assert str_out == str_ref, (str_out, str_ref)

        memory = Memory(start=3, endex=9)
        str_out = str(memory)
        str_ref = str([])
        assert str_out == str_ref, (str_out, str_ref)

        data = b'abc'
        memory = Memory(data=data, offset=3)
        str_out = str(memory)
        str_ref = str([[3, bytearray(data)]])
        assert str_out == str_ref, (str_out, str_ref)

        data = b'abc' * 1000
        memory = Memory(data=data, offset=3)
        str_out = str(memory)
        str_ref = repr(memory)
        assert str_out == str_ref, (str_out, str_ref)

    def test___bool__(self):
        Memory = self.Memory
        assert Memory(memory=Memory(data=b'\0'))
        assert Memory(data=b'\0')
        assert Memory(blocks=[[0, b'\0']])

        assert not Memory()
        assert not Memory(memory=Memory())
        assert not Memory(data=b'')
        assert not Memory(blocks=[])

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
        assert memory != Memory(data=bytes(1))

    def test___eq___memory(self):
        Memory = self.Memory
        memory1 = Memory(blocks=create_template_blocks())
        memory2 = Memory(blocks=create_template_blocks())
        assert memory1 == memory2, (memory1._blocks == memory2._blocks)

        memory1.append(0)
        assert memory1 != memory2, (memory1._blocks == memory2._blocks)

        memory1.pop()
        memory1.shift(1)
        assert memory1 != memory2, (memory1._blocks == memory2._blocks)

    def test___eq___bytelike(self):
        Memory = self.Memory
        data = bytes(range(256))
        memory = Memory(data=data, offset=256)
        assert memory == data
        assert memory != data + b'\0'

        memory.shift(1)
        assert memory == data
        assert memory != data + b'\0'

    def test___eq___generator(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=create_template_blocks())
        values = blocks_to_values(blocks)[memory.start:memory.endex]

        assert memory == iter(values), (values,)
        assert memory != reversed(values), (values[::-1],)
        assert memory != iter(values[:-1]), (values[:-1],)
        assert memory != iter(values + [0]), (values + [0],)

    def test___eq___bitmask(self):
        Memory = self.Memory
        bitmask_size = BITMASK_SIZE
        index_base = (1 << (bitmask_size - 1)) | 1
        for bitmask_index in range(1, bitmask_size - 1):
            index = index_base | bitmask_index
            values = create_bitmask_values(index, bitmask_size)
            blocks = values_to_blocks(values)
            memory = Memory(blocks=blocks)
            assert memory == values, (values,)

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
        memory = Memory(blocks=blocks)
        assert all(x == y for x, y in zip(memory, values)), (values,)

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
        memory = Memory(blocks=blocks)
        assert all(x == y for x, y in zip(reversed(memory), reversed(values))), (values[::-1],)

    def test___add___template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        blocks2 = create_hello_world_blocks()

        block = blocks1[-1]
        offset = block[0] + len(block[1])
        for block in blocks2:
            block[0] += offset

        memory1 = Memory(blocks=blocks1)
        memory2 = Memory(blocks=blocks2)
        memory3 = memory1 + memory2
        blocks_out = memory3._blocks

        values = blocks_to_values(blocks1)
        values += blocks_to_values(blocks2)
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test___iadd___template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        blocks2 = create_hello_world_blocks()

        block = blocks1[-1]
        offset = block[0] + len(block[1])
        for block in blocks2:
            block[0] += offset

        memory1 = Memory(blocks=blocks1)
        memory2 = Memory(blocks=blocks2)
        memory1 += memory2
        blocks_out = memory1._blocks

        values = blocks_to_values(blocks1)
        values += blocks_to_values(blocks2)
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test___mul___template(self):
        Memory = self.Memory
        for times in range(-1, MAX_TIMES):
            blocks = create_template_blocks()

            memory1 = Memory(blocks=blocks)
            memory2 = memory1 * times
            blocks_out = memory2._blocks

            values = blocks_to_values(blocks)
            offset = blocks[0][0]
            values = ([None] * offset) + (values[offset:] * times)
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test___imul___template(self):
        Memory = self.Memory
        for times in range(-1, MAX_TIMES):
            blocks = create_template_blocks()

            memory = Memory(blocks=blocks)
            memory *= times
            blocks_out = memory._blocks

            values = blocks_to_values(blocks)
            offset = blocks[0][0]
            values = ([None] * offset) + (values[offset:] * times)
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test___len___empty(self):
        Memory = self.Memory
        assert len(Memory()) == 0
        assert len(Memory(memory=Memory())) == 0
        assert len(Memory(data=b'')) == 0
        assert len(Memory(blocks=[])) == 0

    def test___len___template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(1, MAX_SIZE):
                endex = start + size
                data = bytes(range(size))
                memory = Memory(data=data, offset=start)

                assert memory.start == start, (memory.start, start)
                assert memory.endex == endex, (memory.endex, endex)
                assert len(memory) == size, (len(memory), size)

    def test___len___bounds(self):
        Memory = self.Memory

        assert len(Memory(start=1, endex=9)) == 9 - 1
        assert len(Memory(memory=Memory(), start=1, endex=9)) == 9 - 1
        assert len(Memory(data=b'', start=1, endex=9)) == 9 - 1
        assert len(Memory(blocks=[], start=1, endex=9)) == 9 - 1

        assert len(Memory(start=1)) == 0
        assert len(Memory(memory=Memory(), start=1)) == 0
        assert len(Memory(data=b'', start=1)) == 0
        assert len(Memory(blocks=[], start=1)) == 0

        assert len(Memory(endex=9)) == 9
        assert len(Memory(memory=Memory(), endex=9)) == 9
        assert len(Memory(data=b'', endex=9)) == 9
        assert len(Memory(blocks=[], endex=9)) == 9

    def test___len__(self):
        Memory = self.Memory

        memory = Memory(blocks=create_hello_world_blocks())
        assert len(memory) == memory.endex - memory.start, (len(memory), memory.endex, memory.start)
        assert len(memory) == (16 - 2), (len(memory), memory.endex, memory.start)

        memory = Memory(blocks=create_template_blocks())
        assert len(memory) == memory.endex - memory.start, (len(memory), memory.endex, memory.start)
        assert len(memory) == (19 - 2), (len(memory), memory.endex, memory.start)

    def test_ofind(self):
        Memory = self.Memory

        memory = Memory(blocks=create_hello_world_blocks())

        assert memory.ofind(b'X') is None
        assert memory.ofind(b'W') == 10
        assert memory.ofind(b'o') == 6
        assert memory.ofind(b'l') == 4

    def test_rofind(self):
        Memory = self.Memory

        memory = Memory(blocks=create_hello_world_blocks())

        assert memory.rofind(b'X') is None
        assert memory.rofind(b'W') == 10
        assert memory.rofind(b'o') == 11
        assert memory.rofind(b'l') == 13

    def test_find(self):
        Memory = self.Memory

        memory = Memory(blocks=create_hello_world_blocks())

        assert memory.find(b'X') == -1
        assert memory.find(b'W') == 10
        assert memory.find(b'o') == 6
        assert memory.find(b'l') == 4

    def test_rfind(self):
        Memory = self.Memory

        memory = Memory(blocks=create_hello_world_blocks())

        assert memory.rfind(b'X') == -1
        assert memory.rfind(b'W') == 10
        assert memory.rfind(b'o') == 11
        assert memory.rfind(b'l') == 13

    def test_index(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory(blocks=blocks)
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
                            assert index, (index,)

                        with pytest.raises(ValueError, match=match):
                            index = memory.index(bytes([c]), start, endex)
                            assert index, (index,)
                    else:
                        index = memory.index(c, start, endex)
                        assert index == expected, (index, expected)

                        index = memory.index(bytes([c]), start, endex)
                        assert index == expected, (index, expected)

        for c in chars:
            expected = None
            for i in range(len(values)):
                if values[i] == c:
                    expected = i
                    break

            if expected is None:
                with pytest.raises(ValueError, match=match):
                    index = memory.index(c)
                    assert index, (index,)

                with pytest.raises(ValueError, match=match):
                    index = memory.index(bytes([c]))
                    assert index, (index,)
            else:
                index = memory.index(c)
                assert index == expected, (index, expected)

                index = memory.index(bytes([c]))
                assert index == expected, (index, expected)

    def test_rindex(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory(blocks=blocks)
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
                            assert index, (index,)

                        with pytest.raises(ValueError, match=match):
                            index = memory.rindex(bytes([c]), start, endex)
                            assert index, (index,)
                    else:
                        index = memory.rindex(c, start, endex)
                        assert index == expected, (index, expected)

                        index = memory.rindex(bytes([c]), start, endex)
                        assert index == expected, (index, expected)

        for c in chars:
            expected = None
            for i in reversed(range(len(values))):
                if values[i] == c:
                    expected = i
                    break

            if expected is None:
                with pytest.raises(ValueError, match=match):
                    index = memory.rindex(c)
                    assert index, (index,)

                with pytest.raises(ValueError, match=match):
                    index = memory.rindex(bytes([c]))
                    assert index, (index,)
            else:
                index = memory.rindex(c)
                assert index == expected, (index, expected)

                index = memory.rindex(bytes([c]))
                assert index == expected, (index, expected)

    def test___contains___empty_bruteforce(self):
        Memory = self.Memory
        memory = Memory()

        checks = [i not in memory for i in range(256)]
        assert all(checks), (checks,)

        checks = [bytes([i]) not in memory for i in range(256)]
        assert all(checks), (checks,)

    def test___contains__(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory(blocks=blocks)
        values = blocks_to_values(blocks)
        chars = (set(values) - {None}) | {b'X'[0]}

        for c in chars:
            expected = c in values

            check = c in memory
            assert check == expected, (check, expected)

            check = bytes([c]) in memory
            assert check == expected, (check, expected)

    def test_count_empty_bruteforce(self):
        Memory = self.Memory
        memory = Memory()

        checks = [not memory.count(i) for i in range(256)]
        assert all(checks), (checks,)

        checks = [not memory.count(bytes([i])) for i in range(256)]
        assert all(checks), (checks,)

        for start in range(8):
            for endex in range(start, 8):
                checks = [not memory.count(i, start, endex) for i in range(8)]
                assert all(checks), (checks,)

                checks = [not memory.count(bytes([i]), start, endex) for i in range(8)]
                assert all(checks), (checks,)

    def test_count(self):
        Memory = self.Memory
        blocks = create_hello_world_blocks()
        memory = Memory(blocks=blocks)
        values = blocks_to_values(blocks)
        chars = (set(values) - {None}) | {b'X'[0]}

        for start in range(MAX_START):
            for endex in range(start, MAX_START):
                for c in chars:
                    expected = values[start:endex].count(c)

                    count = memory.count(c, start, endex)
                    assert count == expected, (count, expected)

                    count = memory.count(bytes([c]), start, endex)
                    assert count == expected, (count, expected)

    def test___getitem___single_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for start in range(MAX_START):
            value = memory[start]
            assert value == values[start], (start, value, values[start])

    def test___getitem___contiguous(self):
        Memory = self.Memory
        blocks = [[3, b'abc']]
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for start in range(3, 6):
            for endex in range(start, 6):
                data_out = list(memory[start:endex].to_memoryview())
                data_ref = values[start:endex]
                assert data_out == data_ref, (start, endex, data_out, data_ref)

    def test___getitem___contiguous_step(self):
        Memory = self.Memory
        data = bytes(range(ord('a'), ord('z') + 1))
        blocks = [[3, data]]
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        stop = 3 + len(data)
        for start in range(3, stop):
            for endex in range(start, stop):
                for step in range(1, 4):
                    data_out = list(memory[start:endex:step].to_memoryview())
                    data_ref = values[start:endex:step]
                    assert data_out == data_ref, (start, endex, step, data_out, data_ref)

                for step in range(-1, 1):
                    blocks_out = memory[start:endex:step]._blocks
                    blocks_ref = []
                    assert blocks_out == blocks_ref, (start, endex, step, blocks_out, blocks_ref)

    def test___getitem___non_contiguous(self):
        Memory = self.Memory
        data = b'abc'
        memory = Memory(data=data, offset=5)
        dot = b'.'

        assert memory[:] == data
        assert memory[::dot] == data
        assert memory[:5] == b''
        assert memory[:5:dot] == b''
        assert memory[8:] == b''
        assert memory[8::dot] == b''
        assert memory[::dot] == data

        extracted = memory[1:9]
        assert extracted._blocks == [[5, data]]
        assert extracted.span == (1, 9)

        extracted = memory[1:7]
        assert extracted._blocks == [[5, data[:7 - 5]]]
        assert extracted.span == (1, 7)

        extracted = memory[7:9]
        assert extracted._blocks == [[7, data[7 - 5:]]]
        assert extracted.span == (7, 9)

        memory = Memory(data=data, offset=5)
        extracted = memory[:]
        assert extracted._blocks == [[5, data]]
        assert extracted.span == (5, 8)

        memory = Memory(data=data, offset=5, start=2, endex=22)
        extracted = memory[:]
        assert extracted._blocks == [[5, data]]
        assert extracted.span == (2, 22)

        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        extracted = memory[:]
        assert extracted._blocks == blocks
        assert extracted.span == memory.span

    def test___getitem___pattern(self):
        Memory = self.Memory
        dot = ord('.')
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                values_out = list(memory[start:endex:b'.'])

                for index in range(start):
                    values[index] = None
                for index in range(start, endex):
                    if values[index] is None:
                        values[index] = dot
                values_ref = values[start:endex]
                assert values_out == values_ref, (start, size, endex, values_out, values_ref)

    def test___setitem___single_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            memory[start] = start
            blocks_out = memory._blocks

            values[start] = start
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test___setitem___replace_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                data = bytearray(range(size))
                endex = start + size

                memory[start:endex] = data
                blocks_out = memory._blocks

                values[start:endex] = list(data)
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___setitem___shrink_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size + 1, MAX_SIZE):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory = Memory(blocks=blocks)
                    memory[start:endex] = data
                    blocks_out = memory._blocks

                    values_ref = values[:]
                    values_ref[start:endex] = list(data)
                    blocks_ref = values_to_blocks(values_ref)

                    assert blocks_out == blocks_ref, (start, target_size, endex, source_size,
                                                      blocks_out, blocks_ref)

    def test___setitem___shrink_unbounded_start_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size + 1, MAX_SIZE):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory = Memory(blocks=blocks)
                    memory[:endex] = data
                    blocks_out = memory._blocks

                    values_ref = values[:]
                    values_ref[blocks[0][0]:endex] = list(data)
                    blocks_ref = values_to_blocks(values_ref)

                    assert blocks_out == blocks_ref, (start, target_size, endex, source_size,
                                                      blocks_out, blocks_ref)

    def test___setitem___shrink_unbounded_end_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size + 1, MAX_SIZE):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory = Memory(blocks=blocks)
                    memory[start:] = data
                    blocks_out = memory._blocks

                    values_ref = values[:]
                    values_ref[start:] = list(data)
                    blocks_ref = values_to_blocks(values_ref)

                    assert blocks_out == blocks_ref, (start, target_size, endex, source_size,
                                                      blocks_out, blocks_ref)

    def test___setitem___enlarge_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for source_size in range(MAX_SIZE):
                for target_size in range(source_size):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory(blocks=blocks)
                    data = bytearray(range(source_size))
                    endex = start + target_size

                    memory[start:endex] = data
                    blocks_out = memory._blocks

                    values[start:endex] = list(data)
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref, (start, target_size, endex, source_size,
                                                      blocks_out, blocks_ref)

    def test___setitem___none_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                memory[start:endex] = None
                blocks_out = memory._blocks

                values[start:endex] = [None] * (endex - start)
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___setitem___none_step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(1, MAX_TIMES):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory(blocks=blocks)
                    endex = start + size

                    memory[start:endex:step] = None
                    blocks_out = memory._blocks

                    values[start:endex:step] = [None] * ((size + step - 1) // step)
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___setitem___value_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                memory[start:endex] = start
                blocks_out = memory._blocks

                values[start:endex] = [start] * (endex - start)
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___setitem___step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(1, MAX_TIMES):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory(blocks=blocks)
                    endex = start + size
                    data = bytes((size + step - 1) // step)

                    memory[start:endex:step] = data
                    blocks_out = memory._blocks

                    values[start:endex:step] = data
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___setitem___misstep_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(-1, 1):
                    blocks = create_template_blocks()
                    memory = Memory(blocks=blocks)
                    endex = start + size

                    memory[start:endex:step] = b''  # unreachable
                    blocks_out = memory._blocks
                    blocks_ref = blocks

                    assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___setitem___size_template(self):
        Memory = self.Memory
        match = r'attempt to assign bytes of size \d+ to extended slice of size \d+'

        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(2, MAX_TIMES):
                    blocks = create_template_blocks()
                    memory = Memory(blocks=blocks)
                    endex = start + size
                    data = bytes(MAX_SIZE)

                    with pytest.raises(ValueError, match=match):
                        memory[start:endex:step] = data

    def test___delitem___single_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            del memory[start]
            blocks_out = memory._blocks

            del values[start]
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test___delitem___step_negative_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for step in range(-1, 1):
                blocks = create_template_blocks()
                memory = Memory(blocks=blocks)

                del memory[start::step]
                blocks_out = memory._blocks
                assert blocks_out == blocks, (start, step)

    def test___delitem___template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                del memory[start:endex]
                blocks_out = memory._blocks

                del values[start:endex]
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___delitem___step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                for step in range(1, MAX_TIMES):
                    blocks = create_template_blocks()
                    values = blocks_to_values(blocks, MAX_SIZE)
                    memory = Memory(blocks=blocks)
                    endex = start + size

                    del memory[start:endex:step]
                    blocks_out = memory._blocks

                    del values[start:endex:step]
                    blocks_ref = values_to_blocks(values)

                    assert blocks_out == blocks_ref, (start, size, endex, step, blocks_out, blocks_ref)

    def test___delitem___unbounded_start_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                del memory[:endex]
                blocks_out = memory._blocks

                del values[blocks[0][0]:endex]
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test___delitem___unbounded_endex_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                del memory[start:]
                blocks_out = memory._blocks

                del values[start:]
                blocks_ref = values_to_blocks(values)

                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test_append_empty_int(self):
        Memory = self.Memory
        memory = Memory()
        memory.append(ord('X'))
        values = [ord('X')]

        blocks_out = memory._blocks
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_append_empty_byte(self):
        Memory = self.Memory
        memory = Memory()
        memory.append(b'X')
        values = [ord('X')]
        blocks_out = memory._blocks
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_append_empty_multi(self):
        Memory = self.Memory
        memory = Memory()
        with pytest.raises(ValueError, match='expecting single item'):
            memory.append(b'XY')

    def test_append(self):
        Memory = self.Memory
        memory = Memory(blocks=create_template_blocks())
        blocks_ref = create_template_blocks()
        blocks_ref[-1][1].append(ord('X'))
        memory.append(ord('X'))
        blocks_out = memory._blocks
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_extend_empty(self):
        Memory = self.Memory
        memory = Memory()
        blocks_ref = []
        memory.extend(b'')
        blocks_out = memory._blocks
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

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
            memory = Memory(blocks=blocks)
            values = blocks_to_values(blocks)
            data = bytes(range(size))

            memory.extend(data)
            blocks_out = memory._blocks

            values.extend(data)
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_extend_template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        blocks2 = create_hello_world_blocks()

        block = blocks1[-1]
        offset = block[0] + len(block[1])
        for block in blocks2:
            block[0] += offset

        memory1 = Memory(blocks=blocks1)
        memory2 = Memory(blocks=blocks2)
        memory1.extend(memory2)
        blocks_out = memory1._blocks

        values = blocks_to_values(blocks1)
        values.extend(blocks_to_values(blocks2))
        blocks_ref = values_to_blocks(values)

        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_pop_empty(self):
        Memory = self.Memory
        memory = Memory()
        value_out = memory.pop()
        assert value_out is None, (value_out,)

    def test_pop(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks)
        memory = Memory(blocks=blocks)

        value_out = memory.pop()
        value_ref = values.pop()
        assert value_out == value_ref, (value_out, value_ref)

        blocks_out = memory._blocks
        blocks_ref = values_to_blocks(values)
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_pop_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            value_out = memory.pop(start)
            value_ref = values.pop(start)
            assert value_out == value_ref, (value_out, value_ref)

            blocks_out = memory._blocks
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test___bytes__(self):
        Memory = self.Memory
        memory = Memory()
        data = memory.__bytes__()
        assert data == b'', (data,)

        memory = Memory(data=b'xyz', offset=5)
        data = memory.__bytes__()
        assert data == b'xyz', (data,)

        blocks = [[5, b'xyz']]
        memory = Memory(blocks=blocks, copy=False)
        data = memory.__bytes__()
        assert data == blocks[0][1], (data, blocks[0][1])

    def test_to_bytes(self):
        Memory = self.Memory
        memory = Memory()
        data = memory.to_bytes()
        assert data == b'', (data,)

        memory = Memory(data=b'xyz', offset=5)
        data = memory.to_bytes()
        assert data == b'xyz', (data,)

        blocks = [[5, b'xyz']]
        memory = Memory(blocks=blocks, copy=False)
        data = memory.to_bytes()
        assert data == blocks[0][1], (data, blocks[0][1])

    def test_to_bytearray(self):
        Memory = self.Memory
        memory = Memory()
        data = memory.to_bytearray()
        assert data == b'', (data,)

        memory = Memory(data=b'xyz', offset=5)
        data = memory.to_bytearray()
        assert data == b'xyz', (data,)

        blocks = [[5, b'xyz']]
        memory = Memory(blocks=blocks, copy=False)
        data = memory.to_bytearray()
        assert data == blocks[0][1], (data, blocks[0][1])

    def test_to_memoryview(self):
        Memory = self.Memory
        memory = Memory()
        data = bytes(memory.to_memoryview())
        assert data == b'', (data,)

        memory = Memory(data=b'xyz', offset=5)
        data = bytes(memory.to_memoryview())
        assert data == b'xyz', (data,)

        blocks = [[5, b'xyz']]
        memory = Memory(blocks=blocks, copy=False)
        data = bytes(memory.to_memoryview())
        assert data == blocks[0][1], (data, blocks[0][1])

    def test___copy___empty(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = memory1.__copy__()
        assert memory1.span == memory2.span, (memory1.span, memory2.span)
        assert memory1.trim_span == memory2.trim_span, (memory1.trim_span, memory2.trim_span)
        assert memory1.content_span == memory2.content_span, (memory1.content_span, memory2.content_span)
        checks = [b1[1] == b2[1] for b1, b2 in zip(memory1._blocks, memory2._blocks)]
        assert all(checks), (checks,)

    def test___copy___template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory1 = Memory(blocks=blocks, copy=False)
        memory2 = memory1.__copy__()
        assert memory1.span == memory2.span, (memory1.span, memory2.span)
        assert memory1.trim_span == memory2.trim_span, (memory1.trim_span, memory2.trim_span)
        assert memory1.content_span == memory2.content_span, (memory1.content_span, memory2.content_span)
        checks = [b1[1] == b2[1] for b1, b2 in zip(memory1._blocks, memory2._blocks)]
        assert all(checks), (checks,)

    def test___deepcopy___empty(self):
        Memory = self.Memory
        memory1 = Memory()
        memory2 = memory1.__copy__()
        assert memory1.span == memory2.span, (memory1.span, memory2.span)
        assert memory1.trim_span == memory2.trim_span, (memory1.trim_span, memory2.trim_span)
        assert memory1.content_span == memory2.content_span, (memory1.content_span, memory2.content_span)
        checks = [b1[1] is not b2[1] for b1, b2 in zip(memory1._blocks, memory2._blocks)]
        assert all(checks), (checks,)
        blocks1, blocks2 = memory1._blocks, memory2._blocks
        assert blocks1 == blocks2, (blocks1, blocks2)

    def test___deepcopy___template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory1 = Memory(blocks=blocks, copy=False)
        memory2 = memory1.__deepcopy__()
        assert memory1.span == memory2.span, (memory1.span, memory2.span)
        assert memory1.trim_span == memory2.trim_span, (memory1.trim_span, memory2.trim_span)
        assert memory1.content_span == memory2.content_span, (memory1.content_span, memory2.content_span)
        checks = [b1[1] is not b2[1] for b1, b2 in zip(memory1._blocks, memory2._blocks)]
        assert all(checks), (checks,)
        blocks1, blocks2 = memory1._blocks, memory2._blocks
        assert blocks1 == blocks2, (blocks1, blocks2)

    def test_contiguous(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.contiguous

        memory = Memory(start=1, endex=9)
        assert not memory.contiguous

        memory = Memory(data=b'xyz', offset=3)
        assert memory.contiguous

        memory = Memory(data=b'xyz', offset=3, start=3, endex=6)
        assert memory.contiguous

        memory = Memory(data=b'xyz', offset=3, start=1)
        assert not memory.contiguous

        memory = Memory(data=b'xyz', offset=3, endex=9)
        assert not memory.contiguous

        memory = Memory(data=b'xyz', offset=3, start=1, endex=9)
        assert not memory.contiguous

        memory = Memory(data=b'xyz')
        assert memory.contiguous

        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        assert not memory.contiguous

        memory.trim_endex = MAX_SIZE
        assert not memory.contiguous

        memory.trim_endex = None
        memory.trim_start = 0
        assert not memory.contiguous

    def test_trim_start_bytes(self):
        Memory = self.Memory
        data = bytes(range(8))
        memory = Memory(data=data)

        for offset in range(8):
            memory.trim_start = offset
            assert memory.content_start == offset, (memory.content_start, offset)
            assert memory.content_endex == 8, (memory.content_endex,)

        for offset in reversed(range(8)):
            memory.trim_start = offset
            assert memory.content_start == 7, (memory.content_start,)
            assert memory.content_endex == 8, (memory.content_endex,)

        memory.trim_start = 8
        assert memory.content_start == 8, (memory.content_start,)
        assert memory.content_endex == 8, (memory.content_endex,)

        for offset in range(8):
            memory.trim_start = offset
            assert memory.content_start == offset, (memory.content_start, offset)
            assert memory.content_endex == offset, (memory.content_endex, offset)

        for offset in reversed(range(8)):
            memory.trim_start = offset
            assert memory.content_start == offset, (memory.content_start, offset)
            assert memory.content_endex == offset, (memory.content_endex, offset)

        memory.trim_endex = None
        assert memory.trim_endex is None, (memory.trim_endex,)
        memory.trim_start = 9
        assert memory.trim_start == 9, (memory.trim_start,)
        assert memory.trim_endex is None, (memory.trim_endex,)

        memory.trim_start = 1
        memory.trim_endex = 5
        assert memory.trim_start == 1, (memory.trim_start,)
        assert memory.trim_endex == 5, (memory.trim_endex,)
        memory.trim_start = 9
        assert memory.trim_start == 9, (memory.trim_start,)
        assert memory.trim_endex == 9, (memory.trim_endex,)

    def test_trim_endex_bytes(self):
        Memory = self.Memory
        data = bytes(range(8))
        memory = Memory(data=data)

        for offset in range(8, 0, -1):
            memory.trim_endex = offset
            assert memory.content_start == 0, (memory.content_start,)
            assert memory.content_endex == offset, (memory.content_endex, offset)

        for offset in range(1, 8):
            memory.trim_endex = offset
            assert memory.content_start == 0, (memory.content_start,)
            assert memory.content_endex == 1, (memory.content_endex,)

        memory.trim_endex = 0
        assert memory.content_start == 0, (memory.content_start,)
        assert memory.content_endex == 0, (memory.content_endex,)

        for offset in range(8, 0, -1):
            memory.trim_endex = offset
            assert memory.content_start == 0, (memory.content_start,)
            assert memory.content_endex == 0, (memory.content_endex,)

        for offset in range(1, 8):
            memory.trim_endex = offset
            assert memory.content_start == 0, (memory.content_start,)
            assert memory.content_endex == 0, (memory.content_endex,)

        memory.trim_start = None
        assert memory.trim_start is None, (memory.trim_start,)
        memory.trim_endex = 9
        assert memory.trim_start is None, (memory.trim_start,)
        assert memory.trim_endex == 9, (memory.trim_endex,)

        memory.trim_start = 5
        memory.trim_endex = 9
        assert memory.trim_start == 5, (memory.trim_start,)
        assert memory.trim_endex == 9, (memory.trim_endex,)
        memory.trim_endex = 1
        assert memory.trim_start == 1, (memory.trim_start,)
        assert memory.trim_endex == 1, (memory.trim_endex,)

    def test_trim_start_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for offset in range(1, MAX_SIZE):
            memory.trim_start = offset
            blocks_out = memory._blocks
            values[offset - 1] = None
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

        blocks_ref = values_to_blocks(values)
        for offset in reversed(range(MAX_SIZE)):
            memory.trim_start = offset
            blocks_out = memory._blocks
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_trim_endex_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for offset in reversed(range(MAX_SIZE)):
            memory.trim_endex = offset
            blocks_out = memory._blocks
            values[offset] = None
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

        blocks_ref = values_to_blocks(values)
        for offset in range(MAX_SIZE):
            memory.trim_endex = offset
            blocks_out = memory._blocks
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_trim_span(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.trim_span == (None, None), (memory.trim_span,)
        memory.trim_span = (1, 9)
        assert memory.trim_span == (1, 9), (memory.trim_span,)
        memory.trim_span = (5, 5)
        assert memory.trim_span == (5, 5), (memory.trim_span,)

    def test_start_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.start == 0, (memory.start,)
        assert memory.content_start == 0, (memory.content_start,)
        assert memory.trim_start is None, (memory.trim_start,)

        start = 123
        memory = Memory(start=start)
        assert memory.start == start, (memory.start, start)
        assert memory.content_start == start, (memory.content_start, start)

    def test_start(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        assert memory.start == blocks[0][0], (memory.start, blocks[0][0])
        assert memory.content_start == blocks[0][0], (memory.content_start, blocks[0][0])
        assert memory.trim_start is None, (memory.trim_start,)

    def test_endex_empty(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.endex == 0, (memory.endex,)
        assert memory.content_endex == 0, (memory.content_endex,)
        assert memory.trim_endex is None, (memory.trim_endex,)

    def test_endex(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        block_start, block_data = blocks[-1]
        block_endex = block_start + len(block_data)
        assert memory.endex == block_endex, (memory.endex, block_endex)
        assert memory.content_endex == block_endex, (memory.content_endex, block_endex)
        assert memory.trim_endex is None, (memory.trim_endex,)

    def test_span(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.span == (0, 0), (memory.span,)

        memory.trim_span = (1, 9)
        assert memory.span == (1, 9), (memory.span,)

        memory.trim_span = (9, 1)
        assert memory.span == (9, 9), (memory.span,)

        memory.trim_span = (None, None)
        assert memory.span == (0, 0), (memory.span,)

        memory.write(5, b'xyz')
        assert memory.span == (5, 8), (memory.span,)

        memory.trim_span = (1, 9)
        assert memory.span == (1, 9), (memory.span,)

        memory.trim_span = (None, None)
        assert memory.span == (5, 8), (memory.span,)

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
        memory = Memory(blocks=blocks)
        block_start, block_data = blocks[-1]
        block_endin = block_start + len(block_data) - 1
        assert memory.endin == block_endin
        assert memory.content_endin == block_endin

        endex = 123
        memory = Memory(blocks=blocks, endex=endex)
        assert memory.endin == endex - 1
        assert memory.content_endin == block_endin

        start = 1
        memory = Memory(blocks=blocks, start=start, endex=endex)
        assert memory.endin == endex - 1
        assert memory.content_endin == block_endin

    def test_content_start(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_start == memory.start, (memory.content_start, memory.start)
        assert memory.content_start == 0, (memory.content_start,)

        memory.write(5, b'xyz')
        assert memory.content_start == memory.start, (memory.content_start, memory.start)
        assert memory.content_start == 5, (memory.content_start,)

    def test_content_start_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        assert memory.content_start == memory.start, (memory.content_start, memory.start)
        assert memory.content_start == blocks[0][0], (memory.content_start, blocks[0][0])

        memory.trim_start = 0
        assert memory.content_start > memory.start, (memory.content_start, memory.start)
        assert memory.content_start == blocks[0][0], (memory.content_start, blocks[0][0])

    def test_content_endex(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_endex == memory.endex, (memory.content_endex, memory.endex)
        assert memory.content_endex == 0, (memory.content_endex,)

        memory.write(5, b'xyz')
        assert memory.content_endex == memory.endex, (memory.content_endex, memory.endex)
        assert memory.content_endex == 8, (memory.content_endex,)

    def test_content_endex_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        endex = blocks[-1][0] + len(blocks[-1][1])
        assert memory.content_endex == memory.endex, (memory.content_endex, memory.endex)
        assert memory.content_endex == endex, (memory.content_endex, endex)

        memory.trim_endex = MAX_SIZE
        assert memory.content_endex < memory.endex, (memory.content_endex, memory.endex)
        assert memory.content_endex == endex, (memory.content_endex, endex)

    def test_content_span(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_span == (0, 0), (memory.content_span,)

        memory.write(5, b'xyz')
        assert memory.content_span == (5, 8), (memory.content_span,)

        memory.trim_span = (1, 9)
        assert memory.content_span == (5, 8), (memory.content_span,)

        memory.trim_span = (None, None)
        assert memory.content_span == (5, 8), (memory.content_span,)

    def test_content_span_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        start = blocks[0][0]
        endex = blocks[-1][0] + len(blocks[-1][1])
        assert memory.content_span == (start, endex), (memory.content_span, start, endex)

        memory.trim_span = (0, MAX_SIZE)
        assert memory.content_span == (start, endex), (memory.content_span, start, endex)

        memory.trim_span = (None, None)
        assert memory.content_span == (start, endex), (memory.content_span, start, endex)

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
        memory = Memory(blocks=blocks)
        endin = blocks[-1][0] + len(blocks[-1][1]) - 1
        assert memory.content_endin == memory.endin
        assert memory.content_endin == endin

        memory.trim_endex = MAX_SIZE
        assert memory.content_endin < memory.endin
        assert memory.content_endin == endin

    def test_content_size(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_size == len(memory), (memory.content_size, len(memory))
        assert memory.content_size == 0, (memory.content_size,)

        memory.write(5, b'xyz')
        assert memory.content_size == len(memory), (memory.content_size, len(memory))
        assert memory.content_size == 3, (memory.content_size,)

        memory.trim_span = (1, 9)
        assert memory.content_size == 3, (memory.content_size,)

    def test_content_parts(self):
        Memory = self.Memory
        memory = Memory()
        assert memory.content_parts == 0, (memory.content_parts,)

        memory.write(5, b'xyz')
        assert memory.content_parts == 1, (memory.content_parts,)

    def test_content_parts_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)
        assert memory.content_parts == len(blocks), (memory.content_parts, len(blocks))

    def test_validate(self):
        Memory = self.Memory
        memory = Memory(validate=False)
        memory.validate()

    def test_bound(self):
        Memory = self.Memory
        memory = Memory(start=11, endex=44)

        bound = memory.bound(11, 44)
        assert bound == (11, 44), (bound,)

        bound = memory.bound(22, 33)
        assert bound == (22, 33), (bound,)

        bound = memory.bound(0, 99)
        assert bound == (11, 44), (bound,)

        bound = memory.bound(0, 0)
        assert bound == (11, 11), (bound,)

        bound = memory.bound(99, 99)
        assert bound == (44, 44), (bound,)

        bound = memory.bound(99, 0)
        assert bound == (44, 44), (bound,)

        bound = memory.bound(None, 44)
        assert bound == (11, 44), (bound,)

        bound = memory.bound(None, 33)
        assert bound == (11, 33), (bound,)

        bound = memory.bound(None, 99)
        assert bound == (11, 44), (bound,)

        bound = memory.bound(11, None)
        assert bound == (11, 44), (bound,)

        bound = memory.bound(22, None)
        assert bound == (22, 44), (bound,)

        bound = memory.bound(0, None)
        assert bound == (11, 44), (bound,)

    def test__block_index_at_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)

        blocks_index_ref: List[Any] = [None] * MAX_SIZE
        for block_index, (block_start, block_data) in enumerate(blocks):
            for offset in range(len(block_data)):
                blocks_index_ref[block_start + offset] = block_index

        blocks_index_out = [memory._block_index_at(address) for address in range(MAX_SIZE)]
        assert blocks_index_out == blocks_index_ref, (blocks_index_out, blocks_index_ref)

    def test__block_index_at_empty(self):
        Memory = self.Memory
        memory = Memory()
        blocks_index_out = [memory._block_index_at(address) for address in range(MAX_SIZE)]
        blocks_index_ref = [None] * MAX_SIZE
        assert blocks_index_out == blocks_index_ref, (blocks_index_out, blocks_index_ref)

    def test__block_index_start(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)

        blocks_index_ref = [len(blocks)] * MAX_SIZE
        for block_index in reversed(range(len(blocks))):
            block_start, block_data = blocks[block_index]
            block_endex = block_start + len(block_data)
            for offset in range(block_endex):
                blocks_index_ref[offset] = block_index

        blocks_index_out = [memory._block_index_start(address) for address in range(MAX_SIZE)]
        assert blocks_index_out == blocks_index_ref, (blocks_index_out, blocks_index_ref)

    def test__block_index_endex(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        memory = Memory(blocks=blocks)

        blocks_index_ref = [0] * MAX_SIZE
        for block_index in range(len(blocks)):
            block_start = blocks[block_index][0]
            block_index += 1
            for offset in range(block_start, MAX_SIZE):
                blocks_index_ref[offset] = block_index

        blocks_index_out = [memory._block_index_endex(address) for address in range(MAX_SIZE)]
        assert blocks_index_out == blocks_index_ref, (blocks_index_out, blocks_index_ref)

    def test_peek_template(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for address in range(MAX_START):
            value = memory.peek(address)
            assert value == values[address], (address, value, values[address])

    def test_poke_value_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            memory.poke(start, start)
            blocks_out = memory._blocks

            values[start] = start
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test_poke_single_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            memory.poke(start, bytes([start]))
            blocks_out = memory._blocks

            values[start] = start
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test_poke_multi_template(self):
        Memory = self.Memory
        match = r'expecting single item'

        for start in range(MAX_START):
            blocks = create_template_blocks()
            memory = Memory(blocks=blocks)

            with pytest.raises(ValueError, match=match):
                memory.poke(start, b'123')

    def test_poke_none_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            memory.poke(start, None)
            blocks_out = memory._blocks

            values[start] = None
            blocks_ref = values_to_blocks(values)

            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test_extract_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)

                for bound in (False, True):
                    extracted = memory.extract(start, start + size, bound=bound)
                    blocks_out = extracted._blocks
                    blocks_ref = values_to_blocks(values[start:(start + size)], start)
                    assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_extract_step_negative_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                memory = Memory(blocks=blocks)

                for step in range(-1, 1):
                    extracted = memory.extract(start, start + size, step=step)
                    blocks_out = extracted._blocks
                    blocks_ref = []
                    assert blocks_out == blocks_ref, (start, size, step, blocks_out, blocks_ref)

    def test_extract_step_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks)
                memory = Memory(blocks=blocks)
                endex = start + size

                for step in range(1, MAX_TIMES):
                    extracted = memory.extract(start, endex, step=step)
                    blocks_out = extracted._blocks
                    blocks_ref = values_to_blocks(values[start:endex:step], start)
                    assert blocks_out == blocks_ref, (start, size, step, blocks_out, blocks_ref)

    def test_extract_pattern_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                pattern = b'xyz'

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                for index in range(size):
                    if values[start + index] is None:
                        values[start + index] = tiled[index]

                extracted = memory.extract(start, start + size, pattern)
                blocks_out = extracted._blocks

                blocks_ref = values_to_blocks(values[start:(start + size)], start)
                assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_shift_empty_template(self):
        Memory = self.Memory
        for offset in range(-MAX_SIZE, MAX_SIZE):
            assert not Memory(offset=offset)

    def test_shift_template(self):
        Memory = self.Memory
        for offset in range(-MAX_SIZE if self.ADDR_NEG else 0, MAX_SIZE):
            blocks_ref = create_template_blocks()
            for block in blocks_ref:
                block[0] += offset

            memory = Memory(blocks=create_template_blocks())
            memory.shift(offset)
            blocks_out = memory._blocks
            assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_shift_backups_template(self):
        Memory = self.Memory
        start, endex = 2, MAX_SIZE
        for offset in range(-start, endex):
            values = blocks_to_values(create_template_blocks()) + ([None] * MAX_SIZE)
            if offset < 0:
                backups_ref = values_to_blocks(values[:(start - offset)])
                del values[:-offset]
                values[:start] = [None] * start
            else:
                backups_ref = values_to_blocks(values[(endex - offset):], (endex - offset))
                values[0:0] = [None] * offset
                del values[endex:]
            blocks_ref = values_to_blocks(values)
            backups_ref = [backups_ref] if backups_ref else []

            blocks = create_template_blocks()
            memory = Memory(blocks=blocks, start=start, endex=endex)
            backups_out = []
            memory.shift(offset, backups=backups_out)
            backups_out = [m._blocks for m in backups_out if m._blocks]
            blocks_out = memory._blocks
            assert blocks_out == blocks_ref, (offset, blocks_out, blocks_ref)
            assert backups_out == backups_ref, (offset, backups_out, backups_ref)

    def test_reserve_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)

                memory.reserve(start, size)
                blocks_out = memory._blocks

                values[start:start] = [None] * size
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_reserve_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks, endex=MAX_SIZE)

                backups = []
                memory.reserve(start, size, backups=backups)

                blocks_out = [m._blocks for m in backups if m]
                values[start:start] = [None] * size
                blocks_ref = values_to_blocks(values[MAX_SIZE:], MAX_SIZE - size)
                if blocks_ref:
                    blocks_ref = [blocks_ref]
                assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_insert_single(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            memory.insert(start, start)
            blocks_out = memory._blocks

            values[start:start] = [start]
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test_insert_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                data = bytes(range(ord('a'), ord('a') + size))

                memory.insert(start, data)
                blocks_out = memory._blocks

                values[start:start] = data
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, data, blocks_out, blocks_ref)

    def test_insert_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks, endex=MAX_SIZE)
                data = bytearray(range(ord('a'), ord('a') + size))

                backups = []
                memory.insert(start, Memory(data=data), backups=backups)

                offset = MAX_SIZE - size
                blocks_out = [m._blocks for m in backups if m._blocks]
                blocks_ref = values_to_blocks(values[offset:], offset)
                if blocks_ref:
                    blocks_ref = [blocks_ref]
                assert blocks_out == blocks_ref, (start, size, data, blocks_out, blocks_ref)

    def test_delete_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                memory.delete(start, endex)
                blocks_out = memory._blocks

                del values[start:endex]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test_delete_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                backups = []
                memory.delete(start, endex, backups=backups)

                blocks_out = [m._blocks for m in backups if m._blocks]
                blocks_ref = values_to_blocks(values[start:endex], start)
                if blocks_ref:
                    blocks_ref = [blocks_ref]
                assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_clear_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                memory.clear(start, endex)
                blocks_out = memory._blocks

                values[start:endex] = [None] * (endex - start)
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test_clear_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                backups = []
                memory.clear(start, endex, backups=backups)

                blocks_out = [m._blocks for m in backups if m._blocks]
                blocks_ref = values_to_blocks(values[start:endex], start)
                if blocks_ref:
                    blocks_ref = [blocks_ref]
                assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_crop_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                memory.crop(start, endex)
                blocks_out = memory._blocks

                values[:start] = [None] * start
                values[endex:] = [None] * (len(values) - endex)
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, endex, blocks_out, blocks_ref)

    def test_crop_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                backups = []
                memory.crop(start, endex, backups=backups)

                blocks_out = [m._blocks for m in backups if m._blocks]
                blocks_ref = [values_to_blocks(values[:start]),
                              values_to_blocks(values[endex:], endex)]
                blocks_ref = [b for b in blocks_ref if b]
                assert blocks_out == blocks_ref, (start, size, blocks_out, blocks_ref)

    def test_write_single(self):
        Memory = self.Memory
        for start in range(MAX_START):
            blocks = create_template_blocks()
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            memory.write(start, start)
            blocks_out = memory._blocks

            values[start] = start
            blocks_ref = values_to_blocks(values)
            assert blocks_out == blocks_ref, (start, blocks_out, blocks_ref)

    def test_write_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                data = bytearray(range(ord('a'), ord('a') + size))
                endex = start + len(data)

                memory.write(start, data)
                blocks_out = memory._blocks

                values[start:endex] = data
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, data, blocks_out, blocks_ref)

    def test_write_noclear_hello_over_template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        values1 = blocks_to_values(blocks1)
        memory1 = Memory(blocks=blocks1)

        blocks2 = create_hello_world_blocks()
        values2 = blocks_to_values(blocks2)
        memory2 = Memory(blocks=blocks2)

        backups = []
        memory1.write(0, memory2, clear=False, backups=backups)

        values_ref = values1[:]
        for block_start, block_data in blocks2:
            block_endex = block_start + len(block_data)
            values_ref[block_start:block_endex] = block_data

        values_out = blocks_to_values(memory1._blocks)
        assert values_out == values_ref, (values1, values2, values_out, values_ref)

    def test_write_clear_hello_over_template(self):
        Memory = self.Memory
        blocks1 = create_template_blocks()
        values1 = blocks_to_values(blocks1)
        memory1 = Memory(blocks=blocks1)

        blocks2 = create_hello_world_blocks()
        values2 = blocks_to_values(blocks2)
        memory2 = Memory(blocks=blocks2)

        backups = []
        memory1.write(0, memory2, clear=True, backups=backups)

        values_ref = values1[:]
        start, endex = memory2.start, memory2.endex
        values_ref[start:endex] = values2[start:endex]

        values_out = blocks_to_values(memory1._blocks)
        assert values_out == values_ref, (values1, values2, values_out, values_ref)

        blocks_out = [m._blocks for m in backups if m._blocks]
        blocks_ref = values_to_blocks(values1[start:endex], start)
        if blocks_ref:
            blocks_ref = [blocks_ref]
        assert blocks_out == blocks_ref, (blocks_out, blocks_ref)

    def test_write_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                data = bytearray(range(ord('a'), ord('a') + size))
                endex = start + len(data)

                backups = []
                memory.write(start, data, backups=backups)
                blocks_out = [m._blocks for m in backups if m._blocks]
                blocks_ref = values_to_blocks(values[start:endex], start)
                if blocks_ref:
                    blocks_ref = [blocks_ref]
                assert blocks_out == blocks_ref, (start, size, data, blocks_out, blocks_ref)

    def test_fill_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                pattern = b'<xyz>'
                endex = start + size

                memory.fill(start, endex, pattern)
                blocks_out = memory._blocks

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                values[start:endex] = tiled
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, pattern, blocks_out, blocks_ref)

    def test_fill_bounded_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks[:-1]) + ([None] * MAX_SIZE)
                memory = Memory(blocks=blocks, start=blocks[0][0], endex=blocks[-1][0])
                pattern = b'<xyz>'
                endex = start + size

                memory.fill(start, endex, pattern)
                blocks_out = memory._blocks

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                values[start:endex] = tiled
                values[:blocks[0][0]] = [None] * blocks[0][0]
                del values[blocks[-1][0]:]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, pattern, blocks_out, blocks_ref)

    def test_fill_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory(blocks=create_template_blocks())
        with pytest.raises(ValueError, match=match):
            memory.fill(pattern=b'')
        with pytest.raises(ValueError, match=match):
            memory.fill(pattern=[])
        with pytest.raises(ValueError, match=match):
            memory.fill(pattern=Memory())
        memory.fill(pattern=0)

    def test_fill_backups_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                pattern = b'<xyz>'
                endex = start + size

                backups = []
                memory.fill(start, endex, pattern, backups=backups)
                blocks_out = [m._blocks for m in backups if m._blocks]
                blocks_ref = values_to_blocks(values[start:endex], start)
                if blocks_ref:
                    blocks_ref = [blocks_ref]
                assert blocks_out == blocks_ref, (start, size, pattern, blocks_out, blocks_ref)

    def test_flood_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                pattern = b'<xyz>'
                endex = start + size

                memory.flood(start, endex, pattern)
                blocks_out = memory._blocks

                tiled = pattern * ((size + len(pattern)) // len(pattern))
                tiled = tiled[:size]
                for index in range(size):
                    if values[start + index] is None:
                        values[start + index] = tiled[index]
                blocks_ref = values_to_blocks(values)
                assert blocks_out == blocks_ref, (start, size, pattern, blocks_out, blocks_ref)

    def test_flood_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory(blocks=create_template_blocks())
        with pytest.raises(ValueError, match=match):
            memory.flood(pattern=b'')
        with pytest.raises(ValueError, match=match):
            memory.flood(pattern=[])
        with pytest.raises(ValueError, match=match):
            memory.flood(pattern=Memory())
        memory.flood(pattern=0)

    def test_flood_template_backups(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        gaps_ref = values_to_gaps(values, bound=True)
        memory = Memory(blocks=blocks)
        backups = []
        memory.flood(backups=backups)
        gaps_out = [(m.start, m.endex) for m in backups]
        assert gaps_out == gaps_ref, (gaps_out, gaps_ref)

    def test_keys_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                endex = start + size
                memory = Memory()
                keys_out = list(islice(memory.keys(start, ...), size))
                keys_ref = list(range(start, endex))
                assert keys_out == keys_ref, (start, size, keys_out, keys_ref)

    def test_values_empty_bruteforce(self):
        Memory = self.Memory
        for size in range(MAX_SIZE):
            blocks = []
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            values_out = list(memory.values(0, size))
            values_ref = list(islice(values, size))
            assert values_out == values_ref, (size, values_out, values_ref)

    def test_values_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                values_out = list(memory.values(start, endex))
                values_ref = list(islice(values, start, endex))
                assert values_out == values_ref, (start, size, values_out, values_ref)

    def test_values_pattern_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                endex = start + size

                values_ref = list(islice(values, start, endex))
                for index, value in enumerate(values_ref):
                    if value is None:
                        values_ref[index] = start

                values_out = memory.values(start, endex, pattern=start)
                values_out = list(islice(values_out, size))
                assert values_out == values_ref, (start, size, values_out, values_ref)

                values_out = memory.values(start, endex, pattern=start)
                values_out = list(islice(values_out, size))
                assert values_out == values_ref, (start, size, values_out, values_ref)

                if start == blocks[0][0]:
                    if endex == blocks[-1][0] + len(blocks[-1][1]):
                        values_out = list(islice(memory.values(pattern=start), size))
                        assert values_out == values_ref, (start, size, values_out, values_ref)

                    values_out = list(memory.values(endex=endex, pattern=start))
                    assert values_out == values_ref, (start, size, values_out, values_ref)

    def test_values_pattern_invalid_template(self):
        Memory = self.Memory
        match = r'non-empty pattern required'
        memory = Memory(blocks=create_template_blocks())
        with pytest.raises(ValueError, match=match):
            list(islice(memory.values(pattern=b''), MAX_SIZE))
        with pytest.raises(ValueError, match=match):
            list(islice(memory.values(pattern=[]), MAX_SIZE))
        with pytest.raises(ValueError, match=match):
            list(islice(memory.values(pattern=Memory()), MAX_SIZE))
        list(islice(memory.values(pattern=0), MAX_SIZE))

    def test_rvalues_empty_bruteforce(self):
        Memory = self.Memory
        for size in range(MAX_SIZE):
            blocks = []
            values = blocks_to_values(blocks, MAX_SIZE)
            memory = Memory(blocks=blocks)

            rvalues_out = list(memory.rvalues(0, size))
            rvalues_ref = list(islice(values, size))[::-1]
            assert rvalues_out == rvalues_ref, (size, rvalues_out, rvalues_ref)

    def test_rvalues_template(self):
        Memory = self.Memory
        for endex in range(MAX_START):
            for size in range(MAX_SIZE if self.ADDR_NEG else endex + 1):
                start = endex - size
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)

                rvalues_out = list(memory.rvalues(start, endex))
                if start < 0:
                    rvalues_ref = list(islice(values, 0, endex))[::-1]
                    rvalues_ref += [None] * -start
                else:
                    rvalues_ref = list(islice(values, start, endex))[::-1]
                assert rvalues_out == rvalues_ref, (endex, size, rvalues_out, rvalues_ref)

    def test_items_template(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for size in range(MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)

                items_out = list(islice(memory.items(start, ...), size))

                values_ref = values[start:(start + len(items_out))]
                keys_ref = list(range(start, start + size))
                items_ref = list(zip(keys_ref, values_ref))
                assert items_out == items_ref, (start, size, items_out, items_ref)

    def test_intervals(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for endex in range(start, MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)
                intervals_ref = values_to_intervals(values, start, endex)
                intervals_out = list(memory.intervals(start, endex))
                assert intervals_out == intervals_ref, (intervals_out, intervals_ref)

    def test_intervals_unbounded(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)
        intervals_ref = values_to_intervals(values)
        intervals_out = list(memory.intervals())
        assert intervals_out == intervals_ref, (intervals_out, intervals_ref)

    def test_intervals_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)
        intervals_ref = values_to_intervals(values)
        intervals_out = list(memory.intervals())
        assert intervals_out == intervals_ref, (intervals_out, intervals_ref)

    def test_gaps(self):
        Memory = self.Memory
        for start in range(MAX_START):
            for endex in range(start, MAX_SIZE):
                blocks = create_template_blocks()
                values = blocks_to_values(blocks, MAX_SIZE)
                memory = Memory(blocks=blocks)

                for bound in (False, True):
                    gaps_ref = values_to_gaps(values, start, endex, bound)
                    gaps_out = list(memory.gaps(start, endex, bound))
                    assert gaps_out == gaps_ref, (gaps_out, gaps_ref)

    def test_gaps_unbounded(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for bound in (False, True):
            gaps_ref = values_to_gaps(values, bound=bound)
            gaps_out = list(memory.gaps(bound=bound))
            assert gaps_out == gaps_ref, (gaps_out, gaps_ref)

    def test_gaps_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for bound in (False, True):
            gaps_ref = values_to_gaps(values, bound=bound)
            gaps_out = list(memory.gaps(bound=bound))
            assert gaps_out == gaps_ref, (gaps_out, gaps_ref)

    def test_equal_span(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for address in range(MAX_START):
            start, endex, value = memory.equal_span(address)
            span = values_to_equal_span(values, address)

            if values[address] is None:
                assert value is None, (value,)
                assert (start, endex) == span, (start, endex, span)
            else:
                assert value is not None, (value,)
                assert (start, endex) == span, (start, endex, span)

    def test_equal_span_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)

        for address in range(MAX_START):
            start, endex, value = memory.equal_span(address)
            span = values_to_equal_span(values, address)

            if values[address] is None:
                assert value is None, (value,)
                assert (start, endex) == span, (start, endex, span)
            else:
                assert value is not None, (value,)
                assert (start, endex) == span, (start, endex, span)

    def test_block_span(self):
        Memory = self.Memory
        blocks = create_template_blocks()
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)
        intervals = set(values_to_intervals(values))
        gaps = set(values_to_gaps(values, bound=False))

        for address in range(MAX_START):
            start, endex, value = memory.block_span(address)

            if values[address] is None:
                assert value is None, (value,)
                assert (start, endex) in gaps, (start, endex, gaps)
            else:
                assert value is not None, (value,)
                assert (start, endex) in intervals, (start, endex, intervals)

    def test_block_span_empty(self):
        Memory = self.Memory
        blocks = []
        values = blocks_to_values(blocks, MAX_SIZE)
        memory = Memory(blocks=blocks)
        intervals = set(values_to_intervals(values))
        gaps = set(values_to_gaps(values, bound=False))

        for address in range(MAX_START):
            start, endex, value = memory.block_span(address)

            if values[address] is None:
                assert value is None, (value,)
                assert (start, endex) in gaps, (start, endex, gaps)
            else:
                assert value is not None, (value,)
                assert (start, endex) in intervals, (start, endex, intervals)
