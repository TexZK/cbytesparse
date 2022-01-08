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

r"""Python wrapper around C (Cython) implementation."""

from typing import Optional

from bytesparse.base import Address
from bytesparse.base import ClosedInterval
from bytesparse.base import ImmutableMemory
from bytesparse.base import MutableMemory
from bytesparse.base import OpenInterval

# noinspection PyUnresolvedReferences
from .c import collapse_blocks
# noinspection PyUnresolvedReferences
from .c import Memory as _Memory


# Proper Python wrapper around the C (Cython) implementation.
class Memory(_Memory):

    @ImmutableMemory.content_endex.getter
    def content_endex(
        self,
    ) -> Address:
        return super().content_endex

    @ImmutableMemory.content_endin.getter
    def content_endin(
        self,
    ) -> Address:
        return super().content_endin

    @ImmutableMemory.content_parts.getter
    def content_parts(
        self,
    ) -> int:
        return super().content_parts

    @ImmutableMemory.content_size.getter
    def content_size(
        self,
    ) -> Address:
        return super().content_size

    @ImmutableMemory.content_span.getter
    def content_span(
        self,
    ) -> ClosedInterval:
        return super().content_span

    @ImmutableMemory.content_start.getter
    def content_start(
        self,
    ) -> Address:
        return super().content_start

    @ImmutableMemory.contiguous.getter
    def contiguous(
        self,
    ) -> bool:
        return super().contiguous

    @ImmutableMemory.endex.getter
    def endex(
        self,
    ) -> Address:
        return super().endex

    @ImmutableMemory.endin.getter
    def endin(
        self,
    ) -> Address:
        return super().endin

    @ImmutableMemory.span.getter
    def span(
        self,
    ) -> ClosedInterval:
        return super().span

    @ImmutableMemory.start.getter
    def start(
        self,
    ) -> Address:
        return super().start

    @ImmutableMemory.trim_endex.getter
    def trim_endex(
        self,
    ) -> Optional[Address]:
        return super().trim_endex

    @trim_endex.setter
    def trim_endex(
        self,
        trim_endex: Address,
    ) -> None:
        super().trim_endex = trim_endex

    #  FIXME: "trim_span" seems not to work...
    #
    # @ImmutableMemory.trim_span.getter
    # def trim_span(
    #     self,
    # ) -> OpenInterval:
    #     return super().trim_span
    #
    # @trim_span.setter
    # def trim_span(
    #     self,
    #     trim_span: OpenInterval,
    # ) -> None:
    #     super().trim_span = trim_span

    @ImmutableMemory.trim_start.getter
    def trim_start(
        self,
    ) -> Optional[Address]:
        return super().trim_start

    @trim_start.setter
    def trim_start(
        self,
        trim_start: Address,
    ) -> None:
        super().trim_start = trim_start
