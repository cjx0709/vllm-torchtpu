# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process-wide `_scheduling_group_id` counter shared by every kernel."""

import itertools
import threading

# Source of `_scheduling_group_id`s for chunked TensorCore/SparseCore
# pipelines (e.g. `deepseek_v4/streamindex_topk`). One counter for the
# process, because the ids are global to a compiled program and a model traces
# such a kernel once per layer: two layers sharing an id would be ONE group to
# XLA, and the legalizer would either annotate everything on the path between
# them or fail the compile. A per-model copy of a kernel must take its ids from
# here too, for the same reason.
#
# The ids end up in the HLO, so they are part of the compile cache key. The
# compiler restarts the counter for every program it builds (see
# `reset_scheduling_group_ids`): otherwise the ids, and with them the key,
# depend on which programs this process traced before, and two ranks building
# the same bucket in a different order miss each other's cached executable.
_scheduling_group_ids = itertools.count(1)
_scheduling_group_ids_lock = threading.Lock()


def reserve_scheduling_group_ids(count: int) -> int:
    """Reserves `count` scheduling group ids nobody else in this process gets.

    Args:
      count: Number of ids wanted; a pipeline over `n` chunks uses `n - 1` of them
        (chunk 0's TensorCore compute and the last chunk's SparseCore compute are
        not annotated).

    Returns:
      The first id; the caller owns `[first, first + count)`.
    """
    if count < 1:
        raise ValueError(f"count ({count}) must be positive.")
    with _scheduling_group_ids_lock:
        first = next(_scheduling_group_ids)
        for _ in range(count - 1):
            next(_scheduling_group_ids)
    return first


def reset_scheduling_group_ids() -> None:
    """Restarts the ids at 1, for the start of a new compiled program.

    Ids only need to be unique within one program, so the next program can
    reuse them; resetting makes a program's ids, and so its cache key, the same
    whatever this process compiled before.
    """
    global _scheduling_group_ids
    with _scheduling_group_ids_lock:
        _scheduling_group_ids = itertools.count(1)
