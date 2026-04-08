"""
Labels for node_size experiments: filenames use letters (exp_A_1.jsonl) but plots
use only initial / mutation depth ranges (see notes.txt).
"""

from __future__ import annotations

# (init_lo, init_hi, mut_lo, mut_hi) per filename letter
NODE_SIZE_SORT_KEY: dict[str, tuple[int, int, int, int]] = {
    "A": (2, 6, 0, 1),
    "B": (1, 4, 0, 1),
    "C": (1, 4, 1, 5),
    "D": (3, 8, 0, 1),
    "E": (3, 8, 1, 5),
    "F": (2, 6, 1, 5),
    "G": (2, 6, 0, 3),
    "H": (1, 4, 0, 3),
    "I": (3, 8, 0, 3),
}


def node_depth_label(letter: str) -> str:
    """Two-line label: init range / mut range (no letter)."""
    L = letter.upper()
    t = NODE_SIZE_SORT_KEY.get(L)
    if not t:
        return L
    i0, i1, m0, m1 = t
    return f"init {i0}–{i1}\nmut {m0}–{m1}"


def node_depth_sort_tuple(letter: str) -> tuple[int, int, int, int, str]:
    """Sort key: initial depth, then mutation depth, then letter."""
    L = letter.upper()
    t = NODE_SIZE_SORT_KEY.get(L)
    if not t:
        return (999, 999, 999, 999, L)
    return (*t, L)


def node_depth_sort_mutation_first(letter: str) -> tuple[int, int, int, int, str]:
    """Sort key: mutation depth range first, then initial depth, then letter (for boxplots)."""
    L = letter.upper()
    t = NODE_SIZE_SORT_KEY.get(L)
    if not t:
        return (999, 999, 999, 999, L)
    i0, i1, m0, m1 = t
    return (m0, m1, i0, i1, L)
