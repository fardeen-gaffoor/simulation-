"""The tag36h11 code table: the single fact module 4 cannot get wrong.

A wrong code renders a tag that cannot be decoded, or - far worse - one that
decodes to a *different id* and teleports the rover.  ``tagcodes`` is a copy of
the upstream table, so these tests re-derive the properties that make the family
usable rather than trusting the transcription.
"""

from __future__ import annotations

import numpy as np
import pytest

from rover_perception import tagcodes


def test_family_metadata():
    member = tagcodes.family("tag36h11")
    assert member.nbits == 36
    assert member.width_at_border == 8
    assert member.total_width == 10
    assert member.hamming == 11
    assert member.data_width == 6, "the black square has a 1-module black border"
    assert member.ncodes == 587
    assert tagcodes.MAX_ID == 586


def test_unknown_family_lists_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        tagcodes.family("tag25h9")
    assert "tag36h11" in str(excinfo.value)


def test_table_size_matches_the_declared_count():
    member = tagcodes.family("tag36h11")
    assert len(member.codes) == 587
    assert len(set(member.codes)) == 587, "the table must not repeat a code"


def test_reference_codes():
    """The first three ids, as published by AprilRobotics."""
    assert tagcodes.code_for(0) == 0xD7E00984B
    assert tagcodes.code_for(1) == 0xDDA664CA7
    assert tagcodes.code_for(2) == 0xDC4A1C821


def test_code_for_rejects_an_impossible_id():
    with pytest.raises(ValueError) as excinfo:
        tagcodes.code_for(587)
    assert "587" in str(excinfo.value)
    with pytest.raises(ValueError):
        tagcodes.code_for(-1)


def test_minimum_hamming_distance_is_the_family_headroom():
    """Every pair of codes differs by >= 11 bits.

    This is what makes accepting a 1-bit correction safe (``detection.hamming_max``)
    and what guarantees a mis-decode cannot produce a *valid* wrong id.
    """
    codes = np.asarray(tagcodes.family("tag36h11").codes, dtype=np.uint64)
    worst = 64
    worst_pair = (-1, -1)
    # pairwise over 587 codes: 171k xor+popcount, cheap and fully deterministic
    for i in range(codes.size):
        distances = np.bitwise_count(np.bitwise_xor(codes[i], codes[i + 1 :]))
        if distances.size:
            index = int(np.argmin(distances))
            if int(distances[index]) < worst:
                worst = int(distances[index])
                worst_pair = (i, i + 1 + index)
    assert worst == 11, f"minimum Hamming distance is {worst} (codes {worst_pair})"
    assert tagcodes.family("tag36h11").hamming == worst, "declared headroom must match reality"


def test_no_code_is_zero_or_all_ones():
    codes = tagcodes.family("tag36h11").codes
    assert 0 not in codes
    assert (1 << 36) - 1 not in codes


def test_bits_for_is_thirty_six_bits_inside_the_black_square():
    member = tagcodes.family("tag36h11")
    for tag_id in (0, 1, 2, 100, 586):
        bits = tagcodes.bits_for(tag_id)
        assert len(bits) == 36
        assert all(value in (0, 1) for _x, _y, value in bits)
        # payload modules live in the 1..6 window of the 8-module black square
        assert all(1 <= x <= 6 and 1 <= y <= 6 for x, y, _value in bits)
        assert len({(x, y) for x, y, _v in bits}) == 36, "each payload bit has its own module"


def test_bits_for_round_trips_the_code():
    """MSB-first packing: re-assembling the bits must give the code back."""
    member = tagcodes.family("tag36h11")
    for tag_id in (0, 3, 17, 250, 586):
        code = tagcodes.code_for(tag_id)
        packed = 0
        for _x, _y, value in tagcodes.bits_for(tag_id):
            packed = (packed << 1) | value
        assert packed == code
    assert member.nbits == 36


def test_bit_layout_matches_the_reference_origin():
    """Bit 0 is the payload's most significant bit, at module (1, 1)."""
    x, y, value = tagcodes.bits_for(0)[0]
    assert (x, y) == (1, 1)
    assert value == (tagcodes.code_for(0) >> 35) & 1 == 1
    x, y, value = tagcodes.bits_for(1)[0]
    assert (x, y) == (1, 1)
    assert value == 1
