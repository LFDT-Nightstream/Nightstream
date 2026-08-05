#!/usr/bin/env python3
"""Independent reference primitives for protocol-profile validation."""

from __future__ import annotations

from typing import Any


def poseidon2_permute(state: list[int], profile: dict[str, Any]) -> list[int]:
    """Apply the exact selected width-8 Poseidon2 permutation."""
    modulus = int(profile["field_modulus"])
    width = profile["width"]
    if len(state) != width:
        raise ValueError("Poseidon2 state width differs")
    m4 = [[int(value) for value in row.split(",")] for row in profile["external_m4_rows"]]
    external = [
        [m4[row % 4][column % 4] * (2 if row // 4 == column // 4 else 1) for column in range(width)]
        for row in range(width)
    ]
    diagonal = profile["internal_diagonal"]

    def linear(matrix: list[list[int]], values: list[int]) -> list[int]:
        return [
            sum(matrix[row][column] * values[column] for column in range(width)) % modulus
            for row in range(width)
        ]

    def sbox(value: int) -> int:
        return pow(value, 7, modulus)

    values = [value % modulus for value in state]
    values = linear(external, values)
    constants = profile["initial_round_constants"]
    for round_index in range(profile["initial_full_rounds"]):
        offset = round_index * width
        values = [(values[lane] + constants[offset + lane]) % modulus for lane in range(width)]
        values = [sbox(value) for value in values]
        values = linear(external, values)
    for constant in profile["partial_round_constants"]:
        values[0] = sbox((values[0] + constant) % modulus)
        total = sum(values) % modulus
        values = [(total + diagonal[lane] * values[lane]) % modulus for lane in range(width)]
    constants = profile["terminal_round_constants"]
    for round_index in range(profile["terminal_full_rounds"]):
        offset = round_index * width
        values = [(values[lane] + constants[offset + lane]) % modulus for lane in range(width)]
        values = [sbox(value) for value in values]
        values = linear(external, values)
    return values


class FieldDuplex:
    """Small reference implementation of the selected field duplex."""

    def __init__(self, permutation_profile: dict[str, Any], transcript_profile: dict[str, Any]):
        self.permutation_profile = permutation_profile
        self.transcript_profile = transcript_profile
        self.modulus = int(permutation_profile["field_modulus"])
        self.state = [0] * permutation_profile["width"]
        self.cursor = 0
        self.mode = "absorb"

    def _permute(self) -> None:
        self.state = poseidon2_permute(self.state, self.permutation_profile)

    def _begin_absorb(self) -> None:
        if self.mode == "squeeze":
            lane = self.transcript_profile["first_capacity_lane"]
            self.state[lane] = (
                self.state[lane] + self.transcript_profile["ratchet_capacity_value"]
            ) % self.modulus
            self._permute()
            self.cursor = 0
            self.mode = "absorb"

    def absorb(self, values: list[int]) -> None:
        self._begin_absorb()
        rate = self.transcript_profile["rate"]
        for value in values:
            self.state[self.cursor] = (self.state[self.cursor] + value) % self.modulus
            self.cursor += 1
            if self.cursor == rate:
                self._permute()
                self.cursor = 0

    def frame(self, tag: int, payload: list[int]) -> None:
        self.absorb([tag, len(payload), *payload])

    def tagged_squeeze(self, frame_tag: int, squeeze_tag: int, count: int) -> list[int]:
        self.frame(frame_tag, [squeeze_tag, count])
        self.state[self.cursor] = (
            self.state[self.cursor] + self.transcript_profile["absorb_padding_cursor_value"]
        ) % self.modulus
        lane = self.transcript_profile["last_rate_lane"]
        self.state[lane] = (
            self.state[lane] + self.transcript_profile["absorb_padding_last_rate_value"]
        ) % self.modulus
        self._permute()
        self.mode = "squeeze"
        result: list[int] = []
        rate = self.transcript_profile["rate"]
        while len(result) < count:
            result.extend(self.state[:rate])
            if len(result) < count:
                lane = self.transcript_profile["first_capacity_lane"]
                self.state[lane] = (
                    self.state[lane] + self.transcript_profile["continuation_capacity_value"]
                ) % self.modulus
                self._permute()
        return result[:count]
