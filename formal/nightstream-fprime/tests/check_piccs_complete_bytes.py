#!/usr/bin/env python3
"""Compare the complete independently produced PiCCS input, phase and proof words."""

import copy
import json
from pathlib import Path
import sys

P = 18446744069414584321


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, separators=(",", ":")).encode()


def proof_words(value):
    require(len(value) == 7 and value[0] == 2, "wrong PiCCS input schema")
    require(len(value[1]) == 1188 and len(value[3]) == 28, "wrong commitment or round count")
    words = list(value[1])
    for polynomial in value[3]:
        require(len(polynomial) == 10, "wrong polynomial width")
        for coefficient in polynomial:
            require(len(coefficient) == 2, "wrong extension width")
            words.extend(coefficient)
    require(len(value[4]) == len(value[5]) == 17, "wrong output source count")
    for pad, matrices in zip(value[4], value[5]):
        require(len(pad) == 54 and len(matrices) == 14, "wrong output family width")
        for lanes in [pad, *matrices]:
            require(len(lanes) == 54, "wrong coefficient width")
            for coefficient in lanes:
                require(len(coefficient) == 2, "wrong extension width")
                words.extend(coefficient)
    require(len(words) == 29288, "wrong complete proof-input word count")
    require(all(type(word) is int and 0 <= word < P for word in words),
            "noncanonical proof-input field")
    return words


def compare(lean_input_bytes, lean_phase_bytes, lean_words_bytes, rust_input_bytes, rust_phase_bytes):
    require(lean_input_bytes == rust_input_bytes, "complete PiCCS input bytes differ")
    require(lean_phase_bytes == rust_phase_bytes, "complete PiCCS phase bytes differ")
    value, phase = json.loads(lean_input_bytes), json.loads(lean_phase_bytes)
    require(len(phase) == 15 and phase[0] == 1, "PiCCS was not accepted")
    require(len(phase[14]) == 8, "wrong outgoing transcript state width")
    require(phase[12] == value[4] and phase[13] == value[5], "phase output families differ")
    words = proof_words(value)
    require(lean_words_bytes == canonical(words) + b"\n", "complete proof-input encoding differs")
    return value, phase, words


def main():
    require(len(sys.argv) == 6, "expected Lean input, phase, proof words, Rust input and Rust phase")
    paths = list(map(Path, sys.argv[1:]))
    raw = [path.read_bytes() for path in paths]
    value, phase, words = compare(*raw)
    mutated_input, mutated_phase = copy.deepcopy(value), copy.deepcopy(phase)
    changed = (mutated_input[5][0][0][53][1] + 1) % P
    mutated_input[5][0][0][53][1] = changed
    mutated_phase[13][0][0][53][1] = changed
    try:
        compare(*raw[:3], canonical(mutated_input), canonical(mutated_phase) + b"\n")
    except ValueError as error:
        require(str(error) == "complete PiCCS input bytes differ", "unrelated target rejection")
    else:
        raise ValueError("consistent changed final output target was accepted")
    changed_words = list(words)
    changed_words[-1] = (changed_words[-1] + 1) % P
    try:
        compare(raw[0], raw[1], canonical(changed_words) + b"\n", raw[3], raw[4])
    except ValueError as error:
        require(str(error) == "complete proof-input encoding differs", "unrelated encoding rejection")
    else:
        raise ValueError("changed final proof-input word was accepted")
    print(json.dumps({"event": "independent_piccs_complete_bytes_match",
                      "rounds": 28, "phase_fields": 15, "output_field_words": 27540,
                      "proof_input_field_words": len(words), "outgoing_state_words": 8,
                      "input_bytes": len(raw[0]), "phase_bytes": len(raw[1]),
                      "consistent_changed_target": "rejected", "changed_proof_word": "rejected",
                      "scope": "complete PiCCS input and phase encoding plus package proof-input words; full NIFS encoding is separate"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError) as error:
        sys.exit(str(error))
