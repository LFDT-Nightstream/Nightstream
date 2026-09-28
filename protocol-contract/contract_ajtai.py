"""Check the indexed setup selected in fprime-stage1-main-ajtai-setup.md."""

from __future__ import annotations


def chacha20_block(seed: list[int], row: int, block: int, lane: int) -> list[int]:
    if len(seed) != 32 or any(not 0 <= value < 256 for value in seed):
        raise ValueError("Ajtai seed must contain 32 bytes")
    if not (0 <= row < 2**32 and 0 <= block < 2**64 and 0 <= lane < 2**32):
        raise ValueError("Ajtai index exceeds the RFC 8439 word encoding")
    mask = 2**32 - 1
    initial = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
    initial += [int.from_bytes(bytes(seed[i:i + 4]), "little") for i in range(0, 32, 4)]
    initial += [lane, row, block & mask, block >> 32]
    state = initial.copy()

    def quarter(a: int, b: int, c: int, d: int) -> None:
        for x, y, z, rotation in ((a, b, d, 16), (c, d, b, 12), (a, b, d, 8), (c, d, b, 7)):
            state[x] = (state[x] + state[y]) & mask
            value = state[z] ^ state[x]
            state[z] = ((value << rotation) | (value >> (32 - rotation))) & mask

    for _ in range(10):
        for indices in ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
                        (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
            quarter(*indices)
    return [(value + original) & mask for value, original in zip(state, initial)]


def check_ajtai_setup(config: dict) -> None:
    # Import at the call boundary; contract_checks also calls this module.
    from contract_checks import require

    setup = config["ajtai_setup_v1"]
    commitment = config["commitment_profile"]
    candidate = config["nightstream_candidate"]
    selected = {
        "id": "nightstream-ajtai-chacha20-wide256-v1",
        "rounds": 20,
        "seed_bytes": 32,
        "seed_hex": "fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702",
        "nonce_encoding": "row-u32-le||message-column-u64-le",
        "counter_encoding": "ring-coefficient-u32",
        "coefficient_encoding": "first-eight-u32-little-endian-words-as-one-256-bit-integer",
        "coefficient_reduction": "integer-mod-Goldilocks-prime",
        "rejection_retry_fallback": "absent",
        "matrix_order": "output-row-then-message-column-then-ring-coefficient",
        "selected_output_rows": 22,
        "ring_coefficient_count": 54,
        "message_columns_authority": "verifier-key-relation-artifact-v1",
    }
    for name, value in selected.items():
        require(setup.get(name) == value, f"Ajtai setup {name} differs")
    require(commitment["setup_expander"] == setup["id"], "commitment setup-expander ID differs")
    require(commitment["setup_seed_bytes"] == setup["seed_bytes"], "commitment seed width differs")
    require(commitment["kappa"] == setup["selected_output_rows"], "Ajtai setup row count differs")
    require(
        setup["message_columns"] == commitment["message_ring_columns"]
        == candidate["assignment_ring_columns"],
        "Ajtai setup message width differs",
    )
    require(setup["ring_coefficient_count"] == config["paper_goldilocks"]["phi_degree"], "Ajtai ring width differs")
    test = config["ajtai_rfc8439_test_v1"]
    require(test["seed"] == list(range(32)), "RFC 8439 seed differs")
    require((test["row"], test["block"], test["lane"]) == (0x09000000, 0x4A000000, 1), "RFC 8439 index differs")
    require(
        chacha20_block(test["seed"], test["row"], test["block"], test["lane"]) == test["words"],
        "Ajtai ChaCha20 RFC 8439 test vector differs",
    )
    # This is a deterministic arithmetic check, not a pseudorandomness claim.
    q = int(config["paper_goldilocks"]["q_decimal"])
    wide = sum(word << (32 * index) for index, word in enumerate(test["words"][:8]))
    require(test["coefficient"] == wide % q, "Ajtai 256-bit reduction test vector differs")
