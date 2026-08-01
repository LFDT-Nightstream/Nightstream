# Crypto

## Purpose
- **What it is**: Off-circuit Poseidon2 hash function over the Goldilocks field (`q = 2^64 - 2^32 + 1`), providing deterministic sponge-based hashing for transcript digests and Fiat-Shamir challenges.
- **Key invariant**: Deterministic -- same input always produces the same output. Uses sponge construction with WIDTH=16, RATE=8 absorption, padding with `+1`, and squeeze from first `DIGEST_LEN` elements.
- **Protocol role**: Generates Fiat-Shamir challenges off-circuit. Neo uses a public-rho embedded verifier architecture where challenges (rho) are computed off-circuit using Poseidon2 transcripts; the embedded verifier circuit only proves multiplication/linearity constraints.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `H: F^* -> F^d` | (not in paper) | `poseidon2_hash(input) -> [Goldilocks; DIGEST_LEN]` | Sponge hash |
| (byte input variant) | (not in paper) | `poseidon2_hash_packed_bytes(input) -> [Goldilocks; DIGEST_LEN]` | 8 bytes/element packing |

## Paper Anchors

Source: ./docs/superneo-paper/

- (No direct paper anchor -- Poseidon2 is transcript infrastructure, not part of the mathematical protocol definitions.)
- Implicit in Section 7 reductions: Fiat-Shamir challenges require a collision-resistant hash.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| (none) | (none) | Poseidon2 implementation is not formalized in Lean |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Constants | `WIDTH` | const | Core | 16 (Poseidon2 state width) |
| Constants | `RATE` | const | Core | 8 (absorption rate per permutation) |
| Constants | `CAPACITY` | const | Core | 8 (sponge capacity) |
| Constants | `DIGEST_LEN` | const | Core | Squeeze length |
| Constants | `SEED` | const | Core | Round constant generation seed |
| Permutation | `PERM` | static | Core | Lazy-initialized cached Poseidon2 permutation |
| Permutation | `permutation()` | fn | Core | Reference to cached permutation |
| Hash | `poseidon2_hash(input)` | fn | Core | Sponge hash over field elements |
| Hash | `poseidon2_hash_packed_bytes(input)` | fn | Core | Byte input with 8-byte packing + length encoding |
| Hash | `poseidon2_hash_bytes(input)` | fn | Helper | Byte input (1 byte/element -- inefficient) |
| Hash | `poseidon2_hash_single(x)` | fn | Helper | Single-element hash |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Determinism: `poseidon2_hash(x) == poseidon2_hash(x)` | Unit test | (none) |
| Domain separation: different inputs produce different outputs | Unit test | (none) |
| `poseidon2_hash_single(x) == poseidon2_hash(&[x])` | Unit test | (none) |
| Packed bytes length encoding: different-length inputs with same prefix are distinguished | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Poseidon2 is collision-resistant over Goldilocks | Cryptographic literature | Standard assumption for algebraic hash functions |
| Round constants from `neo-params` are correct | `neo-params` crate | Single source of truth for all protocol parameters |
| `once_cell::Lazy` initialization is thread-safe | `once_cell` crate | Standard Rust synchronization primitive |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-params`: Poseidon2 round constants and parameters
- `p3-poseidon2`: Poseidon2 permutation implementation
- `p3-symmetric`: Sponge construction
- `p3-goldilocks`: Goldilocks field type
- `once_cell`: Lazy static initialization

Downstream consumers:
- `neo-transcript`: transcript hashing
- `neo-ccs::utils`: `direct_sum_transcript_mixed` uses `poseidon2_hash_packed_bytes`
- `neo-ajtai::prg`: PRG seed expansion

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Poseidon2 correctness is verified via unit tests and parameter conformance |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- Permutation is lazily cached (`Lazy<Poseidon2Goldilocks>`) -- initialization is expensive (~ms)
- All parameters (`WIDTH`, `RATE`, `CAPACITY`, `SEED`, `DIGEST_LEN`) sourced from `neo-params`
- `poseidon2_hash_packed_bytes` appends input length as final element to prevent length-extension

## Acceptance Criteria

- `cargo test -p neo-ccs --release` succeeds
- All crypto invariant tests pass
- Parameters match `neo-params` constants

## Out of Scope

- In-circuit Poseidon2 computation (not needed: public-rho architecture)
- Alternative hash functions (SHA-256, Blake3)
- Merkle tree construction
