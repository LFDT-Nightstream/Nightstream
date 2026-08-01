# Transcript

## Purpose

- **What it is**: A Fiat-Shamir transcript oracle built on Poseidon2 (WIDTH=8, RATE=4 over Goldilocks), plus a transcript-bound CSPRNG for prover randomness.
- **Key invariant**: Label+length framing ensures that different absorption patterns produce cryptographically distinct challenge sequences; domain gates prevent state reuse between absorb and squeeze phases.
- **Protocol role**: Converts the interactive public-coin reductions (Pi_CCS, Pi_RLC, Pi_DEC) into non-interactive proofs. Every verifier challenge in the protocol is derived from a `challenge_field` or `challenge_bytes` call on this transcript.

## Target Formulas (Paper -> Rust)

The transcript is not explicitly formalized in the SuperNeo paper. Fiat-Shamir is treated as a standard background technique. The implementation follows the Merlin-style API pattern with Poseidon2 as the hash function.

| Paper concept | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| Fiat-Shamir oracle | Background technique, implicit in all interactive reductions | `Transcript` trait | 7-method interface |
| Challenge `rho` | Section 7.3, Lemma 3 (Pi_CCS) | `challenge_field(b"chal/rho")` | Random linear combination challenge |
| Protocol phases | Sections 7.3-7.5 | `labels::PI_CCS`, `PI_RLC`, `PI_DEC` | Label constants for each phase |
| Poseidon2 hash | Not in paper (implementation choice) | `Poseidon2Transcript` | Goldilocks-native sponge |

## Paper Anchors

Source: ./docs/superneo-paper/

- Sections 7.3-7.5: Interactive reductions Pi_CCS, Pi_RLC, Pi_DEC are defined as public-coin protocols. The transcript makes them non-interactive via Fiat-Shamir.
- No dedicated transcript formalization in the paper; the construction is a standard cryptographic primitive.

## Lean Cross-Reference

No Lean formalization exists for the transcript. The Lean project formalizes the mathematical reductions assuming an ideal random oracle; the transcript implementation is on the Rust side only.

| Lean spec | Lean module | Relationship |
|---|---|---|
| (none) | (none) | Transcript is Rust-only infrastructure |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Traits | `Transcript` | trait | Core | 7-method Fiat-Shamir oracle interface |
| Traits | `TranscriptProtocol` | trait | Core | 4 structured absorption helpers for CCS/poly/commit data |
| Sponge | `Poseidon2Transcript` | struct | Core | Poseidon2 sponge (WIDTH=8, RATE=4) with domain separation |
| Sponge | `Poseidon2Transcript::new(app_label)` | fn | Core | Initialize with APP_DOMAIN + app_label absorption |
| Sponge | `append_message(label, msg)` | fn | Core | Absorb label bytes + length + message bytes |
| Sponge | `append_fields(label, fs)` | fn | Core | Absorb label bytes + length + field elements (batch) |
| Sponge | `challenge_bytes(label, out)` | fn | Core | Squeeze bytes; absorbs `b"chal/label"` + label, then domain gate (ONE + permute) per 32-byte block |
| Sponge | `challenge_field(label)` | fn | Core | Squeeze one field element; absorbs label, domain gate, returns `st[0]` |
| Sponge | `fork(scope)` | fn | Core | Clone transcript + absorb `b"fork"` + scope for sub-protocol isolation |
| Sponge | `digest32()` | fn | Core | Squeeze 32-byte digest; domain gate (ONE + permute), reads `st[0..4]` |
| Convenience | `challenge_nonzero_field(label)` | fn | Helper | Rejection sampling: squeeze until non-zero |
| Convenience | `append_u64s(label, us)` | fn | Helper | Absorb label + length + u64 values as field elements |
| Convenience | `append_fields_iter(label, len, iter)` | fn | Helper | Streaming absorption with chunked buffering; panics on length mismatch |
| Convenience | `challenge_fields(label, n)` | fn | Helper | Squeeze `n` field elements with shared label |
| RNG | `TranscriptRngBuilder` | struct | Core | Builder binding transcript state + witness data |
| RNG | `TranscriptRngBuilder::from_transcript(tr)` | fn | Core | Capture current sponge state |
| RNG | `TranscriptRngBuilder::from_state(state)` | fn | Helper | Initialize from raw sponge state array |
| RNG | `TranscriptRngBuilder::rekey_with_witness_fields(label, ws)` | fn | Core | Mix witness data into builder state + permute |
| RNG | `TranscriptRngBuilder::finalize(rng)` | fn | Core | Mix 32 bytes external entropy + sentinel + permute -> TranscriptRng |
| RNG | `TranscriptRng` | struct | Core | Streaming CSPRNG from Poseidon2 permutation |
| RNG | `TranscriptRng::fill_bytes(out)` | fn | Core | Fill buffer with pseudo-random bytes (permute per 32-byte block) |
| RNG | `TranscriptRng::field()` | fn | Core | Generate one random field element from 8 random bytes |
| Labels | `labels::CCS_HEADER` | const | Helper | `b"ccs/header"` |
| Labels | `labels::CCS_DIMS` | const | Helper | `b"ccs/dims"` |
| Labels | `labels::POLY_SPARSE` | const | Helper | `b"poly/sparse"` |
| Labels | `labels::POLY_LEN` | const | Helper | `b"poly/len"` |
| Labels | `labels::POLY_COEFF` | const | Helper | `b"poly/coeff"` |
| Labels | `labels::POLY_EXPS` | const | Helper | `b"poly/exps"` |
| Labels | `labels::COMMIT_COORDS` | const | Helper | `b"commit/coords"` |
| Labels | `labels::ACC_DIGEST` | const | Helper | `b"acc/digest"` |
| Labels | `labels::STEP_DIGEST` | const | Helper | `b"step/digest"` |
| Labels | `labels::CHAL_RHO` | const | Helper | `b"chal/rho"` |
| Labels | `labels::PI_CCS` | const | Helper | `b"phase/pi-ccs"` |
| Labels | `labels::PI_RLC` | const | Helper | `b"phase/pi-rlc"` |
| Labels | `labels::PI_DEC` | const | Helper | `b"phase/pi-dec"` |
| Labels | `labels::EV` | const | Helper | `b"ev"` |
| Labels | `labels::STEP` | const | Helper | `b"ivc/step"` |
| Debug | `debug::Event` | struct | Helper | Transcript operation log entry (op, label, len, state prefix) |
| Debug | `fs_guard::reset(tag)` | fn | Helper | Clear global event log (feature-gated: `fs-guard`) |
| Debug | `fs_guard::record(evt)` | fn | Helper | Push event to global log (feature-gated: `fs-guard`) |
| Debug | `fs_guard::take()` | fn | Helper | Take all recorded events (feature-gated: `fs-guard`) |
| Debug | `fs_guard::first_mismatch(spec, actual)` | fn | Helper | Find first event divergence between spec and actual traces |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Framing: different label/msg splits produce different digests | Unit test (`framing_distinguishes_splits`) | (none) |
| Label sensitivity: different challenge labels produce different challenges | Unit test (`label_changes_challenge`) | (none) |
| Fork isolation: different fork scopes produce different challenge sequences | Unit test (`fork_isolated`) | (none) |
| Determinism: identical transcript operations produce identical outputs | Unit test (`determinism_identical_operations`) | (none) |
| Domain separation: different app labels produce different challenge sequences | Unit test (`domain_separation_app_labels`) | (none) |
| Domain gate: squeeze absorbs `Goldilocks::ONE` before permuting | Unit test (`domain_gate_squeeze_changes_output`) | (none) |
| `challenge_nonzero_field` never returns zero | Unit test (`challenge_nonzero_field_never_zero`) | (none) |
| RNG binding: different witness/entropy produce different RNG output | Unit test (`rng_binding_changes_on_inputs`) | (none) |
| RNG determinism: identical inputs produce identical RNG output | Unit test (`rng_determinism_same_inputs`) | (none) |
| `append_fields_iter` panics on iterator length mismatch | Unit test (`append_fields_iter_length_mismatch_panics`) | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Poseidon2 is a secure permutation over Goldilocks | Cryptographic assumption | Standard; Poseidon2 parameters from `neo-ccs::crypto::poseidon2_goldilocks` |
| Fiat-Shamir heuristic is sound for public-coin protocols | Standard model assumption | Well-established in the random oracle model |
| Sponge construction with domain separation provides collision resistance | Cryptographic assumption | Follows the sponge framework (Bertoni et al.) |
| `challenge_field` output is uniform over F_q | Poseidon2 indistinguishability | Single limb `st[0]` is the full field element (64-bit Goldilocks) |
| External entropy from `finalize(rng)` is cryptographically random | Caller responsibility | Builder mixes 32 bytes from caller-provided CSPRNG |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `F` type alias (Goldilocks field element)
- `neo-ccs::crypto::poseidon2_goldilocks`: Poseidon2 permutation instance, `WIDTH`, `RATE`, `DIGEST_LEN` constants
- `p3-field`: `PrimeCharacteristicRing`, `PrimeField64` traits
- `p3-goldilocks`: `Goldilocks` type, `Poseidon2Goldilocks` permutation type
- `p3-symmetric`: `Permutation` trait
- `rand`, `rand_chacha`: `RngCore`, `CryptoRng` for TranscriptRng finalization

Downstream consumers:
- `neo-reductions`: all engines (pi_ccs, pi_rlc_dec, sumcheck) use `Transcript` + `TranscriptProtocol` for challenge derivation
- `neo-fold-clean`: proof construction, verification, and public-digest binding use transcript for Fiat-Shamir

## Lean Oracle Conformance

No lean_oracles exist for the transcript. The transcript is Rust-only infrastructure without a Lean formalization.

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | (none) |

## Quality Expectations

- `#![forbid(unsafe_code)]` (enforced crate-wide)
- No `todo!()` or `unimplemented!()` in the contract surface
- All absorption methods include label+length framing (prevents ambiguous splits)
- Domain gate (absorb `Goldilocks::ONE` + permute) before every squeeze operation
- `debug::Event` and `fs_guard` are feature-gated (`debug-log`, `fs-guard`) — zero cost in production builds
- `TranscriptRng` must not implement `CryptoRng` (it is transcript-bound, not general-purpose)

## Acceptance Criteria

- `cargo test -p neo-transcript --release` succeeds (runs both `tests/` and `spec-tests/`)
- All 10 invariant obligations have at least one test in `spec-tests/`
- `cargo clippy -p neo-transcript --all-targets --release -- -D warnings` is clean
- No panics on well-formed inputs

## Out of Scope

- Poseidon2 permutation correctness (belongs to `neo-ccs::crypto`)
- Wire-level serialization of transcript state
- Constant-time guarantees (transcript operates on public data only)
- `debug-log` and `fs-guard` feature internals (debug/audit tooling, not security-critical)
- Alternative hash function backends (only Poseidon2 is supported)
