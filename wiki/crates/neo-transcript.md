# neo-transcript

Poseidon2 Fiat-Shamir transcript — the only randomness oracle in protocol-binding
paths. `#![forbid(unsafe_code)]`.

## Owns

- **`Transcript` trait** — byte-first, Merlin-inspired: `new(app_label)`,
  `append_message` / `append_fields`, `challenge_bytes` / `challenge_field(s)`,
  `fork(scope)` (domain-separated sub-transcript), `digest32`.
- **`TranscriptProtocol`** — typed absorbs for protocol objects: CCS headers, sparse
  polynomials, commitment coordinates, public fields.
- **`Poseidon2Transcript`** — the production implementation, built on
  `neo_params::poseidon2_goldilocks`.
- **`labels`** — the static label namespace. Every absorb/challenge call carries a
  `&'static [u8]` label; transcript audits grep this module.
- **`TranscriptRng` / `TranscriptRngBuilder`** — transcript-bound RNG for prover-side
  sampling.

## Features

- `fs-guard` — runtime Fiat-Shamir misuse guard (e.g. challenge before absorb) for
  tests.
- `debug-log` — transcript event logging for divergence debugging.

## Contract

The binding requirements for the selected path—exactly which data must be
absorbed before each challenge—are defined by
[NS-TRANSCRIPT-ORDER](../../protocol-contract/src/normative/80-nightstream-verifier.md#ns-transcript-order--fold-transcript-schedule).
The active Lean transcript model provides the formal authority. Rust framing
and red-team checks live in `crates/neo-transcript/tests`.
See [Transcript & digests](../protocol/transcript-and-digests.md).
